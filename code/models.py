"""Lazy neural dependencies and shared input/scoring contracts."""

from __future__ import annotations

import inspect
import math
import re
import sys
from collections import Counter

from .data import LABELS, digest, file_hash, observed_rule_text, policy_state, read_json
from .metrics import class_prediction, rank_prediction

NEURAL = ("dual", "scene", "cbert", "learned_prototypes", "cross")
RETRIEVERS = ("dual", "learned_prototypes", "cross", "tfidf")


def words(text):
    return Counter(re.findall(r"[a-z0-9_]+", text.lower()))


def rule_inputs(candidates, config):
    if not config.get("anonymous_rules", False):
        return [p["text"] for p in candidates]
    index = {cid: i for i, cid in enumerate(config["candidate_ids"])}
    if any(p["semantic_id"] not in index for p in candidates):
        raise ValueError("Anonymous dual cannot introduce an unseen semantic ID")
    return [f"candidate item {index[p['semantic_id']]:04d}" for p in candidates]


class BasicScorer:
    def __init__(self, method, train, candidates, policy):
        self.method, self.policy = method, policy
        self.metadata = {"method": method, "training_reference": "train_only",
                         "training_rows": len(train), "train_digest": digest(train)}
        self.lookup = {}
        self.prior = Counter(r["true_label"] for r in train)
        for r in train:
            self.lookup.setdefault(r["scene_text"], Counter())[r["true_label"]] += 1
        self.signatures = [(set(words(text)), counts) for text, counts in sorted(self.lookup.items())]
        documents = [words(r["scene_text"]) for r in train] + [words(p["text"]) for p in candidates]
        df = Counter(term for doc in documents for term in doc)
        self.idf = {term: math.log((1 + len(documents)) / (1 + count)) + 1 for term, count in df.items()}
        self.default_idf = math.log(1 + len(documents)) + 1

    def vector(self, text):
        counts = words(text)
        values = {t: (1 + math.log(n)) * self.idf.get(t, self.default_idf) for t, n in counts.items()}
        norm = math.sqrt(sum(v * v for v in values.values())) or 1
        return {t: v / norm for t, v in values.items()}

    def score(self, rows, candidates):
        output = []
        vectors = [self.vector(p["text"]) for p in candidates]
        for r in rows:
            if self.method == "tfidf":
                q = self.vector(r["scene_text"])
                scores = [sum(v * q.get(t, 0) for t, v in vec.items()) for vec in vectors]
                pred, ranked = rank_prediction(scores, candidates)
            else:
                if self.method == "rule":
                    label = policy_state(r["relations"], self.policy)[0]
                    logits = [0.0 if x == label else -50.0 for x in LABELS]
                elif self.method == "lookup":
                    counts = self.lookup.get(r["scene_text"])
                    if counts is None:
                        tokens = set(words(r["scene_text"]))
                        nearest = sorted(((len(tokens & ts) / max(1, len(tokens | ts)), cs)
                                          for ts, cs in self.signatures), key=lambda item: -item[0])[:3]
                        counts = Counter()
                        for similarity, cs in nearest:
                            for label, n in cs.items():
                                counts[label] += similarity * n / sum(cs.values())
                        if not counts:
                            counts = self.prior
                    logits = [math.log(counts[x] + 0.001) for x in LABELS]
                else:
                    raise ValueError("Unknown basic method")
                pred, ranked = class_prediction(logits), None
            output.append({**pred, "ranking": ranked})
        return output


def build_model(config, project, cache):
    import torch
    from torch import nn
    from transformers import AutoModelForSequenceClassification

    if not (project / "src/modeling.py").is_file():
        raise FileNotFoundError("--hazard-project-root must contain src/modeling.py")
    if str(project.resolve()) not in sys.path:
        sys.path.insert(0, str(project.resolve()))
    from src.modeling import DualEncoderModel, TextEncoder, ProjectionHead

    method = config["method"]
    base = config["pretrained_model"]
    options = {"cache_dir": str(cache) if cache else None, "local_files_only": True}
    if method in ("scene", "cbert", "cross"):
        return AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=1 if method == "cross" else len(LABELS),
            hidden_dropout_prob=config.get("dropout", 0.1),
            attention_probs_dropout_prob=config.get("dropout", 0.1), **options)
    params = {key: config[key] for key in ("projection_hidden_dim", "projection_dim", "dropout", "share_backbone")}
    if method == "dual":
        return DualEncoderModel(base, **params, **options)

    class LearnedPrototypeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scene_encoder = TextEncoder(base, **options)
            self.scene_projection = ProjectionHead(self.scene_encoder.hidden_size,
                                                  params["projection_hidden_dim"], params["projection_dim"], params["dropout"])
            self.scene_projection_norm = nn.LayerNorm(params["projection_dim"])
            self.prototypes = nn.Embedding(len(config["candidate_ids"]), params["projection_dim"])
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        def encode_scene(self, inputs):
            features = self.scene_encoder(inputs["input_ids"], inputs["attention_mask"])
            return torch.nn.functional.normalize(self.scene_projection_norm(self.scene_projection(features)), dim=-1)

    if method != "learned_prototypes":
        raise ValueError("Unsupported model")
    return LearnedPrototypeModel()


class NeuralScorer:
    def __init__(self, config, args, policy, load=True):
        import torch
        from transformers import AutoTokenizer

        self.torch, self.config, self.policy = torch, config, policy
        self.method, self.device = config["method"], args.device
        self.batch_size, self.pair_batch_size = args.batch_size, args.pair_batch_size
        self.max_length = int(config.get("max_length", 96))
        self.pair_max_length = int(config.get("pair_max_length", 192))
        cache = args.model_cache_dir or config.get("model_cache_dir")
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model"], cache_dir=str(cache) if cache else None,
                                                      local_files_only=True)
        if config.get("tokenizer_vocab_sha256") and digest(self.tokenizer.get_vocab()) != config["tokenizer_vocab_sha256"]:
            raise ValueError("Tokenizer vocabulary differs from training")
        self.model = build_model(config, args.hazard_project_root, cache).to(self.device)
        if load:
            options = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                options["weights_only"] = False
            state = torch.load(str(args.checkpoint), **options)
            state = state.get("model", state)
            state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.cached_key, self.cached_vectors = None, None
        self.rule_vector_bank = {}
        self.truncations = {"scene": 0, "rule": 0, "pair": 0}
        self.encoded_texts = {"scene": 0, "rule": 0, "pair": 0}
        self.audit_lengths = True
        self.metadata = {"method": self.method, "checkpoint_sha256": file_hash(args.checkpoint) if load else None,
                         "train_config": config, "parameters": sum(p.numel() for p in self.model.parameters()),
                         "max_length": self.max_length, "pair_max_length": self.pair_max_length,
                         "cbert_rule_selection": "observed_evidence_context_not_clean_context" if self.method == "cbert" else None}

    def tokens(self, texts, side, pairs=None):
        maximum = self.pair_max_length if pairs is not None else self.max_length
        self.encoded_texts[side] += len(texts)
        if self.audit_lengths:
            lengths = self.tokenizer(texts, text_pair=pairs, padding=False, truncation=False)["input_ids"]
            self.truncations[side] += sum(len(ids) > maximum for ids in lengths)
        tokens = self.tokenizer(texts, text_pair=pairs, padding=True, truncation=True,
                                max_length=maximum, return_tensors="pt")
        return {k: v.to(self.device) for k, v in tokens.items()}

    def encode(self, texts, side):
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            tokens = self.tokens(texts[i:i + self.batch_size], side)
            encoder = self.model.encode_scene if side == "scene" else self.model.encode_rule
            chunks.append(encoder(tokens))
        return self.torch.cat(chunks)

    def candidate_vectors(self, candidates):
        torch = self.torch
        if self.method == "learned_prototypes":
            index = {cid: i for i, cid in enumerate(self.config["candidate_ids"])}
            if any(p["semantic_id"] not in index for p in candidates):
                raise ValueError("Learned prototype control cannot introduce an unseen semantic ID")
            ids = torch.tensor([index[p["semantic_id"]] for p in candidates], device=self.device)
            return torch.nn.functional.normalize(self.model.prototypes(ids), dim=-1)
        inputs = rule_inputs(candidates, self.config)
        if self.model.training:
            return self.encode(inputs, "rule")
        texts = list(dict.fromkeys(inputs))
        missing = [text for text in texts if text not in self.rule_vector_bank]
        if missing:
            vectors = self.encode(missing, "rule")
            self.rule_vector_bank.update(zip(missing, vectors))
        return torch.stack([self.rule_vector_bank[text] for text in inputs])

    def clear_cache(self):
        self.cached_key, self.cached_vectors = None, None
        self.rule_vector_bank.clear()

    def score_tensor(self, rows, candidates):
        torch = self.torch
        texts = [r["scene_text"] for r in rows]
        if self.method in ("scene", "cbert"):
            rules = [observed_rule_text(r["relations"], self.policy) for r in rows] if self.method == "cbert" else None
            tokens = self.tokens(texts, "pair" if rules is not None else "scene", rules)
            return self.model(**tokens).logits
        if self.method == "cross":
            left = [text for text in texts for _ in candidates]
            right = [p["text"] for _ in texts for p in candidates]
            scores = []
            for i in range(0, len(left), self.pair_batch_size):
                tokens = self.tokens(left[i:i + self.pair_batch_size], "pair", right[i:i + self.pair_batch_size])
                scores.append(self.model(**tokens).logits.reshape(-1))
            return torch.cat(scores).reshape(len(rows), len(candidates))
        key = digest(candidates)
        if self.model.training:
            vectors = self.candidate_vectors(candidates)
        else:
            if key != self.cached_key:
                self.cached_vectors = self.candidate_vectors(candidates)
                self.cached_key = key
            vectors = self.cached_vectors
        queries = self.encode(texts, "scene")
        if self.method == "dual":
            return self.model.compute_similarity(queries, vectors)
        return self.model.logit_scale.exp().clamp(max=100) * (queries @ vectors.T)

    def score(self, rows, candidates):
        output = []
        with self.torch.no_grad():
            for i in range(0, len(rows), self.batch_size):
                scores = self.score_tensor(rows[i:i + self.batch_size], candidates).cpu().tolist()
                for values in scores:
                    pred, ranked = (rank_prediction(values, candidates) if self.method in RETRIEVERS
                                    else (class_prediction(values), None))
                    output.append({**pred, "ranking": ranked})
        return output


def load_scorer(args, train, candidates, policy):
    if args.method not in NEURAL:
        return BasicScorer(args.method, train, candidates, policy)
    if not args.checkpoint or not args.train_config:
        raise ValueError("Neural evaluation requires --checkpoint and --train-config")
    config = read_json(args.train_config)
    recorded = config.get("method", "dual")
    if recorded != args.method:
        raise ValueError("Method differs from checkpoint training configuration")
    if config.get("labels", LABELS) != LABELS:
        raise ValueError("Checkpoint has another label space")
    if config.get("candidate_family", "coarse") != args.family:
        raise ValueError("Checkpoint was trained on another candidate family")
    config["method"] = recorded
    for key, value in {"projection_hidden_dim": 512, "projection_dim": 256, "dropout": 0.1,
                       "share_backbone": False}.items():
        config.setdefault(key, value)
    return NeuralScorer(config, args, policy)


def training_loss(scorer, rows, candidates, family, alignment="balanced_bce"):
    torch = scorer.torch
    scores = scorer.score_tensor(rows, candidates)
    targets = torch.tensor([r["true_label_id"] for r in rows], device=scorer.device)
    if scorer.method in ("scene", "cbert"):
        return torch.nn.functional.cross_entropy(scores, targets)
    positive = torch.tensor([[p["semantic_id"] in r["relevant"][family] for p in candidates] for r in rows],
                            dtype=torch.bool, device=scorer.device)
    if not positive.any(dim=1).all() or positive.all(dim=1).any():
        raise ValueError("Every training row needs positive and negative candidates")
    groups = [torch.tensor([p["major_label"] == label for p in candidates], device=scorer.device) for label in LABELS]
    logits = torch.stack([scores[:, group].max(dim=1).values if group.any() else scores.new_full((len(rows),), -1e9)
                          for group in groups], dim=1)
    classification_loss = torch.nn.functional.cross_entropy(logits, targets)
    if alignment == "multi_positive":
        log_probs = torch.nn.functional.log_softmax(scores, dim=1)
        align = -torch.logsumexp(log_probs.masked_fill(~positive, -torch.inf), dim=1).mean()
    elif alignment == "balanced_bce":
        pos = (-torch.nn.functional.logsigmoid(scores) * positive).sum(1) / positive.sum(1)
        neg = (-torch.nn.functional.logsigmoid(-scores) * ~positive).sum(1) / (~positive).sum(1)
        align = ((pos + neg) / 2).mean()
    else:
        raise ValueError("Unknown alignment loss")
    return classification_loss + align
