"""Distil the LLM Router into a classifier over question embeddings.

    .venv/bin/python -m scripts.train_router_student           # generate, label, train
    .venv/bin/python -m scripts.train_router_student --train   # retrain from saved data

Teacher: the LLM Router itself (`classify`, qwen2.5:7b). It labels synthetic employee
questions, and a question is kept only when the teacher's label agrees with the route
the question was generated for — disagreement means ambiguous, which is exactly what
the student should not learn to be confident about.

Student: multinomial logistic regression on the mxbai embedding of the question. At
request time it costs one embedding (~40ms) instead of a 613-token LLM prompt (~1.6s),
returns real probabilities, and cannot be steered by instructions in the question.

Held out: any generated question too similar to one in evals/cases.py is dropped, so
the Router evals still measure generalisation.
"""

import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.agents.router import SELECTABLE_ROUTES, classify
from src.config import get_settings
from src.llm.factory import get_chat_model, get_embeddings

DATA = Path("evals/data/router_training.jsonl")
MODEL = Path("src/agents/router_student.json")

# What each route is for, phrased as generation briefs. Broad on purpose: the student
# must cover questions nobody wrote an eval for.
BRIEFS = {
    "rag": [
        "refund and chargeback policy", "payment outages and incident write-ups",
        "customer PII and data handling rules", "HR benefits, leave and payroll policy",
        "laptops, devices, VPN and IT access", "AML and compliance procedures",
        "information security policy", "onboarding and training for new staff",
        "travel and expense claims", "customer support processes and escalation paths",
        "engineering on-call and runbooks", "office facilities and workplace rules",
    ],
    "sql": [
        "counts of transactions in a month or quarter",
        "total value of payments or refunds, optionally for one department",
        "average transaction amount over a period",
        "how many transactions were flagged for AML in a period",
        "comparisons of transaction volumes between months or departments",
    ],
    "clarify": [
        "questions that only point at something without naming it, like 'that issue' or 'the other one'",
        "follow-ups that depend on an earlier conversation the assistant never saw",
        "requests about 'it', 'this' or 'they' with no subject anywhere in the question",
        "status checks on an unnamed thing, like 'any update?' or 'is that fixed yet?'",
    ],
}

GENERATE = """Write 25 different questions an employee at Aurelia Financial, a Singapore
fintech, might type into an internal knowledge assistant. Topic: {brief}.

Vary the phrasing, length and tone: some formal, some casual, some with typos, some long
and rambling. Output one question per line, with no numbering and nothing else."""


def _generate(rng: random.Random) -> list[tuple[str, str]]:
    from langchain_community.chat_models import ChatOllama

    s = get_settings()
    writer = ChatOllama(model=s.ollama_model, base_url=s.ollama_base_url, temperature=0.9)
    out = []
    for route, briefs in BRIEFS.items():
        for brief in briefs:
            text = str(writer.invoke(GENERATE.format(brief=brief)).content)
            lines = [ln.strip(" -*•\t0123456789.") for ln in text.splitlines()]
            out += [(route, q) for q in lines if len(q) > 8 and q.endswith("?")]
            print(f"  {route:<8} {brief[:50]:<50} {len(out)}", flush=True)
    rng.shuffle(out)
    return out


def _eval_questions() -> list[str]:
    from evals.cases import ROUTER_ADVERSARIAL, ROUTER_CASES, ROUTER_UNANSWERABLE

    return [q for q, _ in ROUTER_CASES] + [q for q, _ in ROUTER_UNANSWERABLE] + [
        q for q, _, _ in ROUTER_ADVERSARIAL
    ]


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def build_dataset() -> None:
    rng = random.Random(13)
    embeddings = get_embeddings()
    chat = get_chat_model()

    print("generating")
    candidates = _generate(rng)
    print(f"labelling {len(candidates)} with the teacher")
    kept = []
    for i, (intended, q) in enumerate(candidates, 1):
        label = classify(q, chat_model=chat)
        if label == intended:
            kept.append((q, label))
        if i % 50 == 0:
            print(f"  {i}/{len(candidates)} kept {len(kept)}", flush=True)

    vectors = _unit(np.array(embeddings.embed_documents([q for q, _ in kept])))
    held_out = _unit(np.array(embeddings.embed_documents(_eval_questions())))
    near_eval = (vectors @ held_out.T).max(axis=1) >= 0.90

    rows, seen = [], []
    for (q, label), v, leak in zip(kept, vectors, near_eval, strict=True):
        if leak or (seen and max(float(v @ s) for s in seen) >= 0.97):
            continue
        seen.append(v)
        rows.append({"query": q, "route": label})

    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {r: sum(1 for x in rows if x["route"] == r) for r in SELECTABLE_ROUTES}
    print(f"kept {len(rows)} of {len(candidates)} {counts} "
          f"({int(near_eval.sum())} dropped as too close to an eval question)")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(x: np.ndarray, y: np.ndarray, classes: int, l2: float = 1e-3, steps: int = 3000):
    """Multinomial logistic regression, full-batch gradient descent. Small enough not
    to need a library: a few hundred rows of 1024 features."""
    w = np.zeros((x.shape[1], classes))
    b = np.zeros(classes)
    onehot = np.eye(classes)[y]
    # Balance classes, so the rarer clarify examples aren't drowned out.
    weight = (len(y) / (classes * np.bincount(y, minlength=classes)))[y][:, None]
    lr = 1.0
    for _ in range(steps):
        grad = (_softmax(x @ w + b) - onehot) * weight / len(y)
        w -= lr * (x.T @ grad + l2 * w)
        b -= lr * grad.sum(axis=0)
    return w, b


def train() -> None:
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    labels = list(SELECTABLE_ROUTES)
    x = _unit(np.array(get_embeddings().embed_documents([r["query"] for r in rows])))
    y = np.array([labels.index(r["route"]) for r in rows])

    order = np.random.default_rng(0).permutation(len(y))
    cut = int(len(y) * 0.8)
    train_idx, val_idx = order[:cut], order[cut:]
    w, b = fit(x[train_idx], y[train_idx], len(labels))
    p = _softmax(x[val_idx] @ w + b)
    val_acc = float((p.argmax(axis=1) == y[val_idx]).mean())

    w, b = fit(x, y, len(labels))  # final model on everything
    MODEL.write_text(json.dumps({
        "labels": labels,
        "embedding_model": get_settings().ollama_embedding_model,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(rows),
        "validation_accuracy": round(val_acc, 4),
        "bias": b.round(6).tolist(),
        "weights": w.round(6).tolist(),
    }))
    print(f"validation accuracy {val_acc:.3f} on {len(val_idx)} held-out rows -> {MODEL}")


if __name__ == "__main__":
    if "--train" not in sys.argv:
        build_dataset()
    train()
