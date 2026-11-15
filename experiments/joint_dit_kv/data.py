"""Shared JSONL contract for joint DiT, text baselines and official LatentMAS."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Example:
    id: str
    question: str
    agents: list[str]
    answer: str
    choices: list[str] = field(default_factory=list)

    def as_json(self):
        return {"id": self.id, "question": self.question, "agents": self.agents,
                "answer": self.answer, "choices": self.choices}


def _as_text(value):
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def normalize(row: dict, max_agents: int = 8) -> Example:
    # HiddenBench native benchmark.json format is evaluation-only.
    if "hidden_information" in row:
        agent_texts = []
        for hidden in row.get("hidden_information", []):
            shared = _as_text(row.get("shared_information", []))
            agent_texts.append("Context: " + str(row.get("description", "")) +
                               "\nShared: " + shared + "\nPrivate: " + _as_text(hidden))
        choices = [str(x) for x in row.get("possible_answers", [])]
        question = str(row.get("description", ""))
        if choices:
            question += "\nOptions: " + " / ".join(choices)
        row = {
            "id": row.get("id", ""),
            "question": question,
            "agents": agent_texts,
            "answer": row.get("correct_answer", ""),
            "choices": choices,
        }
    agents = row.get("agents") or row.get("agent_contexts")
    if not isinstance(agents, list):
        raise ValueError("agents must be a list of private evidence strings")
    agents = [_as_text(x.get("text", x) if isinstance(x, dict) else x) for x in agents]
    question = str(row.get("question", row.get("query", ""))).strip()
    answer = str(row.get("answer", row.get("gold_answer", ""))).strip()
    if not (question and answer and 2 <= len(agents) <= max_agents):
        raise ValueError("Expected question, answer and 2..max_agents evidence strings")
    return Example(
        id=str(row.get("id", hashlib.sha256((question + answer).encode()).hexdigest()[:16])),
        question=question, agents=agents, answer=answer,
        choices=[str(x) for x in row.get("choices", [])],
    )


def load_examples(path: str | Path, max_agents: int = 8) -> list[Example]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    else:
        raw = json.loads(path.read_text())
        rows = raw if isinstance(raw, list) else raw["examples"]
    result = [normalize(r, max_agents=max_agents) for r in rows]
    if len({x.id for x in result}) != len(result):
        raise ValueError("Repeated example IDs found")
    return result


def check_disjoint(train: list[Example], dev: list[Example]):
    train_q = {hashlib.sha256(e.question.strip().lower().encode()).hexdigest() for e in train}
    dev_q = {hashlib.sha256(e.question.strip().lower().encode()).hexdigest() for e in dev}
    if train_q.intersection(dev_q):
        raise ValueError("Question overlap between train and dev; split by task before fitting")


def exact_match(predicted: str, gold: str, choices: list[str] | None = None) -> bool:
    """Conservative first-line scorer; use each benchmark's official scorer in paper."""
    answer = predicted.strip().split("\n")[0]
    answer = re.sub(r"^(answer\s*[:\-]\s*|the answer is\s+)", "", answer, flags=re.I).strip()
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    p, g = norm(answer), norm(gold)
    if not (p and g):
        return False
    if choices:
        found = [c for c in choices if re.search(r"\b" + re.escape(norm(c)) + r"\b", p)]
        if len(found) == 1:
            return norm(found[0]) == g
        # Exact one-letter response only, not substring of a paragraph.
        if len(p) == 1 and p.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            ix = ord(p.upper()) - 65
            return ix < len(choices) and norm(choices[ix]) == g
        return False
    return p == g


def make_toy(directory: str | Path, train_count: int = 256, dev_count: int = 64):
    """Mechanism smoke data only. Not a publishable benchmark or LatentMAS comparison."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    colors = ["blue", "green", "red", "yellow", "purple", "orange"]
    rng = random.Random(1927)

    def create(idx):
        # Crucially, no single agent contains the full join.
        case = "CASE" + str(100000 + idx)
        key = "K" + str(rng.randint(10000, 99999))
        locker = "L" + str(rng.randint(10000, 99999))
        distractor = "L" + str(rng.randint(10000, 99999))
        color = rng.choice(colors)
        other = rng.choice([x for x in colors if x != color])
        return Example(
            id="toy-" + str(idx),
            question="What color is stored at the locker connected to " + case +
                     "? Answer with exactly one color word.",
            agents=[
                "Record A: " + case + " is assigned key " + key + ".",
                "Record B: key " + key + " opens locker " + locker + ".",
                "Record C: locker " + locker + " stores color " + color + ".",
                "Record D: locker " + distractor + " stores color " + other + ".",
            ],
            answer=color,
            choices=colors,
        )

    all_rows = [create(i) for i in range(train_count + dev_count)]
    for name, subset in (("train", all_rows[:train_count]),
                         ("dev", all_rows[train_count:])):
        with (root / (name + ".jsonl")).open("w") as fp:
            for e in subset:
                fp.write(json.dumps(e.as_json()) + "\n")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-toy", required=True)
    ap.add_argument("--train", type=int, default=256)
    ap.add_argument("--dev", type=int, default=64)
    args = ap.parse_args()
    print(make_toy(args.make_toy, args.train, args.dev))


if __name__ == "__main__":
    main()
