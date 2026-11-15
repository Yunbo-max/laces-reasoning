"""Make independent multi-agent training/dev examples from HotpotQA TRAIN split.

Retrieval uses only question/doc strings. Supporting-fact labels are used ONLY
to report source coverage, never to select the agent documents. Official
HotpotQA validation/test are not used by this utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

STOP = {"the", "a", "an", "of", "at", "in", "on", "was", "are", "is", "were",
        "to", "with", "by", "what", "when", "where", "who", "which", "how",
        "does", "did", "and", "or", "for", "from", "than", "into", "between"}


def terms(text):
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if t not in STOP and len(t) >= 2}


def standardize_context(record):
    context = record["context"]
    if isinstance(context, dict):
        return [(str(title), list(sentences))
                for title, sentences in zip(context["title"], context["sentences"])]
    return [(str(entry[0]), [str(s) for s in entry[1]]) for entry in context]


def selected_documents(record, agents=4, sentences_per_agent=3):
    q = terms(record["question"])
    docs = standardize_context(record)
    scored = []
    for index, (title, sentences) in enumerate(docs):
        title_overlap = len(terms(title) & q)
        body_overlap = len(terms(" ".join(sentences)) & q)
        scored.append((2 * title_overlap + body_overlap, -index, index, title, sentences))
    # No answer or supporting_fact labels enter the retrieval ranking.
    top = sorted(scored, reverse=True)[:agents]
    if len(top) < 2:
        raise ValueError("Too few candidate documents")
    messages = []
    titles = []
    for _, __, index, title, sentences in top:
        if not sentences:
            chosen = []
        else:
            indexed = [(len(terms(s) & q), -j, j) for j, s in enumerate(sentences)]
            best = sorted(indexed, reverse=True)[:sentences_per_agent]
            chosen = [sentences[j] for j in sorted(x[2] for x in best)]
        messages.append("Document title: " + title + "\n" + " ".join(chosen))
        titles.append(title)
    return messages, titles


def supported_titles(record):
    supp = record.get("supporting_facts", {})
    if isinstance(supp, dict):
        return set(supp.get("title", []))
    return {str(x[0]) for x in supp}


def build_example(record, agents, sentences_per_agent):
    messages, selected_titles = selected_documents(
        record, agents=agents, sentences_per_agent=sentences_per_agent)
    answer = str(record.get("answer", "")).strip()
    question = str(record.get("question", "")).strip()
    if not (answer and question):
        return None
    selected_support = supported_titles(record)
    example = {
        "id": str(record.get("id", record.get("_id", ""))),
        "question": question, "agents": messages, "answer": answer,
        "choices": [],
    }
    return example, len(selected_support & set(selected_titles)), len(selected_support)


def source_iter(args):
    if args.local_json:
        raw = json.loads(Path(args.local_json).read_text())
        if not isinstance(raw, list):
            raise ValueError("--local-json must be official HotpotQA train array")
        yield from raw
    else:
        from datasets import load_dataset
        # Explicitly TRAIN ONLY; never train from official validation/test.
        yield from load_dataset("hotpotqa/hotpot_qa", "distractor", split="train",
                                streaming=True)


def split_key(example_id):
    # Deterministic document/question-level holdout, independent of iteration order.
    return int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) % 10000


def convert(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    out_paths = {"train": root / "train.jsonl", "dev": root / "dev.jsonl"}
    stats = {"input": 0, "written_train": 0, "written_dev": 0,
             "support_docs_selected": 0, "support_docs_total": 0, "skipped": 0}
    seen = set()
    with out_paths["train"].open("w") as train_f, out_paths["dev"].open("w") as dev_f:
        writers = {"train": train_f, "dev": dev_f}
        for record in source_iter(args):
            stats["input"] += 1
            extracted = build_example(record, args.agents, args.sentences_per_agent)
            if extracted is None:
                stats["skipped"] += 1
                continue
            example, selected, possible = extracted
            if not example["id"] or example["id"] in seen:
                stats["skipped"] += 1
                continue
            seen.add(example["id"])
            dest = "dev" if split_key(example["id"]) < args.holdout_percent * 100 else "train"
            writers[dest].write(json.dumps(example, ensure_ascii=False) + "\n")
            stats["written_" + dest] += 1
            stats["support_docs_selected"] += selected
            stats["support_docs_total"] += possible
            if args.limit and stats["input"] >= args.limit:
                break
    stats["support_doc_recall"] = (stats["support_docs_selected"] /
                                    max(stats["support_docs_total"], 1))
    (root / "manifest.json").write_text(json.dumps({
        "origin": args.local_json or "hotpotqa/hotpot_qa:distractor:train",
        "official_validation_used": False,
        "official_test_used": False,
        "selection": "question-word-overlap only, no gold/supervision during retrieval",
        "dev_split": "deterministic SHA256 by ID over original train only",
        "agents": args.agents,
        "sentences_per_agent": args.sentences_per_agent,
        "stats": stats,
    }, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-json", default="",
                   help="Optional official HotpotQA TRAIN JSON download; otherwise HF train")
    p.add_argument("--out", required=True)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--sentences-per-agent", type=int, default=3)
    p.add_argument("--holdout-percent", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.agents < 2 or not 0 < args.holdout_percent < 50:
        p.error("agents must be >=2 and holdout-percent in 1..49")
    convert(args)


if __name__ == "__main__":
    main()
