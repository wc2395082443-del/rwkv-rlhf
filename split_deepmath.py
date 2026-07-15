#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk


def normalize_text(value):
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split()).casefold()


def normalize_answer(value):
    text = normalize_text(value)
    return text.replace(r"\dfrac", r"\frac")


def difficulty_bucket(value):
    difficulty = float(value)
    if difficulty < 0:
        return "unknown"
    if difficulty < 5:
        return "easy"
    if difficulty < 7:
        return "medium"
    return "hard"


def topic_domain(topic):
    parts = [part.strip() for part in str(topic).split("->")]
    return parts[1] if len(parts) > 1 else parts[0]


def stable_digest(seed, normalized_question):
    payload = f"{seed}\0{normalized_question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def allocate_validation(pool, validation_count, seed):
    by_bucket = defaultdict(list)
    for row in pool:
        by_bucket[row["difficulty_bucket"]].append(row)

    allocation = {}
    fractions = []
    for bucket, rows in sorted(by_bucket.items()):
        exact = validation_count * len(rows) / len(pool)
        allocation[bucket] = math.floor(exact)
        fractions.append((exact - allocation[bucket], bucket))

    remaining = validation_count - sum(allocation.values())
    for _, bucket in sorted(fractions, reverse=True)[:remaining]:
        allocation[bucket] += 1

    validation_ids = set()
    for bucket, rows in sorted(by_bucket.items()):
        ranked = sorted(rows, key=lambda row: stable_digest(seed, row["normalized_question"]))
        validation_ids.update(row["id"] for row in ranked[: allocation[bucket]])

    train = [row for row in pool if row["id"] not in validation_ids]
    validation = [row for row in pool if row["id"] in validation_ids]
    return train, validation


def summarize(rows):
    return {
        "count": len(rows),
        "difficulty_buckets": dict(sorted(Counter(row["difficulty_bucket"] for row in rows).items())),
        "difficulty_values": dict(
            sorted(Counter(str(row["difficulty"]) for row in rows).items(), key=lambda item: float(item[0]))
        ),
        "topic_domains": dict(sorted(Counter(row["topic_domain"] for row in rows).items())),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Create a leakage-free DeepMath split for RWKV RL training")
    parser.add_argument(
        "--dataset",
        default="/root/autodl-tmp/official_repro_assets/DeepMath-103K-trl-deepmath_r1",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_from_disk(args.dataset)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    groups = {}
    raw_split_sizes = {}
    for split in ("train", "test"):
        rows = dataset[split]
        raw_split_sizes[split] = len(rows)
        for source_index, source in enumerate(rows):
            question = str(source["question"]).strip()
            answer = str(source["ground_truth"]).strip()
            normalized_question = normalize_text(question)
            group = groups.setdefault(
                normalized_question,
                {
                    "row_count": 0,
                    "answers": set(),
                    "source_splits": set(),
                    "representative": None,
                },
            )
            group["row_count"] += 1
            group["source_splits"].add(split)
            if answer:
                group["answers"].add(normalize_answer(answer))

            candidate = {
                "id": f"deepmath-{hashlib.sha256(normalized_question.encode('utf-8')).hexdigest()[:20]}",
                "problem": question,
                "answer": answer,
                "solution": answer,
                "difficulty": float(source["difficulty"]),
                "difficulty_bucket": difficulty_bucket(source["difficulty"]),
                "topic": str(source["topic"]),
                "topic_domain": topic_domain(source["topic"]),
                "source_split": split,
                "source_index": source_index,
                "normalized_question": normalized_question,
            }
            representative = group["representative"]
            test_preferred = split == "test" and representative and representative["source_split"] != "test"
            answer_preferred = answer and representative and not representative["answer"]
            if representative is None or test_preferred or answer_preferred:
                group["representative"] = candidate

    raw_rows = sum(group["row_count"] for group in groups.values())
    duplicate_groups = sum(group["row_count"] > 1 for group in groups.values())
    duplicate_extra_rows = raw_rows - len(groups)
    source_overlap_groups = sum(len(group["source_splits"]) > 1 for group in groups.values())

    clean_rows = []
    dropped_empty = 0
    dropped_conflict = 0
    for group in groups.values():
        if not group["answers"]:
            dropped_empty += 1
            continue
        if len(group["answers"]) > 1:
            dropped_conflict += 1
            continue
        row = group["representative"]
        if not row["answer"]:
            dropped_empty += 1
            continue
        clean_rows.append(row)

    test = [row for row in clean_rows if row["source_split"] == "test"]
    train_pool = [row for row in clean_rows if row["source_split"] != "test"]
    validation_count = round(len(clean_rows) * args.validation_fraction)
    train, validation = allocate_validation(train_pool, validation_count, args.seed)

    split_rows = {"train": train, "validation": validation, "test": test}
    for split, rows in split_rows.items():
        rows.sort(key=lambda row: stable_digest(args.seed, row["normalized_question"]))
        for row in rows:
            row.pop("normalized_question", None)
        write_jsonl(output / f"{split}.jsonl", rows)

    bucket_dir = output / "by_difficulty"
    bucket_dir.mkdir(exist_ok=True)
    for split, rows in split_rows.items():
        for bucket in ("unknown", "easy", "medium", "hard"):
            bucket_rows = [row for row in rows if row["difficulty_bucket"] == bucket]
            write_jsonl(bucket_dir / f"{split}_{bucket}.jsonl", bucket_rows)

    normalized_sets = {
        split: {normalize_text(row["problem"]) for row in rows}
        for split, rows in split_rows.items()
    }
    overlaps = {
        "train_validation": len(normalized_sets["train"] & normalized_sets["validation"]),
        "train_test": len(normalized_sets["train"] & normalized_sets["test"]),
        "validation_test": len(normalized_sets["validation"] & normalized_sets["test"]),
    }
    manifest = {
        "input_dataset": args.dataset,
        "seed": args.seed,
        "validation_fraction_of_clean_total": args.validation_fraction,
        "policy": "Deduplicate normalized questions; preserve official test; test wins cross-split overlap; stratify validation by difficulty bucket.",
        "raw_split_sizes": raw_split_sizes,
        "raw_total": raw_rows,
        "raw_unique_questions": len(groups),
        "duplicate_question_groups": duplicate_groups,
        "duplicate_extra_rows": duplicate_extra_rows,
        "source_train_test_overlap_groups": source_overlap_groups,
        "dropped_empty_answer_groups": dropped_empty,
        "dropped_conflicting_answer_groups": dropped_conflict,
        "clean_total": len(clean_rows),
        "output_splits": {split: summarize(rows) for split, rows in split_rows.items()},
        "output_question_overlap": overlaps,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
