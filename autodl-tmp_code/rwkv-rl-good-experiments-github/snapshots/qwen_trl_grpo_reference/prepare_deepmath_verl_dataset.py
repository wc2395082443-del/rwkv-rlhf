import argparse
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DeepMath-103K raw parquet shards into verl parquet format."
    )
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--official_mode",
        type=str,
        default="deepmath_r1",
        choices=["deepmath_r1", "deepmath_zero"],
        help="Prompt style matching the DeepMath official training branch.",
    )
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted((raw_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found under: {raw_dir / 'data'}")

    dataset = load_dataset("parquet", data_files=[str(path) for path in parquet_files], split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = str(example["question"]).strip()
            ground_truth = str(example["final_answer"]).strip()
            if args.official_mode == "deepmath_zero":
                prompt = [
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {"role": "user", "content": question},
                ]
            else:
                prompt = [{"role": "user", "content": question}]

            return {
                "data_source": "deepmath",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    "difficulty": str(example["difficulty"]),
                    "topic": str(example["topic"]),
                },
            }

        return process_fn

    split = dataset.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    train_dataset = split["train"].map(make_map_fn("train"), with_indices=True, remove_columns=split["train"].column_names)
    test_dataset = split["test"].map(make_map_fn("test"), with_indices=True, remove_columns=split["test"].column_names)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_dataset.to_parquet(str(train_path))
    test_dataset.to_parquet(str(test_path))

    print(f"Saved train parquet to: {train_path}")
    print(f"Saved test parquet to:  {test_path}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size:  {len(test_dataset)}")
    print("Example row:")
    print(train_dataset[0])


if __name__ == "__main__":
    main()
