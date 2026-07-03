import argparse
from pathlib import Path

from datasets import load_dataset


DEFAULT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw DeepMath-103K parquet shards into the TRL GRPO prompt/solution format."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Directory that contains the raw DeepMath-103K `data/*.parquet` shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the converted Hugging Face dataset.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        help="Fraction of samples reserved for the test split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--official_mode",
        type=str,
        default="deepmath_zero",
        choices=["deepmath_zero", "deepmath_r1"],
        help=(
            "Which official DeepMath-style prompt format to emit. "
            "`deepmath_zero` adds the official simplerl system prompt; "
            "`deepmath_r1` keeps a plain user question for R1-style models."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for fast smoke tests.",
    )
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

    def convert_row(row):
        question = row["question"].strip()
        ground_truth = str(row["final_answer"]).strip()
        if args.official_mode == "deepmath_zero":
            prompt = [
                {"role": "system", "content": DEFAULT_INSTRUCTION},
                {"role": "user", "content": question},
            ]
        else:
            prompt = [{"role": "user", "content": question}]
        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "solution": f"\\boxed{{${ground_truth}$}}" if ground_truth else "",
            "question": question,
            "difficulty": str(row["difficulty"]),
            "topic": str(row["topic"]),
            "data_source": "deepmath",
            "extra_info": {
                "difficulty": str(row["difficulty"]),
                "topic": str(row["topic"]),
            },
        }

    converted = dataset.map(
        convert_row,
        remove_columns=dataset.column_names,
        desc="Converting DeepMath rows into TRL prompt/solution format",
    )

    split = converted.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    split.save_to_disk(str(output_dir))

    print(f"Saved converted dataset to: {output_dir}")
    print(f"Train size: {len(split['train'])}")
    print(f"Test size:  {len(split['test'])}")
    print("Example prompt:")
    print(split["train"][0]["prompt"])
    print("Example ground_truth:")
    print(split["train"][0]["ground_truth"])
    print("Example wrapped solution:")
    print(split["train"][0]["solution"])


if __name__ == "__main__":
    main()
