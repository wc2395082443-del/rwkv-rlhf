import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


BOOL_STYLE_ANSWERS = {"True", "False", "Yes", "No"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild DeepMath-103K in the exact Hugging Face TRL dataset format."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Directory containing the raw DeepMath-103K parquet shards under data/*.parquet.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the converted Hugging Face dataset will be saved.",
    )
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def process_example(example):
    solution = str(example["final_answer"]).strip()
    if solution not in BOOL_STYLE_ANSWERS:
        solution = f"${solution}$"
    prompt = [{"role": "user", "content": example["question"].strip()}]
    return {"prompt": prompt, "solution": solution}


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

    dataset = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        desc="Converting DeepMath rows into the official TRL prompt/solution format",
    )
    split = DatasetDict(dataset.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True))
    split.save_to_disk(str(output_dir))

    print(f"Saved converted dataset to: {output_dir}")
    print(f"Train size: {len(split['train'])}")
    print(f"Test size:  {len(split['test'])}")
    print("Example prompt:")
    print(split["train"][0]["prompt"])
    print("Example solution:")
    print(split["train"][0]["solution"])


if __name__ == "__main__":
    main()

