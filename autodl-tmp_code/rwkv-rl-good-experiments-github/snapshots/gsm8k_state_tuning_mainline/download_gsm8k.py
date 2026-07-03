import requests
import json

SPLITS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}

OUTPUT_FILES = {
    "train": "gsm8k_train_formatted.jsonl",
    "test": "gsm8k_test_formatted.jsonl",
}


def download_and_convert(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")

    url = SPLITS[split]
    out_file = OUTPUT_FILES[split]

    print(f"Downloading GSM8K {split} split...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"download failed for {split}: status={response.status_code}")

    total_lines = 0
    converted_data = []

    for line in response.iter_lines():
        if not line:
            continue
        item = json.loads(line)
        question = item["question"]
        raw_answer = item["answer"]
        final_number = raw_answer.split("####")[-1].strip()

        new_record = {
            "problem": question,
            "solution": f"The answer is \\boxed{{{final_number}}}",
            "original_answer": raw_answer,
        }
        converted_data.append(new_record)
        total_lines += 1

    with open(out_file, "w", encoding="utf-8") as f:
        for record in converted_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done: {split} ({total_lines} items) -> {out_file}")


def main() -> None:
    for split in ("train", "test"):
        download_and_convert(split)


if __name__ == "__main__":
    main()
