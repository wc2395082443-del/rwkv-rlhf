import json
import argparse
import pandas as pd
from . import common
from .gpqa_eval_qwen import GPQAEvalQwen
from .aime24_eval_qwen import AIME24EvalQwen
from .aime25_eval_qwen import AIME25EvalQwen
from .gsm8k_eval_qwen import Gsm8kEvalQwen
from .minerva_eval_qwen import MinervaEvalQwen
from .amc_eval_qwen import AmcEvalQwen
from .math_eval_qwen import MathEvalQwen
from .olympiad_eval_qwen import OlympiadEvalQwen
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.qwen_chat_completion_sampler import QwenChatCompletionSampler


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()

    models = {
        "Qwen2.5-7B": QwenChatCompletionSampler(
            model="Qwen/Qwen2.5-7B",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
        "Qwen2.5-14B": QwenChatCompletionSampler(
            model="Qwen/Qwen2.5-14B",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
        "Qwen2.5-7B-Instruct": QwenChatCompletionSampler(
            model="Qwen/Qwen2.5-7B-Instruct",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
        "Qwen2.5-14B-Instruct": QwenChatCompletionSampler(
            model="Qwen/Qwen2.5-14B-Instruct",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
        "Qwen-2.5-7B-SimpleRL-Zoo": QwenChatCompletionSampler(
            model="hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
        "Qwen-2.5-14B-SimpleRL-Zoo": QwenChatCompletionSampler(
            model="hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
        "General-Reasoner-7B-preview": QwenChatCompletionSampler(
            model="TIGER-Lab/General-Reasoner-7B-preview",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
        "General-Reasoner-7B-preview-step560": QwenChatCompletionSampler(
            model="MrLight/scale-reasoning-data-v2-nos-fixmc-fil0-fil8-Qwen2.5-7B",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
        "General-Reasoner-14B-preview": QwenChatCompletionSampler(
            model="TIGER-Lab/General-Reasoner-14B-preview",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1,
            max_tokens=8192,
        ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return
        models = {args.model: models[args.model]}

    equality_checker = ChatCompletionSampler(model="gpt-4o")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "math":
                return MathEvalQwen(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else 1,
                )
            case "aime24":
                return AIME24EvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1 if debug_mode else 32, num_examples=num_examples
                )
            case "aime25":
                return AIME25EvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1 if debug_mode else 32, num_examples=num_examples
                )
            case 'olympiad':
                return OlympiadEvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1 if debug_mode else 1, num_examples=num_examples
                )
            case 'gsm8k':
                return Gsm8kEvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1 if debug_mode else 1, num_examples=num_examples
                )
            case "minerva":
                return MinervaEvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1 if debug_mode else 1, num_examples=num_examples
                )
            case "amc":
                return AmcEvalQwen(
                    equality_checker=equality_checker,
                    n_repeats=1 if debug_mode else 1, num_examples=num_examples
                )
            case "gpqa":
                return GPQAEvalQwen(
                    n_repeats=1 if debug_mode else 1, num_examples=num_examples
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.debug)
        for eval_name in ["aime24", "aime25"]#['amc', "math", "aime24", "aime25", "gsm8k", "minerva", "olympiad", "gpqa"]
    }
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            report_filename = f"./{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"./{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
