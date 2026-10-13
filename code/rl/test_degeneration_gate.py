#!/usr/bin/env python3

from reward import calculate_reward_details, detect_response_degeneration


def reward_for(text: str, *, ended_eod: bool = True, truncated: bool = False):
    return calculate_reward_details(
        text=text,
        ground_truth="4",
        token_length=256,
        ended_eod=ended_eod,
        truncated=truncated,
        zstd_penalty_weight=0.0,
    )


def main() -> None:
    clean = "We calculate 2+2=4.</think><answer>\\boxed{4}</answer>"
    multilingual = "We calculate 2+2=4. 中文中文</think><answer>\\boxed{4}</answer>"
    repeated = ("therefore the answer follows " * 12) + "</think><answer>\\boxed{4}</answer>"

    clean_reward, clean_correct, clean_format, clean_details = reward_for(clean)
    assert clean_reward == 1.0 and clean_correct and clean_format
    assert not clean_details["degenerate"]

    language_reward, language_correct, language_format, language_details = reward_for(multilingual)
    assert language_reward == 0.0 and language_correct and not language_format
    assert language_details["degenerate_multilingual"]
    assert language_details["structural_format_valid"]
    assert not language_details["quality_gate_pass"]

    repeat_reward, repeat_correct, repeat_format, repeat_details = reward_for(repeated)
    assert repeat_reward == 0.0 and repeat_correct and not repeat_format
    assert repeat_details["degenerate_repeat"]

    truncated_reward, truncated_correct, truncated_format, truncated_details = reward_for(
        clean,
        truncated=True,
    )
    assert truncated_reward == 0.0 and truncated_correct and not truncated_format
    assert truncated_details["structural_format_valid"]
    assert not truncated_details["quality_gate_pass"]

    assert not detect_response_degeneration(clean)["degenerate"]
    print("degeneration gate tests passed")


if __name__ == "__main__":
    main()
