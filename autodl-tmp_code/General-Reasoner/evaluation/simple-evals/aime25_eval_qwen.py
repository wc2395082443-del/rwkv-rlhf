"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import random
import re
from typing import Literal

import pandas

from . import common
from .common import ANSWER_PATTERN_BOXED, HTML_JINJA, check_equality
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

from datasets import load_dataset

QUERY_TEMPLATE = """
<question>

Please reason step by step, and put your final answer within \\boxed{}.
""".strip()


class AIME25EvalQwen(Eval):
    def __init__(
        self,
        equality_checker: SamplerBase,
        num_examples: int | None = None,
        n_repeats: int = 16,
        split: Literal["train"] = "train",
    ):
        dataset = load_dataset("yentinglin/aime_2025", 'default')['train']
        examples = [row for row in dataset]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            content = QUERY_TEMPLATE.replace("<question>", row["problem"])
            prompt_messages = [
                sampler._pack_message(content=content, role="user")
            ]
            response_text = sampler(prompt_messages)
            match = re.search(ANSWER_PATTERN_BOXED, response_text)
            extracted_answer = match.group(1) if match else None
            score = float(check_equality(self.equality_checker, row["answer"], extracted_answer))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
