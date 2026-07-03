import base64
import time
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."

class QwenChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-14B-Instruct",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        port: int = 8000,
    ):
        # self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
        )
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
