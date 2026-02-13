from openai import AsyncOpenAI

from poseblend.models.vlm.base import BaseModelType, BaseVLM, Message, MessageContent


def _convert_message_content(content: MessageContent) -> dict:
    if content["type"] == "text":
        return {"type": "text", "text": content["content"]}
    else:
        return {"type": "image_url", "image_url": {"url": content["content"]}}


def _convert_messages(messages: list[Message]) -> list[dict]:
    return [
        {
            "role": msg["role"],
            "content": [_convert_message_content(c) for c in msg["content"]],
        }
        for msg in messages
    ]


class OpenAIVLM(BaseVLM):
    def __init__(self, model: str = "gpt-5.2-2025-12-11", **client_kwargs):
        self.model = model
        self.client = AsyncOpenAI(**client_kwargs)

    async def infer_text(
        self,
        *,
        messages: list[Message],
        **kwargs,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=_convert_messages(messages),
            **kwargs,
        )
        return response.choices[0].message.content

    async def infer_structured(
        self,
        *,
        messages: list[Message],
        response_model: type[BaseModelType],
        **kwargs,
    ) -> BaseModelType:
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=_convert_messages(messages),
            response_format=response_model,
            **kwargs,
        )
        return response.choices[0].message.parsed
