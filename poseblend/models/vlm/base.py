from abc import ABC, abstractmethod
from typing import Literal, TypedDict, TypeVar

from pydantic import BaseModel

BaseModelType = TypeVar("BaseModelType", bound=BaseModel)


class MessageContent(TypedDict):
    type: Literal["text", "image_url"]
    content: str


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: list[MessageContent]


class BaseVLM(ABC):
    @abstractmethod
    async def infer_text(
        self,
        *,
        messages: list[Message],
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def infer_structured(
        self,
        *,
        messages: list[Message],
        response_model: type[BaseModelType],
        **kwargs,
    ) -> BaseModelType:
        pass
