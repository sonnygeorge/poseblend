from abc import ABC, abstractmethod


class BaseT2IModel(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        **kwargs,
    ) -> str:
        pass
