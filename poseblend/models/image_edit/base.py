from abc import ABC, abstractmethod


class BaseImageEditModel(ABC):
    @abstractmethod
    async def edit(
        self,
        *,
        prompt: str,
        image_urls: list[str],
        **kwargs,
    ) -> str:
        pass
