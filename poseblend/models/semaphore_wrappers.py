import asyncio

from poseblend.models.image_edit.base import BaseImageEditModel
from poseblend.models.vlm.base import BaseModelType, BaseVLM, Message


class SemaphoreVLM(BaseVLM):
    def __init__(self, vlm: BaseVLM, semaphore: asyncio.Semaphore):
        self._vlm = vlm
        self._sem = semaphore

    async def infer_text(
        self,
        *,
        messages: list[Message],
        **kwargs,
    ) -> str:
        async with self._sem:
            return await self._vlm.infer_text(messages=messages, **kwargs)

    async def infer_structured(
        self,
        *,
        messages: list[Message],
        response_model: type[BaseModelType],
        **kwargs,
    ) -> BaseModelType:
        async with self._sem:
            return await self._vlm.infer_structured(
                messages=messages, response_model=response_model, **kwargs
            )


class SemaphoreImageEditModel(BaseImageEditModel):
    def __init__(self, model: BaseImageEditModel, semaphore: asyncio.Semaphore):
        self._model = model
        self._sem = semaphore

    async def edit(
        self,
        *,
        prompt: str,
        image_urls: list[str],
        **kwargs,
    ) -> str:
        async with self._sem:
            return await self._model.edit(
                prompt=prompt, image_urls=image_urls, **kwargs
            )
