from fal_client import AsyncClient

from poseblend.models.image_edit.base import BaseImageEditModel


class FalMultiImageEditModel(BaseImageEditModel):
    def __init__(self, endpoint: str, **client_kwargs):
        self.endpoint = endpoint
        self.client = AsyncClient(**client_kwargs)

    async def edit(
        self,
        *,
        prompt: str,
        image_urls: list[str],
        **kwargs,
    ) -> str:
        result = await self.client.subscribe(
            self.endpoint,
            arguments={"prompt": prompt, "image_urls": image_urls, **kwargs},
            with_logs=True,
        )
        return result["images"][0]["url"]


class FalSingleImageEditModel(BaseImageEditModel):
    def __init__(self, endpoint: str, **client_kwargs):
        self.endpoint = endpoint
        self.client = AsyncClient(**client_kwargs)

    async def edit(
        self,
        *,
        prompt: str,
        image_urls: list[str],
        **kwargs,
    ) -> str:
        result = await self.client.subscribe(
            self.endpoint,
            arguments={"prompt": prompt, "image_url": image_urls[0], **kwargs},
            with_logs=True,
        )
        return result["images"][0]["url"]
