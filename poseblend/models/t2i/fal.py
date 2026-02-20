from fal_client import AsyncClient

from poseblend.models.t2i.base import BaseT2IModel


class FalT2IModel(BaseT2IModel):
    def __init__(self, endpoint: str, **client_kwargs):
        self.endpoint = endpoint
        self.client = AsyncClient(**client_kwargs)

    async def generate(
        self,
        *,
        prompt: str,
        **kwargs,
    ) -> str:
        result = await self.client.subscribe(
            self.endpoint,
            arguments={"prompt": prompt, **kwargs},
            with_logs=True,
        )
        return result["images"][0]["url"]
