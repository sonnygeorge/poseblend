import asyncio

from poseblend.models.image_edit.base import BaseImageEditModel
from poseblend.models.registry import get_image_edit_model, get_vlm
from poseblend.models.semaphore_wrappers import SemaphoreImageEditModel, SemaphoreVLM
from poseblend.models.vlm.base import BaseVLM
from poseblend.schema.run_data import RunData


class RunContext:
    def __init__(self, run_data: RunData):
        self.run_data = run_data
        self._inference_semaphore = asyncio.Semaphore(
            run_data.config.max_concurrent_inference_requests
        )
        self._blender_semaphore = asyncio.Semaphore(
            run_data.config.max_concurrent_blender_processes
        )

    @property
    def blender_semaphore(self) -> asyncio.Semaphore:
        return self._blender_semaphore

    def get_vlm(self, model: str) -> BaseVLM:
        return SemaphoreVLM(get_vlm(model), self._inference_semaphore)

    def get_image_edit_model(self, model: str) -> BaseImageEditModel:
        return SemaphoreImageEditModel(
            get_image_edit_model(model), self._inference_semaphore
        )
