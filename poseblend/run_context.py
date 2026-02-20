import asyncio
from collections.abc import Callable

from poseblend.models.image_edit.base import BaseImageEditModel
from poseblend.models.registry import get_image_edit_model, get_t2i_model, get_vlm
from poseblend.models.semaphore_wrappers import (
    SemaphoreImageEditModel,
    SemaphoreT2IModel,
    SemaphoreVLM,
)
from poseblend.models.t2i.base import BaseT2IModel
from poseblend.models.vlm.base import BaseVLM
from poseblend.schema.run_data import RunData


class RunContext:
    def __init__(self, run_data: RunData, on_update: Callable[[], None] | None = None):
        self.run_data = run_data
        self._on_update = on_update
        self._inference_semaphore = asyncio.Semaphore(
            run_data.config.max_concurrent_inference_requests
        )
        self._blender_semaphore = asyncio.Semaphore(
            run_data.config.max_concurrent_blender_processes
        )

    def on_run_data_changed(self):
        self.run_data.save()
        if self._on_update is not None:
            self._on_update()

    @property
    def blender_semaphore(self) -> asyncio.Semaphore:
        return self._blender_semaphore

    def get_vlm(self, model: str) -> BaseVLM:
        return SemaphoreVLM(get_vlm(model), self._inference_semaphore)

    def get_image_edit_model(self, model: str) -> BaseImageEditModel:
        return SemaphoreImageEditModel(
            get_image_edit_model(model), self._inference_semaphore
        )

    def get_t2i_model(self, model: str) -> BaseT2IModel:
        return SemaphoreT2IModel(
            get_t2i_model(model), self._inference_semaphore
        )
