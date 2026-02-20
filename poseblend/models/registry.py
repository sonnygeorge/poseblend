from functools import partial

from poseblend.models.image_edit.base import BaseImageEditModel
from poseblend.models.image_edit.fal import FalMultiImageEditModel, FalSingleImageEditModel
from poseblend.models.t2i.base import BaseT2IModel
from poseblend.models.t2i.fal import FalT2IModel
from poseblend.models.vlm.base import BaseVLM
from poseblend.models.vlm.openai import OpenAIVLM

_VLM_FACTORIES: dict[str, partial[BaseVLM]] = {
    "gpt-5.2-2025-12-11": partial(OpenAIVLM, "gpt-5.2-2025-12-11"),
}

_IMAGE_EDIT_MODEL_FACTORIES: dict[str, partial[BaseImageEditModel]] = {
    "fal-ai/gemini-25-flash-image/edit": partial(FalMultiImageEditModel, "fal-ai/gemini-25-flash-image/edit"),
    "fal-ai/gpt-image-1.5/edit": partial(FalMultiImageEditModel, "fal-ai/gpt-image-1.5/edit"),
    "fal-ai/flux-2/edit": partial(FalMultiImageEditModel, "fal-ai/flux-2/edit"),
    "fal-ai/flux-2/turbo/edit": partial(FalMultiImageEditModel, "fal-ai/flux-2/turbo/edit"),
    "fal-ai/hunyuan-image/v3/instruct/edit": partial(FalMultiImageEditModel, "fal-ai/hunyuan-image/v3/instruct/edit"),
    "fal-ai/nano-banana/edit": partial(FalMultiImageEditModel, "fal-ai/nano-banana/edit"),
    "xai/grok-imagine-image/edit": partial(FalSingleImageEditModel, "xai/grok-imagine-image/edit"),
}

_T2I_MODEL_FACTORIES: dict[str, partial[BaseT2IModel]] = {
    "fal-ai/flux-2": partial(FalT2IModel, "fal-ai/flux-2"),
    "fal-ai/flux-2/turbo": partial(FalT2IModel, "fal-ai/flux-2/turbo"),
    "fal-ai/gemini-25-flash-image": partial(FalT2IModel, "fal-ai/gemini-25-flash-image"),
}

_vlm_cache: dict[str, BaseVLM] = {}
_image_edit_model_cache: dict[str, BaseImageEditModel] = {}
_t2i_model_cache: dict[str, BaseT2IModel] = {}


def get_vlm(model: str) -> BaseVLM:
    if model not in _vlm_cache:
        if model not in _VLM_FACTORIES:
            raise ValueError(f"Unknown VLM: {model}")
        _vlm_cache[model] = _VLM_FACTORIES[model]()
    return _vlm_cache[model]


def get_image_edit_model(model: str) -> BaseImageEditModel:
    if model not in _image_edit_model_cache:
        if model not in _IMAGE_EDIT_MODEL_FACTORIES:
            raise ValueError(f"Unknown image edit model: {model}")
        _image_edit_model_cache[model] = _IMAGE_EDIT_MODEL_FACTORIES[model]()
    return _image_edit_model_cache[model]


def get_t2i_model(model: str) -> BaseT2IModel:
    if model not in _t2i_model_cache:
        if model not in _T2I_MODEL_FACTORIES:
            raise ValueError(f"Unknown T2I model: {model}")
        _t2i_model_cache[model] = _T2I_MODEL_FACTORIES[model]()
    return _t2i_model_cache[model]

