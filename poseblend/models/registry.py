from poseblend.models.vlm.openai import OpenAIVLM
from poseblend.models.image_edit.fal import FalMultiImageEditModel, FalSingleImageEditModel

VLMS = {"gpt-5.2-2025-12-11": OpenAIVLM("gpt-5.2-2025-12-11")}

IMAGE_EDIT_MODELS = {
    # Fal-ai multi-image edit models
    "fal-ai/gemini-25-flash-image/edit": FalMultiImageEditModel("fal-ai/gemini-25-flash-image/edit"),
    "fal-ai/gpt-image-1.5/edit": FalMultiImageEditModel("fal-ai/gpt-image-1.5/edit"),
    "fal-ai/flux-2/edit": FalMultiImageEditModel("fal-ai/flux-2/edit"),
    "fal-ai/flux-2/turbo/edit": FalMultiImageEditModel("fal-ai/flux-2/turbo/edit"),
    "fal-ai/hunyuan-image/v3/instruct/edit": FalMultiImageEditModel("fal-ai/hunyuan-image/v3/instruct/edit"),
    "fal-ai/nano-banana/edit": FalMultiImageEditModel("fal-ai/nano-banana/edit"),

    # Fal-ai single-image edit models
    "xai/grok-imagine-image/edit": FalSingleImageEditModel("xai/grok-imagine-image/edit"),
}

