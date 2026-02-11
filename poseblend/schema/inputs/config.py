from __future__ import annotations

from pydantic import BaseModel


class AttemptRangeModelSelectionProbs(BaseModel):
    """Selection probability distribution for models by attempt range."""

    from_attempt: int = 0
    to_attempt: int | None = None
    model_probs: dict[str, float] = {"fal-ai/gemini-25-flash-image/edit": 1.0}


class PoseBlendConfig(BaseModel):
    """Configuration for PoseBlend image generation."""

    base_scene_path: str = "inputs/objaverse/base_scene.blend"
    objects_dir_path: str = "inputs/objaverse/shapes"
    blender_lm: str = "gpt-5.2-2025-12-11"
    critic_vlm: str = "gpt-5.2-2025-12-11"
    edit_model_selection_schedule: list[AttemptRangeModelSelectionProbs] = (
        AttemptRangeModelSelectionProbs()
    )
    num_blender_scenes: int = 3
    num_renders: int = 2
    min_render_quality_score: float = 0.6
    max_edit_attempts: int = 3
    background_strs: list[str] = [
        "sandy background",
        "grassy background",
        "snowy background",
    ]
