from __future__ import annotations

from pydantic import BaseModel


class AttemptRangeModelSelectionProbs(BaseModel):
    from_attempt: int
    to_attempt: int | None
    model_probs: dict[str, float]


class PoseBlendConfig(BaseModel):
    num_blender_scenes: int
    num_renders: int
    num_edit_chains: int
    min_render_quality_score: float
    max_edit_attempts: int
    base_scene_path: str
    objects_dir_path: str
    blender_lm: str
    critic_vlm: str
    edit_model_selection_schedule: list[AttemptRangeModelSelectionProbs]
    background_strs: list[str]
