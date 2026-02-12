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
    max_concurrent_inference_requests: int
    max_concurrent_blender_processes: int = 3  # recommended to be a divisor of num_blender_scenes
    base_scene_path: str
    objects_dir_path: str
    output_dir_path: str = "outputs"
    render_resolution_x: int
    render_resolution_y: int
    camera_fov_degrees: float
    blender_lm: str
    blender_lm_temperature: float
    critic_vlm: str
    critic_vlm_temperature: float
    seed: int | None = None
    edit_model_selection_schedule: list[AttemptRangeModelSelectionProbs]
    background_strs: list[str]
