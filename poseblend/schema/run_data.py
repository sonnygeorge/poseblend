from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from poseblend.schema.inputs.blender_objects import BlenderObjectRegistry
from poseblend.schema.inputs.config import PoseBlendConfig
from poseblend.schema.inputs.scene_spec import PoseBlendSceneSpec
from poseblend.schema.lm_outputs import BlenderSceneParams, CriticResult
from poseblend.utils import load_yaml, save_json


class GateDecision(BaseModel):
    is_passing: bool
    reason: str


class CriticInvocation(BaseModel):
    requirement: str
    result: CriticResult


class AttemptedEdit(BaseModel):
    seed: int | None
    before_img_path: Path
    after_img_path: Path
    prompt_used: str | None
    model_used: str | None
    critic_invocations: list[CriticInvocation]
    gate_decision: GateDecision


class EditChain(BaseModel):
    starting_render_path: Path
    edits: list[list[AttemptedEdit]]
    candidate_final_img_path: Path | None
    final_critic_invocations: list[CriticInvocation] = Field(default_factory=list)
    gate_decision: GateDecision | None = None


class SceneRender(BaseModel):
    render_id: int
    image_path: Path | None = None
    mask_dir_path: Path | None = None
    critic_invocations: list[CriticInvocation] = Field(default_factory=list)
    render_quality_score: float | None = None
    gate_decision: GateDecision | None = None


class BlenderScene(BaseModel):
    scene_id: int
    seed: int | None = None
    params: BlenderSceneParams
    blend_file_path: Path | None = None
    renders: list[SceneRender] = Field(default_factory=list)
    scene_quality_score: float | None = None
    is_selected: bool = False
    gate_decision: GateDecision | None = None


class RunData(BaseModel):
    config: PoseBlendConfig
    scene_spec: PoseBlendSceneSpec
    blender_object_registry: BlenderObjectRegistry
    run_id: str
    scenes: list[BlenderScene] = Field(default_factory=list)
    edit_chains: list[EditChain] = Field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        return Path(self.config.output_dir_path) / self.run_id

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "run_data.json"
        save_json(self.model_dump(mode="json"), path)
        return path

    @model_validator(mode="after")
    def _validate_role_assignments_in_registry(self) -> "RunData":
        invalid = set(self.scene_spec.role_assignments.values()) - set(
            self.blender_object_registry.objects.keys()
        )
        if invalid:
            msg = f"Scene role_assignments reference unknown blender objects: {invalid}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_input_yaml_paths(
        cls,
        config_path: Path | str,
        scene_path: Path | str,
        blender_object_data_path: Path | str,
    ) -> "RunData":
        config = PoseBlendConfig.model_validate(load_yaml(config_path))
        scene_spec = PoseBlendSceneSpec.model_validate(load_yaml(scene_path))
        blender_object_registry = BlenderObjectRegistry.model_validate(load_yaml(blender_object_data_path))
        run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

        return cls(
            config=config,
            scene_spec=scene_spec,
            blender_object_registry=blender_object_registry,
            run_id=run_id,
        )
