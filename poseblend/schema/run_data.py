from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, model_validator

from poseblend.schema.inputs.scene_spec import PoseBlendSceneSpec
from poseblend.schema.inputs.config import PoseBlendConfig
from poseblend.schema.inputs.blender_objects import BlenderObjectRegistry
from poseblend.schema.lm_outputs import BlenderSceneParams
from poseblend.schema.edit_chain import EditChain
from poseblend.utils import load_yaml, save_json


class SceneRender(BaseModel):
    render_id: int
    image_path: Path | None = None
    mask_dir_path: Path | None = None


class BlenderScene(BaseModel):
    scene_id: int
    seed: int | None = None
    params: BlenderSceneParams
    blend_file_path: Path | None = None
    renders: list[SceneRender] = []


class RunData(BaseModel):
    config: PoseBlendConfig
    scene_spec: PoseBlendSceneSpec
    blender_object_registry: BlenderObjectRegistry
    run_id: str
    scenes: list[BlenderScene] = []
    edit_chains: list[EditChain] = []

    @property
    def run_dir(self) -> Path:
        return Path(self.config.output_dir_path) / self.run_id

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "run_data.json"
        save_json(self.model_dump(mode="json"), path)
        return path

    @model_validator(mode="after")
    def _validate_role_assignments_in_registry(self) -> RunData:
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
    ) -> RunData:
        config = PoseBlendConfig.model_validate(load_yaml(config_path))
        scene_spec = PoseBlendSceneSpec.model_validate(load_yaml(scene_path))
        blender_object_registry = BlenderObjectRegistry.model_validate(load_yaml(blender_object_data_path))
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        return cls(
            config=config,
            scene_spec=scene_spec,
            blender_object_registry=blender_object_registry,
            run_id=run_id,
        )
