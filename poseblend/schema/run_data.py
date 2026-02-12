from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from poseblend.schema.inputs.scene_spec import PoseBlendSceneSpec
from poseblend.schema.inputs.config import PoseBlendConfig
from poseblend.schema.inputs.blender_objects import BlenderObjectRegistry
from poseblend.schema.edit_chain import EditChain
from poseblend.utils import load_yaml


class RunData(BaseModel):
    config: PoseBlendConfig
    scene_spec: PoseBlendSceneSpec
    blender_object_registry: BlenderObjectRegistry
    edit_chains: list[EditChain] = []

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

        return cls(
            config=config,
            scene_spec=scene_spec,
            blender_object_registry=blender_object_registry,
        )