from dataclasses import dataclass


# See equivalent pydantic class in poseblend/schema/inputs/blender_objects.py
@dataclass
class BlenderObjectMetadata:
    name: str
    file: str
    scale_factor: float
    default_facing_orientation: list[float] | None


# See equivalent pydantic class in poseblend/schema/llm_outputs.py
@dataclass
class ObjectPlacementParams:
    name: str
    target_location: list[float]
    target_facing_direction: list[float] | None
    touching_ground: bool


@dataclass
class BlenderObjectSpec:
    name: str
    file_path: str
    scale_factor: float
    default_facing_orientation: list[float] | None


@dataclass
class RenderJob:
    base_scene_path: str
    objects: list[BlenderObjectSpec]
    placements: list[ObjectPlacementParams]
    output_dir: str
    num_renders: int
    resolution_x: int
    resolution_y: int
    camera_fov_degrees: float
    seed: int | None = None
    save_blend_file: bool = False
