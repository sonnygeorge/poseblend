from dataclasses import dataclass


# See equivalent pydantic class in poseblend/schema/inputs/blender_objects.py
@dataclass
class BlenderObjectMetadata:
    name: str
    file: str
    scale_factor: float
    default_facing_orientation: list[float] | None
