from pydantic import BaseModel


# See equivalent dataclass in blender/schema.py
class BlenderObjectMetadata(BaseModel):
    """Metadata for a Blender object in registry.

    Attributes:
        name: Name of the object.
        file: File name of the object.
        scale_factor: Factor by which the object's size should be scaled for realism after
            being loaded into Blender.
        default_facing_orientation: Euler angles (x, y, z in radians) of the direction the
            object "faces" by default when loaded into Blender. None is the object is
            isotrophic (e.g., a basketball) or otherwise does not have an angle that would
            be considered its "facing" direction. For example, a "chair's" facing direction
            would be the direction pointing horizontally outward from the seat/backrest.
    """

    name: str
    file: str
    scale_factor: float
    default_facing_orientation: list[float] | None


class BlenderObjectRegistry(BaseModel):
    objects: dict[str, BlenderObjectMetadata]
