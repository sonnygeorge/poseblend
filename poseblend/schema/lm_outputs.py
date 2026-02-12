from pydantic import BaseModel, Field


class YesNoAnswer(BaseModel):
    reasoning: str = Field(description="Reasoning for the answer")
    answer: bool = Field(description="Binary yes/no answer")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence between 0 (no confidence) and 1 (full confidence)",
    )


# See equivalent dataclass in blender/schema.py
class ObjectPlacementParams(BaseModel):
    """Parameters for placing an object in a Blender scene."""

    name: str = Field(description="Name of the object to place")
    target_location: list[float] = Field(
        description="Target (x, y, z) placement location of the object"
    )
    target_facing_direction: list[float] | None = Field(
        default=None,
        description="Target Euler angles (x, y, z in radians) defining the direction the object should face",
    )
    touching_objects: list[str] = Field(
        description="List of names of objects that the object should be touching"
    )
    touching_ground: bool = Field(
        description="Whether the object should be touching the ground"
    )


class BlenderSceneParams(BaseModel):
    placements: list[ObjectPlacementParams] = Field(
        description="List of placement parameters for each object in the scene"
    )
