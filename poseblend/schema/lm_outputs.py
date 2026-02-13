from enum import IntEnum

from pydantic import BaseModel, Field


class LikertScore(IntEnum):
    CLEARLY_VIOLATED = 1
    MOSTLY_VIOLATED = 2
    AMBIGUOUS = 3
    MOSTLY_SATISFIED = 4
    CLEARLY_SATISFIED = 5


class CriticResult(BaseModel):
    reasoning: str = Field(
        description="Step-by-step visual analysis of the requirement"
    )
    score: LikertScore = Field(
        description=(
            "How well the requirement is satisfied in the image: "
            "1=clearly violated, 2=mostly violated, "
            "3=ambiguous/partial, 4=mostly satisfied, "
            "5=clearly satisfied"
        ),
    )

    @property
    def normalized_score(self) -> float:
        return (self.score - 1) / 4.0


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
