from __future__ import annotations

from pydantic import BaseModel


class LocalizedEditSpec(BaseModel):
    """...  # TODO

    Attributes:
        region_contains: List of objects that together, define the region of the image to
            be edited.
        requirements: F-string templates for language expressions that must be true in
            in order for the edit to be acceptable.
    """

    region_contains: list[str]
    requirements: list[str]


class PoseBlendSceneSpec(BaseModel):
    """Input actions scene specification for PoseBlend image generation."""

    scene_as_natural_language: str
    action: str
    role_assignments: dict[str, str]
    localized_edits: list[LocalizedEditSpec]
