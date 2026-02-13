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
    """Input actions scene specification for PoseBlend image generation.

    Attributes:
        scene_as_natural_language: Plain English description of the scene, e.g.
            "A person throws a ball over a fence to a dog."
        action: The core action being depicted, e.g. "throws".
        role_assignments: Mapping from action roles to object names in the
            BlenderObjectRegistry, e.g. {"thrower": "person", "thrown": "ball", ...}.
        localized_edits: Ordered list of region-level edit specs whose requirement
            templates reference keys from role_assignments.
    """

    scene_as_natural_language: str
    action: str
    role_assignments: dict[str, str]
    localized_edits: list[LocalizedEditSpec]

    def get_hydrated_edit_requirements(self) -> list[list[str]]:
        strs_by_role = {k: f"the {v}" for k, v in self.role_assignments.items()}
        result = []
        for edit in self.localized_edits:
            hydrated_reqs = []
            for req in edit.requirements:
                hydrated = req.format(**strs_by_role)
                hydrated = hydrated[0].upper() + hydrated[1:]
                hydrated_reqs.append(hydrated)
            result.append(hydrated_reqs)
        return result
