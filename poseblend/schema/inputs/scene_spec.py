from pydantic import BaseModel, Field


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
        aliases: Optional mapping from object names to display strings used in
            prompts/requirements, e.g. {"athlete": "human athlete"}. When absent,
            the raw object name from role_assignments is used.
    """

    scene_as_natural_language: str
    action: str
    role_assignments: dict[str, str]
    localized_edits: list[LocalizedEditSpec]
    final_requirements: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    scale_amounts: dict[str, float] = Field(default_factory=dict)

    def get_object_alias(self, obj_name: str) -> str:
        return self.aliases.get(obj_name, obj_name)

    def _hydrate_requirement(self, template: str) -> str:
        strs_by_role = {
            k: f"the {self.get_object_alias(v)}"
            for k, v in self.role_assignments.items()
        }
        hydrated = template.format(**strs_by_role)
        return hydrated[0].upper() + hydrated[1:]

    def get_hydrated_edit_requirements(self) -> list[list[str]]:
        return [
            [self._hydrate_requirement(req) for req in edit.requirements]
            for edit in self.localized_edits
        ]

    def get_hydrated_final_requirements(self) -> list[str]:
        return [self._hydrate_requirement(req) for req in self.final_requirements]
