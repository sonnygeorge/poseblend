import asyncio
from pathlib import Path

from poseblend.models.vlm.base import BaseVLM, Message, MessageContent
from poseblend.schema.lm_outputs import BlenderSceneParams
from poseblend.schema.run_data import RunData

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "generate_blender_params_system.txt").read_text()
USER_PROMPT_TEMPLATE = (PROMPTS_DIR / "generate_blender_params_user.txt").read_text()


def _build_messages(run_data: RunData) -> list[Message]:
    role_assignments_str = "\n".join(
        f"  {role} -> {obj}" for role, obj in run_data.scene_spec.role_assignments.items()
    )

    relevant_object_names = set(run_data.scene_spec.role_assignments.values())
    object_lines = []
    for name in relevant_object_names:
        meta = run_data.blender_object_registry.objects[name]
        line = f"  - {meta.name} (scale_factor={meta.scale_factor}"
        if meta.default_facing_orientation is not None:
            line += f", default_facing_orientation={meta.default_facing_orientation}"
        line += ")"
        object_lines.append(line)
    object_details_str = "\n".join(object_lines)

    hydrated_reqs = run_data.scene_spec.get_hydrated_edit_requirements()
    req_lines = []
    for edit_reqs in hydrated_reqs:
        for req in edit_reqs:
            req_lines.append(f"  - {req}")
    requirements_str = "\n".join(req_lines)

    user_text = USER_PROMPT_TEMPLATE.format(
        scene_description=run_data.scene_spec.scene_as_natural_language,
        action=run_data.scene_spec.action,
        role_assignments=role_assignments_str,
        object_details=object_details_str,
        requirements=requirements_str,
    )

    return [
        Message(
            role="system",
            content=[MessageContent(type="text", content=SYSTEM_PROMPT)],
        ),
        Message(
            role="user",
            content=[MessageContent(type="text", content=user_text)],
        ),
    ]


async def generate_blender_params(
    run_data: RunData,
    vlm: BaseVLM,
) -> list[BlenderSceneParams]:
    messages = _build_messages(run_data)
    tasks = [
        vlm.infer_structured(messages=messages, response_model=BlenderSceneParams)
        for _ in range(run_data.config.num_blender_scenes)
    ]
    return await asyncio.gather(*tasks)
