import asyncio
import random
from pathlib import Path

from loguru import logger

from poseblend.models.vlm.base import Message, MessageContent
from poseblend.run_context import RunContext
from poseblend.schema.run_data import RunData
from poseblend.utils import derive_seed, derive_seeds, download_image, messages_to_prompt_string

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "generate_t2i_prompt_system.txt").read_text()
USER_PROMPT_TEMPLATE = (PROMPTS_DIR / "generate_t2i_prompt_user.txt").read_text()


def _build_messages(run_data: RunData, background_str: str) -> list[Message]:
    spec = run_data.scene_spec
    role_assignments_str = "\n".join(
        f"  {role} -> {spec.get_object_alias(obj)}"
        for role, obj in spec.role_assignments.items()
    )
    unique_objects = sorted(set(spec.role_assignments.values()))
    object_names_str = "\n".join(
        f"  - {spec.get_object_alias(obj)}" for obj in unique_objects
    )

    user_text = USER_PROMPT_TEMPLATE.format(
        scene_description=spec.scene_as_natural_language,
        action=spec.action,
        role_assignments=role_assignments_str,
        object_names=object_names_str,
        background_str=background_str,
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


async def generate_t2i_images(
    ctx: RunContext,
    seed: int | None,
) -> tuple[list[tuple[Path, str]], str]:
    config = ctx.run_data.config
    num_images = config.num_edit_chains
    t2i_model_name = config.t2i_model
    assert t2i_model_name is not None

    lm = ctx.get_vlm(config.blender_lm)
    t2i_model = ctx.get_t2i_model(t2i_model_name)
    image_seeds = derive_seeds(seed, num_images)
    t2i_dir = ctx.run_data.run_dir / "t2i"
    t2i_dir.mkdir(parents=True, exist_ok=True)

    lm_prompts: list[str] = []

    async def _generate_one(idx: int, img_seed: int | None) -> tuple[Path, str]:
        prompt_seed = derive_seed(img_seed, 1) if img_seed is not None else None
        bg_str = random.Random(prompt_seed).choice(config.background_strs)
        messages = _build_messages(ctx.run_data, bg_str)
        lm_prompts.append(messages_to_prompt_string(messages))
        t2i_prompt = await lm.infer_text(
            messages=messages,
            temperature=config.blender_lm_temperature,
            **({"seed": prompt_seed} if prompt_seed is not None else {}),
        )
        logger.info(f"T2I image {idx}: generated prompt: {t2i_prompt}")

        gen_seed = derive_seed(img_seed, 2) if img_seed is not None else None
        result_url = await t2i_model.generate(
            prompt=t2i_prompt,
            **({"seed": gen_seed} if gen_seed is not None else {}),
        )
        save_path = t2i_dir / f"t2i_{idx}.png"
        await download_image(result_url, save_path)
        return save_path, t2i_prompt

    results = await asyncio.gather(*[
        _generate_one(i, s) for i, s in enumerate(image_seeds)
    ])
    return list(results), lm_prompts[0]
