import base64
import json
import mimetypes
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
import yaml

from poseblend.models.vlm.base import Message, MessageContent
from poseblend.schema.lm_outputs import CriticResult

if TYPE_CHECKING:
    from poseblend.run_context import RunContext

PROMPTS_DIR = Path(__file__).parent / "prompts"
CRITIC_SYSTEM_PROMPT = (PROMPTS_DIR / "critic_system.txt").read_text()
CRITIC_USER_PROMPT_TEMPLATE = (PROMPTS_DIR / "critic_user.txt").read_text()


def load_yaml(path: os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: os.PathLike) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def derive_seed(base_seed: int, n: int) -> int:
    """Advance a PRNG seeded with ``base_seed`` by ``n`` steps and return the last value."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    rng = random.Random(base_seed)
    for _ in range(n):
        derived = rng.randint(0, 2**32 - 1)
    return derived


def derive_seeds(base_seed: int | None, count: int) -> list[int | None]:
    """Derive ``count`` deterministic seeds from ``base_seed`` (returns ``[None] * count`` if None)."""
    if base_seed is None:
        return [None] * count
    return [derive_seed(base_seed, i + 1) for i in range(count)]


def sample_from_discrete_distribution(
    distribution: dict[str, float], seed: int | None = None
) -> str:
    """Randomly select a key from a discrete probability distribution.

    Args:
        distribution: A mapping of string keys to their probability weights.
        seed: Optional random seed for reproducibility.

    Returns:
        A single key sampled according to the given probability weights.
    """
    keys = list(distribution.keys())
    weights = list(distribution.values())
    rng = random.Random(seed)
    return rng.choices(keys, weights=weights, k=1)[0]


def grammatical_join(items: list[str]) -> str:
    """Join strings into a grammatical, sentence-insertable listing.

    Examples:
        >>> grammatical_join(["foo"])
        'foo'
        >>> grammatical_join(["foo", "bar"])
        'foo and bar'
        >>> grammatical_join(["foo", "bar", "fizz"])
        'foo, bar, and fizz'
    """
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def image_path_to_data_uri(path: str | os.PathLike) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


async def download_image(url: str, save_path: Path) -> Path:
    """Download an image from ``url`` and write it to ``save_path``."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session, session.get(url) as resp:
        resp.raise_for_status()
        save_path.write_bytes(await resp.read())
    return save_path


def build_single_object_requirements(ctx: "RunContext") -> list[str]:
    unique_objects = sorted(set(ctx.run_data.scene_spec.role_assignments.values()))
    return [
        f"A single {obj} is visible in the image." for obj in unique_objects
    ]


async def invoke_critic(
    ctx: "RunContext",
    image_path: str | os.PathLike,
    requirement: str,
) -> CriticResult:
    vlm = ctx.get_vlm(ctx.run_data.config.critic_vlm)
    image_uri = image_path_to_data_uri(image_path)
    messages = [
        Message(
            role="system",
            content=[MessageContent(type="text", content=CRITIC_SYSTEM_PROMPT)],
        ),
        Message(
            role="user",
            content=[
                MessageContent(type="image_url", content=image_uri),
                MessageContent(
                    type="text",
                    content=CRITIC_USER_PROMPT_TEMPLATE.format(requirement=requirement),
                ),
            ],
        ),
    ]
    return await vlm.infer_structured(
        messages=messages,
        response_model=CriticResult,
        temperature=ctx.run_data.config.critic_vlm_temperature,
    )
