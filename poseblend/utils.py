import json
import os
import random

import yaml
from PIL import Image

from poseblend.models.vlm.base import BaseVLM
from poseblend.schema.lm_outputs import YesNoAnswer


def load_yaml(path: os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: os.PathLike) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def derive_seed(base_seed: int, n: int) -> int:
    """Advance a PRNG seeded with ``base_seed`` by ``n`` steps and return the last value."""
    rng = random.Random(base_seed)
    for _ in range(n):
        derived = rng.randint(0, 2**32 - 1)
    return derived


def derive_seeds(base_seed: int | None, count: int) -> list[int | None]:
    """Derive ``count`` deterministic seeds from ``base_seed`` (returns ``[None] * count`` if None)."""
    if base_seed is None:
        return [None] * count
    return [derive_seed(base_seed, i + 1) for i in range(count)]


def ask_lm_yes_no_question_about_image(
    vlm: BaseVLM,
    question: str,
    image_path: os.PathLike | Image.Image,
) -> YesNoAnswer:
    pass  # TODO
