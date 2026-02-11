import os

import yaml
from PIL import Image

from poseblend.models.vlm.base import BaseVLM
from poseblend.schema.lm_outputs import YesNoAnswer


def load_yaml(path: os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ask_lm_yes_no_question_about_image(
    vlm: BaseVLM,
    question: str,
    image_path: os.PathLike | Image.Image,
) -> YesNoAnswer:
    pass  # TODO
