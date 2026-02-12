import asyncio

from poseblend.generate_blender_params import generate_blender_params
from poseblend.models.registry import get_vlm
from poseblend.models.semaphore_wrappers import SemaphoreVLM
from poseblend.schema.run_data import RunData


async def run_poseblend(config_path: str, scene_path: str, blender_object_data_path: str) -> None:
    run_data = RunData.from_input_yaml_paths(config_path, scene_path, blender_object_data_path)

    inference_semaphore = asyncio.Semaphore(run_data.config.max_concurrent_inference_requests)
    blender_vlm = SemaphoreVLM(get_vlm(run_data.config.blender_lm), inference_semaphore)

    run_data.blender_scene_params = await generate_blender_params(run_data, blender_vlm)
