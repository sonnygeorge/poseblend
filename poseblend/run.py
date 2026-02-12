import time

from loguru import logger

from poseblend.pipeline_steps.generate_blender_params import generate_blender_params
from poseblend.pipeline_steps.render_scenes import render_all_scenes
from poseblend.run_context import RunContext
from poseblend.schema.run_data import BlenderScene, RunData
from poseblend.utils import derive_seeds


async def run_poseblend(config_path: str, scene_path: str, blender_object_data_path: str) -> None:
    run_start = time.time()
    # Initialize run data and run context objects
    run_data = RunData.from_input_yaml_paths(config_path, scene_path, blender_object_data_path)
    ctx = RunContext(run_data)
    n_scenes = run_data.config.num_blender_scenes
    n_renders = run_data.config.num_renders
    # Derive per-scene seeds from config seed
    scene_seeds = derive_seeds(run_data.config.seed, n_scenes)
    # Make async llm calls to generate params for all candidate blender scenes
    t0 = time.time()
    params_list = await generate_blender_params(ctx, seeds=scene_seeds)
    logger.info(
        f"Generated {n_scenes} blender params in {time.time() - t0:.2f}s "
        f"(total elapsed: {time.time() - run_start:.2f}s)"
    )
    run_data.scenes = [
        BlenderScene(scene_id=i + 1, seed=seed, params=params)
        for i, (params, seed) in enumerate(zip(params_list, scene_seeds))
    ]
    # Render all blender scenes (spawns blender subprocesses)
    t0 = time.time()
    await render_all_scenes(ctx)
    total_renders = n_scenes * n_renders
    logger.info(
        f"Rendered {total_renders} renders ({n_renders} renders for {n_scenes} scenes) "
        f"in {time.time() - t0:.2f}s (total elapsed: {time.time() - run_start:.2f}s)"
    )
    # TODO: score renders, select best scene, run edit chains...
    # Save run data
    run_data.save()
    logger.info(f"Total elapsed time: {time.time() - run_start:.2f}s")
