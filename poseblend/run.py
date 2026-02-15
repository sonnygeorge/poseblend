import time
from collections.abc import Callable

from loguru import logger

from poseblend.exceptions import NoSceneGoodEnoughError
from poseblend.pipeline_steps.generate_blender_params import generate_blender_params
from poseblend.pipeline_steps.render_scenes import render_all_scenes
from poseblend.pipeline_steps.perform_edits import perform_all_edits
from poseblend.pipeline_steps.score_renders import score_all_renders
from poseblend.pipeline_steps.select_renders import select_renders
from poseblend.run_context import RunContext
from poseblend.schema.run_data import BlenderScene, RunData
from poseblend.utils import derive_seeds


async def run_poseblend(
    config_path: str,
    scene_path: str,
    blender_object_data_path: str,
    on_update: Callable[[], None] | None = None,
    run_data: RunData | None = None,
) -> RunData:
    run_start = time.time()
    # Initialize run data and run context objects
    if run_data is None:
        run_data = RunData.from_input_yaml_paths(config_path, scene_path, blender_object_data_path)
    ctx = RunContext(run_data, on_update=on_update)
    n_scenes = run_data.config.num_blender_scenes
    n_renders = run_data.config.num_renders
    # Derive per-scene seeds from config seed
    scene_seeds = derive_seeds(run_data.config.seed, n_scenes)
    # Make async llm calls to generate params for all candidate blender scenes
    step_start = time.time()
    params_list = await generate_blender_params(ctx, seeds=scene_seeds)
    logger.info(
        f"Generated {n_scenes} blender params in {time.time() - step_start:.2f}s "
        f"(total elapsed: {time.time() - run_start:.2f}s)"
    )
    run_data.scenes = [
        BlenderScene(scene_id=i + 1, seed=seed, params=params)
        for i, (params, seed) in enumerate(zip(params_list, scene_seeds))
    ]
    ctx.on_run_data_changed()
    # Render all blender scenes (spawns blender subprocesses)
    step_start = time.time()
    await render_all_scenes(ctx)
    total_renders = n_scenes * n_renders
    logger.info(
        f"Rendered {total_renders} renders ({n_renders} renders for {n_scenes} scenes) "
        f"in {time.time() - step_start:.2f}s (total elapsed: {time.time() - run_start:.2f}s)"
    )
    # Score all renders via critic VLM
    step_start = time.time()
    await score_all_renders(ctx)
    logger.info(
        f"Scored {total_renders} renders in {time.time() - step_start:.2f}s "
        f"(total elapsed: {time.time() - run_start:.2f}s)"
    )
    # Select best scene and top renders
    try:
        selected_renders = select_renders(ctx)
    except NoSceneGoodEnoughError as e:
        logger.error(f"Pipeline stopped: {e}")
        ctx.on_run_data_changed()
        logger.info(f"Total elapsed time: {time.time() - run_start:.2f}s")
        return run_data
    ctx.on_run_data_changed()
    # Perform edits on selected renders
    step_start = time.time()
    await perform_all_edits(ctx, selected_renders)
    logger.info(
        f"Performed edits on {len(selected_renders)} edit chains "
        f"in {time.time() - step_start:.2f}s (total elapsed: {time.time() - run_start:.2f}s)"
    )
    logger.info(f"Total elapsed time: {time.time() - run_start:.2f}s")
    return run_data
