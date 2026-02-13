import time

from loguru import logger

from poseblend.exceptions import NoSceneGoodEnoughError
from poseblend.pipeline_steps.generate_blender_params import generate_blender_params
from poseblend.pipeline_steps.render_scenes import render_all_scenes
from poseblend.pipeline_steps.score_renders import score_all_renders
from poseblend.pipeline_steps.select_renders import select_renders
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
        select_renders(ctx)  # TODO: run edit chains on selected_renders
    except NoSceneGoodEnoughError as e:
        logger.error(f"Pipeline stopped: {e}")
        run_data.save()
        logger.info(f"Total elapsed time: {time.time() - run_start:.2f}s")
        return

    # Save run data
    run_data.save()
    logger.info(f"Total elapsed time: {time.time() - run_start:.2f}s")


    # # TODO: DELETE THIS -- quick visual debug of all renders ranked by score
    # from PIL import Image, ImageDraw

    # selected_set = {id(r) for r in selected_renders}
    # all_render_info = []
    # for scene in run_data.scenes:
    #     for render in scene.renders:
    #         all_render_info.append((scene, render))
    # all_render_info.sort(key=lambda sr: sr[1].render_quality_score or 0.0, reverse=True)

    # if all_render_info:
    #     imgs = [Image.open(r.image_path) for _, r in all_render_info]
    #     w, h = imgs[0].size
    #     header_h = 48
    #     canvas = Image.new("RGB", (w * len(imgs), h + header_h), (0, 0, 0))
    #     draw = ImageDraw.Draw(canvas)
    #     for i, (img, (scene, render)) in enumerate(zip(imgs, all_render_info)):
    #         x = i * w
    #         score_str = f"{render.render_quality_score:.3f}" if render.render_quality_score is not None else "N/A"
    #         scene_color = "lime" if scene.is_selected else "white"
    #         draw.text((x + 4, 4), f"S{scene.scene_id}/R{render.render_id}  score={score_str}", fill=scene_color)
    #         if scene.is_selected:
    #             tag = "SELECTED FOR EDIT" if id(render) in selected_set else "NOT SELECTED"
    #             tag_color = "lime" if id(render) in selected_set else "red"
    #             draw.text((x + 4, 22), tag, fill=tag_color)
    #         canvas.paste(img, (x, header_h))
    #     canvas.save(run_data.run_dir / "debug_all_renders_ranked.png")
    #     canvas.show(title="All renders ranked by score")
