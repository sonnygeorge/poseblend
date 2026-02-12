from poseblend.generate_blender_params import generate_blender_params
from poseblend.render_scenes import render_all_scenes
from poseblend.run_context import RunContext
from poseblend.schema.run_data import BlenderScene, RunData
from poseblend.utils import derive_seeds


async def run_poseblend(config_path: str, scene_path: str, blender_object_data_path: str) -> None:
    # Initialize run data and run context objects
    run_data = RunData.from_input_yaml_paths(config_path, scene_path, blender_object_data_path)
    ctx = RunContext(run_data)
    # Derive per-scene seeds from config seed
    scene_seeds = derive_seeds(run_data.config.seed, run_data.config.num_blender_scenes)
    # Make async llm calls to generate params for all candidate blender scenes
    params_list = await generate_blender_params(ctx, seeds=scene_seeds)
    run_data.scenes = [
        BlenderScene(scene_id=i + 1, seed=seed, params=params)
        for i, (params, seed) in enumerate(zip(params_list, scene_seeds))
    ]
    # Render all blender scenes (spawns blender subprocesses)
    await render_all_scenes(ctx)
    # TODO: score renders, select best scene, run edit chains...
    # Save run data
    run_data.save()
