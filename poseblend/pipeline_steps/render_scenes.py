import asyncio
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from poseblend.blender.schema import BlenderObjectSpec, ObjectPlacementParams, RenderJob
from poseblend.run_context import RunContext
from poseblend.schema.run_data import BlenderScene, SceneRender

RENDER_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "blender" / "render_scene.py"
BLENDER_EXE_ENV_VAR = "BLENDER_EXE"


def _resolve_blender_exe() -> str:
    return os.environ.get(BLENDER_EXE_ENV_VAR, "blender")


def _build_render_job(
    scene: BlenderScene,
    scene_dir: Path,
    ctx: RunContext,
) -> dict:
    config = ctx.run_data.config
    registry = ctx.run_data.blender_object_registry

    objects = []
    for placement in scene.params.placements:
        meta = registry.objects[placement.name]
        file_path = str(Path(config.objects_dir_path) / meta.file)
        objects.append(asdict(BlenderObjectSpec(
            name=meta.name,
            file_path=file_path,
            scale_factor=meta.scale_factor,
            default_facing_orientation=meta.default_facing_orientation,
        )))

    placements = [
        asdict(ObjectPlacementParams(
            name=p.name,
            target_location=p.target_location,
            target_facing_direction=p.target_facing_direction,
            touching_ground=p.touching_ground,
        ))
        for p in scene.params.placements
    ]

    job = RenderJob(
        base_scene_path=config.base_scene_path,
        objects=[BlenderObjectSpec(**o) for o in objects],
        placements=[ObjectPlacementParams(**p) for p in placements],
        output_dir=str(scene_dir),
        num_renders=config.num_renders,
        resolution_x=config.render_resolution_x,
        resolution_y=config.render_resolution_y,
        camera_fov_degrees=config.camera_fov_degrees,
        seed=scene.seed,
        save_blend_file=config.save_blend_files,
    )
    return asdict(job)


async def _render_single_scene(
    scene: BlenderScene,
    ctx: RunContext,
) -> None:
    run_data = ctx.run_data
    scene_dir = run_data.run_dir / f"scene_{scene.scene_id}"

    job_dict = _build_render_job(scene, scene_dir, ctx)

    # Write job JSON to a temp file
    tmp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".json", delete=False, prefix="poseblend_job_"
    )
    try:
        json.dump(job_dict, tmp_file)
        tmp_file.close()

        blender_exe = _resolve_blender_exe()
        proc = await asyncio.create_subprocess_exec(
            blender_exe,
            "--background",
            "--python", str(RENDER_SCRIPT_PATH),
            "--", tmp_file.name,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        await proc.wait()

        if proc.returncode != 0:
            logger.error(f"Blender process for scene {scene.scene_id} failed (exit code {proc.returncode})")
            return

        # Read manifest
        manifest_path = scene_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error(f"No manifest found for scene {scene.scene_id} at {manifest_path}")
            return

        manifest = json.loads(manifest_path.read_text())

        blend_path = manifest.get("blend_file_path")
        scene.blend_file_path = Path(blend_path) if blend_path else None
        scene.renders = [
            SceneRender(
                render_id=i + 1,
                image_path=Path(r["image_path"]),
                mask_dir_path=Path(r["mask_dir_path"]),
            )
            for i, r in enumerate(manifest["renders"])
        ]
        ctx.on_run_data_changed()
        manifest_path.unlink()
    finally:
        Path(tmp_file.name).unlink()


async def render_all_scenes(ctx: RunContext) -> None:
    blender_exe = _resolve_blender_exe()

    # Validate Blender is callable
    try:
        proc = await asyncio.create_subprocess_exec(
            blender_exe, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Blender executable '{blender_exe}' returned non-zero exit code. "
                f"Set {BLENDER_EXE_ENV_VAR} environment variable to the correct path."
            )
    except FileNotFoundError:
        raise RuntimeError(
            f"Blender executable '{blender_exe}' not found. "
            f"Set {BLENDER_EXE_ENV_VAR} environment variable to the correct path."
        )

    # Ascertain run directory
    ctx.run_data.run_dir.mkdir(parents=True, exist_ok=True)

    async def _gated_render(scene: BlenderScene) -> None:
        async with ctx.blender_semaphore:
            await _render_single_scene(scene, ctx)

    tasks = [_gated_render(scene) for scene in ctx.run_data.scenes]
    await asyncio.gather(*tasks)

    scenes = ctx.run_data.scenes
    if all(not scene.renders for scene in scenes):
        raise RuntimeError(
            f"All {len(scenes)} scene(s) failed to render. "
            "Check Blender logs above for details."
        )
