import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from poseblend.blender.schema import (
    BlenderObjectSpec,
    ObjectPlacementParams,
    RenderJob,
)
from poseblend.blender.utils import (
    compute_combined_bbox,
    min_camera_distance_for_bbox,
    pitch_tilt_to_unit_vector,
    place_objects,
    render_object_masks,
    save_masks,
)

MAX_CAMERA_ANGLE_SAMPLES = 50
MAX_RENDER_ATTEMPTS = 5

CAMERA_TILT_MIN_RADS = math.radians(7.5)
CAMERA_TILT_MAX_RADS = math.radians(81)
CAMERA_TILT_MEAN_RADS = math.radians(34)
CAMERA_TILT_STD_RADS = math.radians(12)


def _load_render_job(job_path: str) -> RenderJob:
    with open(job_path, "r") as f:
        data: dict = json.load(f)
    return RenderJob(
        base_scene_path=data["base_scene_path"],
        objects=[BlenderObjectSpec(**o) for o in data["objects"]],
        placements=[ObjectPlacementParams(**p) for p in data["placements"]],
        output_dir=data["output_dir"],
        num_renders=data["num_renders"],
        resolution_x=data["resolution_x"],
        resolution_y=data["resolution_y"],
        camera_fov_degrees=data["camera_fov_degrees"],
        seed=data.get("seed"),
    )


def render_scene(job: RenderJob) -> dict:
    rng = np.random.default_rng(job.seed)

    bpy.ops.wm.open_mainfile(filepath=job.base_scene_path)

    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.film_transparent = False

    # Set up world environment lighting (grey ambient light)
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = world.node_tree.nodes.new("ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
    bg_node.inputs["Strength"].default_value = 1.0

    # Build object lookup from job specs
    objects_by_name = {obj.name: obj for obj in job.objects}

    # Place objects
    placed_objects = place_objects(job.placements, objects_by_name)

    # Ensure all placed objects are visible and in scene collection
    scene_collection = bpy.context.scene.collection
    for obj in placed_objects:
        obj.hide_render = False
        obj.hide_viewport = False
        if scene_collection.name not in (c.name for c in obj.users_collection):
            scene_collection.objects.link(obj)

    bbox_all = compute_combined_bbox(placed_objects)

    # Setup camera
    camera_fov_rads = math.radians(job.camera_fov_degrees)
    camera = bpy.data.objects["Camera"]
    camera.data.angle = camera_fov_rads
    render_args.resolution_x = job.resolution_x
    render_args.resolution_y = job.resolution_y
    aspect_ratio = job.resolution_x / job.resolution_y
    render_args.resolution_percentage = 100
    camera.rotation_mode = "XYZ"

    # Ascertain output directories
    output_dir = Path(job.output_dir)
    renders_dir = output_dir / "renders"
    masks_dir = output_dir / "masks"
    renders_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_renders = []

    for i in range(job.num_renders):
        render_id = i + 1

        # Try to find a camera angle with no visual overlap between objects
        masks: list[np.ndarray] = []
        found_usable_angle = False
        for _attempt in range(MAX_CAMERA_ANGLE_SAMPLES):
            tilt = float(np.clip(
                rng.normal(CAMERA_TILT_MEAN_RADS, CAMERA_TILT_STD_RADS),
                CAMERA_TILT_MIN_RADS,
                CAMERA_TILT_MAX_RADS,
            ))
            pan = float(rng.uniform(-math.pi, math.pi))
            min_distance = min_camera_distance_for_bbox(
                bbox=bbox_all,
                camera_pitch=tilt,
                camera_tilt=pan,
                camera_fov_angle_rads=camera_fov_rads,
                camera_aspect_ratio=aspect_ratio,
            )
            distance = float(rng.uniform(0.015, 0.05)) * min_distance + min_distance
            look_dir = pitch_tilt_to_unit_vector(tilt, pan)
            camera.location = bbox_all.center - distance * look_dir
            camera.rotation_euler = look_dir.to_track_quat("-Z", "Y").to_euler("XYZ")

            masks = render_object_masks(placed_objects)
            has_overlap = bool(np.any(np.sum(masks, axis=0) > 1))
            if has_overlap:
                continue
            found_usable_angle = True
            break

        if not found_usable_angle:
            print(
                f"WARNING: Failed to find non-overlapping camera angle for render {render_id} "
                f"after {MAX_CAMERA_ANGLE_SAMPLES} attempts. Using last sampled angle."
            )

        # Save masks
        render_mask_dir = str(masks_dir / str(render_id))
        save_masks(masks, placed_objects, render_mask_dir)

        # Render final image
        render_path = str(renders_dir / f"{render_id}.png")
        render_args.filepath = render_path
        for attempt in range(MAX_RENDER_ATTEMPTS):
            try:
                bpy.ops.render.render(write_still=True)
                break
            except Exception as e:
                print(f"Render attempt {attempt + 1}/{MAX_RENDER_ATTEMPTS} failed: {e}")
        else:
            print(f"Gave up after {MAX_RENDER_ATTEMPTS} render attempts for render {render_id}.")

        manifest_renders.append({
            "image_path": render_path,
            "mask_dir_path": render_mask_dir,
        })

    # Save the .blend file
    blend_path = str(output_dir / "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    # Write manifest
    manifest = {
        "blend_file_path": blend_path,
        "renders": manifest_renders,
    }
    manifest_path = str(output_dir / "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    argv = sys.argv
    separator_idx = argv.index("--")
    script_args = argv[separator_idx + 1:]
    job_path = script_args[0]

    job = _load_render_job(job_path)
    render_scene(job)
