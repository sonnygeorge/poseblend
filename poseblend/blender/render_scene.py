import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

# Blender runs scripts with its own bundled Python, which doesn't have the project
# root on sys.path. We add it manually so that poseblend imports resolve correctly.
_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from poseblend.blender import config as cfg  # noqa: E402
from poseblend.blender.schema import (  # noqa: E402
    BlenderObjectSpec,
    ObjectPlacementParams,
    RenderJob,
)
from poseblend.blender.utils import (  # noqa: E402
    adjust_object_positions,
    compute_combined_bbox,
    elevation_azimuth_to_unit_vector,
    min_camera_distance_for_bbox,
    place_objects,
    render_object_masks,
    save_masks,
)

def _apply_grid_floor_material() -> None:
    floor = bpy.data.objects.get(cfg.FLOOR_OBJECT_NAME)
    if floor is None:
        return

    mat = bpy.data.materials.new("GridFloor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")

    tex_coord = nodes.new("ShaderNodeTexCoord")
    sep_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex_coord.outputs["Object"], sep_xyz.inputs["Vector"])

    spacing = cfg.FLOOR_GRID_SPACING
    line_w = cfg.FLOOR_GRID_LINE_WIDTH

    # Wrap negative coords into [0, spacing) via ((x % s) + s) % s
    mod_x1 = nodes.new("ShaderNodeMath")
    mod_x1.operation = "MODULO"
    mod_x1.inputs[1].default_value = spacing
    links.new(sep_xyz.outputs["X"], mod_x1.inputs[0])

    add_x = nodes.new("ShaderNodeMath")
    add_x.operation = "ADD"
    add_x.inputs[1].default_value = spacing
    links.new(mod_x1.outputs[0], add_x.inputs[0])

    mod_x2 = nodes.new("ShaderNodeMath")
    mod_x2.operation = "MODULO"
    mod_x2.inputs[1].default_value = spacing
    links.new(add_x.outputs[0], mod_x2.inputs[0])

    lt_x = nodes.new("ShaderNodeMath")
    lt_x.operation = "LESS_THAN"
    lt_x.inputs[1].default_value = line_w
    links.new(mod_x2.outputs[0], lt_x.inputs[0])

    mod_y1 = nodes.new("ShaderNodeMath")
    mod_y1.operation = "MODULO"
    mod_y1.inputs[1].default_value = spacing
    links.new(sep_xyz.outputs["Y"], mod_y1.inputs[0])

    add_y = nodes.new("ShaderNodeMath")
    add_y.operation = "ADD"
    add_y.inputs[1].default_value = spacing
    links.new(mod_y1.outputs[0], add_y.inputs[0])

    mod_y2 = nodes.new("ShaderNodeMath")
    mod_y2.operation = "MODULO"
    mod_y2.inputs[1].default_value = spacing
    links.new(add_y.outputs[0], mod_y2.inputs[0])

    lt_y = nodes.new("ShaderNodeMath")
    lt_y.operation = "LESS_THAN"
    lt_y.inputs[1].default_value = line_w
    links.new(mod_y2.outputs[0], lt_y.inputs[0])

    grid_mask = nodes.new("ShaderNodeMath")
    grid_mask.operation = "MAXIMUM"
    links.new(lt_x.outputs[0], grid_mask.inputs[0])
    links.new(lt_y.outputs[0], grid_mask.inputs[1])

    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    links.new(grid_mask.outputs[0], mix.inputs["Factor"])
    mix.inputs[6].default_value = cfg.FLOOR_GRID_BASE_COLOR
    mix.inputs[7].default_value = cfg.FLOOR_GRID_LINE_COLOR

    links.new(mix.outputs[2], bsdf.inputs["Color"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])

    floor.data.materials.clear()
    floor.data.materials.append(mat)


def _load_render_job(job_path: str) -> RenderJob:
    with open(job_path, "r") as f:
        data: dict = json.load(f)
    data["objects"] = [BlenderObjectSpec(**o) for o in data["objects"]]
    data["placements"] = [ObjectPlacementParams(**p) for p in data["placements"]]
    return RenderJob(**data)


def render_scene(job: RenderJob) -> dict:
    rng = np.random.default_rng(job.seed)

    bpy.ops.wm.open_mainfile(filepath=job.base_scene_path)

    render_args = bpy.context.scene.render
    render_args.engine = cfg.RENDER_ENGINE
    bpy.context.scene.cycles.seed = (job.seed or 0) % 2_147_483_647  # Keep in int32 range
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

    _apply_grid_floor_material()

    # Compute camera params early (needed by adjust_object_positions)
    camera_fov_rads = math.radians(job.camera_fov_degrees)
    aspect_ratio = job.resolution_x / job.resolution_y

    # Build object lookup from job specs
    objects_by_name = {obj.name: obj for obj in job.objects}

    # Place objects
    placed_objects = place_objects(job.placements, objects_by_name)

    # Contract positions toward centroid & enforce airspace between objects
    adjust_object_positions(
        placed_objects, job.placements, camera_fov_rads, aspect_ratio, job.contraction_strength,
    )

    # Ensure all placed objects are visible and in scene collection
    scene_collection = bpy.context.scene.collection
    for obj in placed_objects:
        obj.hide_render = False
        obj.hide_viewport = False
        if scene_collection.name not in (c.name for c in obj.users_collection):
            scene_collection.objects.link(obj)

    bbox_all = compute_combined_bbox(placed_objects)

    # Setup camera
    camera = bpy.data.objects["Camera"]
    camera.data.angle = camera_fov_rads
    render_args.resolution_x = job.resolution_x
    render_args.resolution_y = job.resolution_y
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
        for _attempt in range(cfg.MAX_CAMERA_ANGLE_SAMPLES):
            elevation = float(np.clip(
                rng.normal(cfg.CAMERA_ELEVATION_MEAN_RADS, cfg.CAMERA_ELEVATION_STD_RADS),
                cfg.CAMERA_ELEVATION_MIN_RADS,
                cfg.CAMERA_ELEVATION_MAX_RADS,
            ))
            azimuth = float(rng.uniform(-math.pi, math.pi))
            min_distance = min_camera_distance_for_bbox(
                bbox=bbox_all,
                camera_elevation=elevation,
                camera_azimuth=azimuth,
                camera_fov_angle_rads=camera_fov_rads,
                camera_aspect_ratio=aspect_ratio,
            )
            distance = float(rng.uniform(0.015, 0.05)) * min_distance + min_distance
            look_dir = elevation_azimuth_to_unit_vector(elevation, azimuth)
            camera.location = bbox_all.center - distance * look_dir
            camera.rotation_euler = look_dir.to_track_quat("-Z", "Y").to_euler("XYZ")

            masks = render_object_masks(placed_objects)
            has_overlap = bool(np.any(np.sum(masks, axis=0) > 1))
            if has_overlap:
                continue
            found_usable_angle = True
            break

        if not found_usable_angle:
            print(  # noqa: T201
                f"WARNING: Failed to find non-overlapping camera angle for render {render_id} "
                f"after {cfg.MAX_CAMERA_ANGLE_SAMPLES} attempts. Using last sampled angle."
            )

        # Save masks
        render_mask_dir = str(masks_dir / str(render_id))
        save_masks(masks, placed_objects, render_mask_dir)

        # Render final image
        render_path = str(renders_dir / f"{render_id}.png")
        render_args.filepath = render_path
        for attempt in range(cfg.MAX_RENDER_ATTEMPTS):
            try:
                bpy.ops.render.render(write_still=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"Render attempt {attempt + 1}/{cfg.MAX_RENDER_ATTEMPTS} failed: {e}")  # noqa: T201
        else:
            print(f"Gave up after {cfg.MAX_RENDER_ATTEMPTS} render attempts for render {render_id}.")  # noqa: T201

        manifest_renders.append({
            "image_path": render_path,
            "mask_dir_path": render_mask_dir,
        })

    # Optionally save the .blend file
    blend_path = None
    if job.save_blend_file:
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
