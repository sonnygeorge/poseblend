import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from poseblend.blender.schema import BlenderObjectSpec, ObjectPlacementParams


@dataclass
class BoundingBox:
    center: Vector
    corners: list[Vector]


def compute_combined_bbox(objects: list[bpy.types.Object]) -> BoundingBox:
    all_corners: list[Vector] = []
    for obj in objects:
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ Vector(corner))
    if not all_corners:
        return BoundingBox(center=Vector((0, 0, 0)), corners=[])
    min_x = min(v.x for v in all_corners)
    max_x = max(v.x for v in all_corners)
    min_y = min(v.y for v in all_corners)
    max_y = max(v.y for v in all_corners)
    min_z = min(v.z for v in all_corners)
    max_z = max(v.z for v in all_corners)
    center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
    corners = [
        Vector((min_x, min_y, min_z)),
        Vector((min_x, min_y, max_z)),
        Vector((min_x, max_y, min_z)),
        Vector((min_x, max_y, max_z)),
        Vector((max_x, min_y, min_z)),
        Vector((max_x, min_y, max_z)),
        Vector((max_x, max_y, min_z)),
        Vector((max_x, max_y, max_z)),
    ]
    return BoundingBox(center=center, corners=corners)


def pitch_tilt_to_unit_vector(pitch: float, tilt: float) -> Vector:
    el, az = pitch, tilt
    vec = Vector(
        (
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            -math.sin(el),
        )
    )
    vec.normalize()
    return vec


def place_objects(
    placements: list[ObjectPlacementParams],
    objects_by_name: dict[str, BlenderObjectSpec],
) -> list[bpy.types.Object]:
    placed: list[bpy.types.Object] = []
    for spec in placements:
        obj_spec = objects_by_name[spec.name]

        # Resolve the path to the Object collection inside the .blend file
        blend_path = Path(obj_spec.file_path)
        object_dir = str(blend_path / "Object")
        bpy.ops.wm.append(directory=object_dir, filename=obj_spec.name)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="BOUNDS")

        selected_obj = bpy.data.objects[obj_spec.name]
        bpy.ops.object.select_all(action="DESELECT")
        selected_obj.select_set(True)
        bpy.context.view_layer.objects.active = selected_obj

        # Set to origin before scaling
        selected_obj.location = (0, 0, 0)

        # Scale
        scale = obj_spec.scale_factor
        bpy.ops.transform.resize(value=(scale, scale, scale))

        # Compute placement
        x, y, z = spec.target_location
        bbox = [
            selected_obj.matrix_world @ Vector(corner)
            for corner in selected_obj.bound_box
        ]
        min_z = min(v.z for v in bbox)

        if spec.touching_ground:
            bpy.ops.transform.translate(value=(x, y, z - min_z))
        else:
            bpy.ops.transform.translate(value=(x, y, z))

        # Apply rotation: compensate for the object's default facing orientation
        selected_obj.rotation_mode = "XYZ"
        if spec.target_facing_direction is not None:
            default = obj_spec.default_facing_orientation or [0.0, 0.0, 0.0]
            rotation = [t - d for t, d in zip(spec.target_facing_direction, default)]
            selected_obj.rotation_euler = rotation

        bpy.ops.object.select_all(action="DESELECT")
        placed.append(selected_obj)
    return placed


def min_camera_distance_for_bbox(
    bbox: BoundingBox,
    camera_pitch: float,
    camera_tilt: float,
    camera_fov_angle_rads: float,
    camera_aspect_ratio: float,
) -> float:
    if not bbox.corners:
        return 0.0
    center = bbox.center
    corners = bbox.corners
    look_dir = pitch_tilt_to_unit_vector(camera_pitch, camera_tilt)
    rot = look_dir.to_track_quat("-Z", "Y").to_matrix()
    right = Vector(rot.col[0])
    up = Vector(rot.col[1])
    half_h = camera_fov_angle_rads / 2
    half_v = math.atan(math.tan(half_h) / camera_aspect_ratio)
    tan_h = math.tan(half_h)
    tan_v = math.tan(half_v)
    d_candidates: list[float] = []
    for p in corners:
        p_rel = p - center
        depth_offset = p_rel.dot(look_dir)
        x_cam = p_rel.dot(right)
        y_cam = p_rel.dot(up)
        min_depth = max(abs(x_cam) / tan_h, abs(y_cam) / tan_v)
        d_candidates.append(min_depth - depth_offset)
    return max(0.0, max(d_candidates))


def render_object_masks(
    placed_objects: list[bpy.types.Object],
) -> list[np.ndarray]:
    scene = bpy.context.scene
    render_args = scene.render

    orig_engine = render_args.engine
    orig_film_transparent = render_args.film_transparent
    orig_file_format = render_args.image_settings.file_format
    orig_color_mode = render_args.image_settings.color_mode
    orig_filepath = render_args.filepath
    orig_hide_render = {obj.name: obj.hide_render for obj in bpy.data.objects}

    render_args.engine = "BLENDER_EEVEE_NEXT"
    render_args.film_transparent = True
    render_args.image_settings.file_format = "PNG"
    render_args.image_settings.color_mode = "RGBA"

    masks: list[np.ndarray] = []
    h, w = render_args.resolution_y, render_args.resolution_x

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, target_obj in enumerate(placed_objects):
            for obj in bpy.data.objects:
                if obj.type == "MESH":
                    obj.hide_render = obj.name != target_obj.name

            tmp_path = os.path.join(tmp_dir, f"mask_{idx}.png")
            render_args.filepath = tmp_path
            bpy.ops.render.render(write_still=True)

            img = bpy.data.images.load(tmp_path)
            pixels = np.array(img.pixels[:]).reshape((h, w, 4))
            mask = (pixels[:, :, 3] > 0.5).astype(np.float32)
            masks.append(mask)
            bpy.data.images.remove(img)

    for obj in bpy.data.objects:
        if obj.name in orig_hide_render:
            obj.hide_render = orig_hide_render[obj.name]
    render_args.engine = orig_engine
    render_args.film_transparent = orig_film_transparent
    render_args.image_settings.file_format = orig_file_format
    render_args.image_settings.color_mode = orig_color_mode
    render_args.filepath = orig_filepath

    return masks


def save_masks(
    masks: list[np.ndarray],
    placed_objects: list[bpy.types.Object],
    mask_dir: str,
) -> None:
    if not masks:
        return
    h, w = masks[0].shape

    os.makedirs(mask_dir, exist_ok=True)

    def _write_mask(mask: np.ndarray, filepath: str) -> None:
        img = bpy.data.images.new("_tmp_mask_save", width=w, height=h)
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, 0] = mask
        rgba[:, :, 1] = mask
        rgba[:, :, 2] = mask
        rgba[:, :, 3] = 1.0
        img.pixels[:] = rgba.flatten()
        img.filepath_raw = filepath
        img.file_format = "PNG"
        img.save()
        bpy.data.images.remove(img)

    for mask, obj in zip(masks, placed_objects):
        path = os.path.join(mask_dir, f"{obj.name}.png")
        _write_mask(mask, path)

    combined = np.clip(np.sum(masks, axis=0), 0.0, 1.0)
    path = os.path.join(mask_dir, "all.png")
    _write_mask(combined, path)
