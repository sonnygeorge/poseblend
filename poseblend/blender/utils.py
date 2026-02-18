import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from poseblend.blender import config as cfg
from poseblend.blender.schema import BlenderObjectSpec, ObjectPlacementParams


@dataclass
class BoundingBox:
    center: Vector
    corners: list[Vector]


def compute_combined_bbox(objects: list[bpy.types.Object]) -> BoundingBox:
    all_corners: list[Vector] = []
    for obj in objects:
        all_corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
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


def bbox_volume(obj: bpy.types.Object) -> float:
    """Return the axis-aligned bounding box volume of a Blender object in world space."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [v.x for v in corners]
    ys = [v.y for v in corners]
    zs = [v.z for v in corners]
    return (max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs))


def build_bvh(obj: bpy.types.Object) -> BVHTree:
    """Build a BVH tree from an object's evaluated mesh in world space."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mesh.transform(obj.matrix_world)
    tree = BVHTree.FromPolygons(
        [v.co[:] for v in mesh.vertices],
        [p.vertices[:] for p in mesh.polygons],
    )
    eval_obj.to_mesh_clear()
    return tree


def mesh_surface_distance(
    tree_a: BVHTree, tree_b: BVHTree,
    obj_a: bpy.types.Object, obj_b: bpy.types.Object,
) -> tuple[float, Vector]:
    """Compute the approximate minimum distance between two meshes using BVH trees.

    Samples the vertices of each mesh and finds the closest surface point on the
    other mesh's BVH. Returns (distance, direction_vector_from_a_to_b). This is
    not an exact minimum but is fast and accurate enough for repulsion purposes.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    best_dist = float("inf")
    best_pt_a = Vector((0, 0, 0))
    best_pt_b = Vector((0, 0, 0))

    # Sample vertices of A, find closest surface point on B
    eval_a = obj_a.evaluated_get(depsgraph)
    mesh_a = eval_a.to_mesh()
    for v in mesh_a.vertices:
        world_co = obj_a.matrix_world @ v.co
        location, _normal, _index, dist = tree_b.find_nearest(world_co)
        if location is not None and dist < best_dist:
            best_dist = dist
            best_pt_a = world_co
            best_pt_b = location
    eval_a.to_mesh_clear()

    # Sample vertices of B, find closest surface point on A
    eval_b = obj_b.evaluated_get(depsgraph)
    mesh_b = eval_b.to_mesh()
    for v in mesh_b.vertices:
        world_co = obj_b.matrix_world @ v.co
        location, _normal, _index, dist = tree_a.find_nearest(world_co)
        if location is not None and dist < best_dist:
            best_dist = dist
            best_pt_a = location
            best_pt_b = world_co
    eval_b.to_mesh_clear()

    # Always push along center-to-center for a predictable repulsion direction,
    # even though distance is measured surface-to-surface.
    direction = obj_b.location - obj_a.location
    if direction.length < 1e-9:
        direction = Vector((1, 0, 0))

    return best_dist, direction.normalized()


def elevation_azimuth_to_unit_vector(elevation: float, azimuth: float) -> Vector:
    vec = Vector(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            -math.sin(elevation),
        )
    )
    vec.normalize()
    return vec


def place_objects(
    placements: list[ObjectPlacementParams],
    objects_by_name: dict[str, BlenderObjectSpec],
) -> list[bpy.types.Object]:
    """Load each object from its .blend file, scale it, position it at the target
    location, and orient it. Objects marked touching_ground are snapped so their
    lowest point sits on the Z=0 plane.
    """
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


def adjust_object_positions(
    placed_objects: list[bpy.types.Object],
    placements: list[ObjectPlacementParams],
    camera_fov_rads: float,
    aspect_ratio: float,
) -> None:
    """Tighten the scene layout in three phases: (1) contract all positions toward
    their centroid by a factor derived from how sparse the objects are relative to
    their volumes, (2) push any pairs that ended up too close back apart so the
    camera can see them without overlap, and (3) re-snap ground objects to Z=0.
    """
    if len(placed_objects) < 2:
        return

    # --- Phase 1: Contraction ---
    centroid = Vector((0, 0, 0))
    for obj in placed_objects:
        centroid += obj.location
    centroid /= len(placed_objects)

    total_obj_volume = sum(bbox_volume(obj) for obj in placed_objects)
    combined_bbox = compute_combined_bbox(placed_objects)
    if not combined_bbox.corners:
        return
    cx = [v.x for v in combined_bbox.corners]
    cy = [v.y for v in combined_bbox.corners]
    cz = [v.z for v in combined_bbox.corners]
    envelope_volume = (max(cx) - min(cx)) * (max(cy) - min(cy)) * (max(cz) - min(cz))

    if envelope_volume > 0:
        density = total_obj_volume / envelope_volume
    else:
        density = 1.0

    s = max(cfg.S_MIN, min(density / cfg.DENSITY_THRESHOLD, 1.0))

    print(f"[adjust] Phase 1: density={density:.4f}, contraction s={s:.4f}")  # noqa: T201
    for obj in placed_objects:
        old_loc = obj.location.copy()
        obj.location = centroid + s * (obj.location - centroid)
        delta = (obj.location - old_loc).length
        print(f"  {obj.name}: moved {delta:.4f} units inward")  # noqa: T201

    bpy.context.view_layer.update()

    # --- Phase 2: Airspace enforcement via mesh surfaces ---
    post_bbox = compute_combined_bbox(placed_objects)
    min_cam_dist = min_camera_distance_for_bbox(
        bbox=post_bbox,
        camera_elevation=cfg.CAMERA_ELEVATION_MEAN_RADS,
        camera_azimuth=0.0,
        camera_fov_angle_rads=camera_fov_rads,
        camera_aspect_ratio=aspect_ratio,
    )
    min_gap = cfg.AIRSPACE_FACTOR * min_cam_dist
    print(f"[adjust] Phase 2: min_cam_dist={min_cam_dist:.4f}, min_gap={min_gap:.4f}")  # noqa: T201

    for iteration in range(cfg.AIRSPACE_REPULSION_ITERATIONS):
        # Rebuild BVH trees each iteration since positions may have shifted
        bvh_trees = [build_bvh(obj) for obj in placed_objects]
        for i in range(len(placed_objects)):
            for j in range(i + 1, len(placed_objects)):
                surface_dist, direction = mesh_surface_distance(
                    bvh_trees[i], bvh_trees[j],
                    placed_objects[i], placed_objects[j],
                )
                if surface_dist < min_gap:
                    correction = (min_gap - surface_dist) / 2
                    placed_objects[i].location -= direction * correction
                    placed_objects[j].location += direction * correction
                    print(  # noqa: T201
                        f"  iter {iteration}: {placed_objects[i].name} <-> "
                        f"{placed_objects[j].name}: surface_dist={surface_dist:.4f}, "
                        f"pushed apart by {correction:.4f} each"
                    )
        bpy.context.view_layer.update()

    bpy.context.view_layer.update()

    # --- Phase 3: Ground re-snap ---
    for obj, spec in zip(placed_objects, placements):
        if spec.touching_ground:
            bbox_corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            snap_z = min(v.z for v in bbox_corners)
            obj.location.z -= snap_z
            print(f"[adjust] Phase 3: {obj.name} re-snapped by z={-snap_z:.4f}")  # noqa: T201


def min_camera_distance_for_bbox(
    bbox: BoundingBox,
    camera_elevation: float,
    camera_azimuth: float,
    camera_fov_angle_rads: float,
    camera_aspect_ratio: float,
) -> float:
    if not bbox.corners:
        return 0.0
    center = bbox.center
    corners = bbox.corners
    look_dir = elevation_azimuth_to_unit_vector(camera_elevation, camera_azimuth)
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
    return max(0.0, *d_candidates)


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

            tmp_path = str(Path(tmp_dir) / f"mask_{idx}.png")
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

    Path(mask_dir).mkdir(parents=True, exist_ok=True)

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
        path = str(Path(mask_dir) / f"{obj.name}.png")
        _write_mask(mask, path)

    combined = np.clip(np.sum(masks, axis=0), 0.0, 1.0)
    path = str(Path(mask_dir) / "all.png")
    _write_mask(combined, path)
