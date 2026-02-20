from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from poseblend.run_context import RunContext

# Padding added around the tight object bounding box before cropping, as a
# fraction of the box dimensions (e.g. 0.05 = 5% padding on each side)
CROP_PADDING_FRACTION = 0.045


def _load_masks_from_dir(mask_dir: Path) -> list[np.ndarray]:
    """Load all per-object mask PNGs from a mask directory and return them as
    binary numpy arrays. Skips the combined 'all.png' mask.
    """
    masks = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        if mask_path.stem == "all":
            continue
        img = Image.open(mask_path).convert("L")
        arr = (np.array(img) > 127).astype(np.float32)
        masks.append(arr)
    return masks


def _mask_bbox(combined_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) of the tight bounding box around non-zero
    pixels in PIL coordinate space (origin top-left).
    """
    rows = np.any(combined_mask > 0, axis=1)
    cols = np.any(combined_mask > 0, axis=0)
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    return int(row_indices[0]), int(row_indices[-1]), int(col_indices[0]), int(col_indices[-1])


def _padded_crop_box(
    top: int, bottom: int, left: int, right: int,
    img_h: int, img_w: int, aspect_ratio: float,
) -> tuple[int, int, int, int]:
    """Expand the tight bbox with padding, then grow to match the target aspect
    ratio. Returns (top, bottom, left, right) clamped to image bounds.
    """
    box_h = bottom - top
    box_w = right - left

    pad_h = int(box_h * CROP_PADDING_FRACTION)
    pad_w = int(box_w * CROP_PADDING_FRACTION)
    top = max(0, top - pad_h)
    bottom = min(img_h - 1, bottom + pad_h)
    left = max(0, left - pad_w)
    right = min(img_w - 1, right + pad_w)

    # Expand to match aspect ratio (width / height)
    box_h = bottom - top
    box_w = right - left
    current_ratio = box_w / max(box_h, 1)

    if current_ratio < aspect_ratio:
        target_w = int(box_h * aspect_ratio)
        expand = target_w - box_w
        left -= expand // 2
        right += expand - expand // 2
    else:
        target_h = int(box_w / aspect_ratio)
        expand = target_h - box_h
        top -= expand // 2
        bottom += expand - expand // 2

    # Clamp to image bounds, shifting the box inward if it overflows
    if top < 0:
        bottom -= top
        top = 0
    if bottom >= img_h:
        top -= bottom - (img_h - 1)
        bottom = img_h - 1
    if left < 0:
        right -= left
        left = 0
    if right >= img_w:
        left -= right - (img_w - 1)
        right = img_w - 1

    top = max(0, top)
    bottom = min(img_h - 1, bottom)
    left = max(0, left)
    right = min(img_w - 1, right)

    return top, bottom, left, right


def _crop_and_upscale_image(
    image_path: Path, crop_box: tuple[int, int, int, int], orig_size: tuple[int, int],
    resample: Image.Resampling = Image.LANCZOS,
) -> None:
    img = Image.open(image_path)
    cropped = img.crop(crop_box).resize(orig_size, resample)
    cropped.save(image_path)


def postprocess_render(
    image_path: Path, mask_dir: Path, aspect_ratio: float,
) -> None:
    """Crop a rendered image and its masks tightly around the objects, preserving
    aspect ratio, then upscale back to original resolution.
    """
    masks = _load_masks_from_dir(mask_dir)
    if not masks:
        return

    combined = np.clip(np.sum(masks, axis=0), 0.0, 1.0)
    if not np.any(combined > 0):
        return

    img = Image.open(image_path)
    orig_w, orig_h = img.size
    img.close()

    top, bottom, left, right = _mask_bbox(combined)
    top, bottom, left, right = _padded_crop_box(
        top, bottom, left, right, orig_h, orig_w, aspect_ratio,
    )

    crop_box = (left, top, right + 1, bottom + 1)
    orig_size = (orig_w, orig_h)

    _crop_and_upscale_image(image_path, crop_box, orig_size, Image.LANCZOS)

    for mask_path in mask_dir.glob("*.png"):
        _crop_and_upscale_image(mask_path, crop_box, orig_size, Image.NEAREST)

    crop_w = right - left + 1
    crop_h = bottom - top + 1
    logger.info(
        f"Cropped render {image_path.name}: "
        f"({crop_w}x{crop_h}) -> ({orig_w}x{orig_h})"
    )


def postprocess_all_renders(ctx: RunContext) -> None:
    """Apply tight crop + upscale to all renders across all scenes."""
    config = ctx.run_data.config
    aspect_ratio = config.render_resolution_x / config.render_resolution_y

    for scene in ctx.run_data.scenes:
        if not scene.renders:
            continue
        for render in scene.renders:
            if render.image_path is None or render.mask_dir_path is None:
                continue
            postprocess_render(render.image_path, render.mask_dir_path, aspect_ratio)
