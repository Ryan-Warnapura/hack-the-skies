"""Uncertain local appearance estimates, never photogrammetry.

Optional REALITYFORGE_ONNX_MODEL contract: local model with one RGB float32
[1,3,H,W] input, scaled 0..1, and [1,4,H,W] logits: background/tree/person/vehicle.
Dynamic H/W defaults to 256. Component counts are estimates, not detections.
No weights or packages are downloaded; missing runtime/model uses color fallback.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

STYLE_MATERIALS = {
    "brick": "minecraft:bricks", "stone": "minecraft:stone_bricks",
    "wood": "minecraft:oak_planks", "modern": "minecraft:white_concrete",
}


def _gps_degrees(value):
    return float(value[0]) + float(value[1]) / 60 + float(value[2]) / 3600


def extract_photo_metadata(original: Image.Image) -> dict:
    """Extract camera metadata before uploads are re-encoded."""
    result = {}
    try:
        gps = original.getexif().get_ifd(34853)
    except (KeyError, TypeError, ValueError, AttributeError, OSError):
        gps = {}
    try:
        if 2 in gps and 4 in gps:
            lat, lon = _gps_degrees(gps[2]), _gps_degrees(gps[4])
            lat *= -1 if str(gps.get(1, "N")).upper() == "S" else 1
            lon *= -1 if str(gps.get(3, "E")).upper() == "W" else 1
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                result.update(exif_lat=lat, exif_lon=lon)
        if 17 in gps:
            heading = float(gps[17])
            if np.isfinite(heading):
                result["exif_heading"] = round(heading % 360, 3)
                result["exif_heading_reference"] = {"T": "true", "M": "magnetic"}.get(str(gps.get(16, "")).strip().upper(), "unknown")
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        pass
    xmp = original.info.get("xmp", original.info.get("XML:com.adobe.xmp", ""))
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", errors="replace")
    if re.search(r"GPano:ProjectionType(?:\s*=\s*['\"]|\s*>\s*)equirectangular", str(xmp), re.I):
        result["metadata_projection"] = "panorama"
    return result


def trim_uniform_borders(im: Image.Image) -> tuple[Image.Image, list[int]]:
    """Trim narrow neutral bars with a sharp edge; leave complex UI for manual crop."""
    sample = im.copy()
    sample.thumbnail((512, 512))
    a = np.asarray(sample.convert("RGB"), dtype=np.float32)
    h, w = a.shape[:2]

    def band_size(lines):
        maximum = max(1, int(len(lines) * .12))
        reference = np.median(lines[0], axis=0)
        if reference.max() - reference.min() > 8 or not (reference.mean() < 30 or reference.mean() > 245):
            return 0
        size = 0
        for row in lines[:maximum + 1]:
            if float((np.max(np.abs(row - reference), axis=1) < 10).mean()) < .97:
                break
            size += 1
        if size < max(2, int(len(lines) * .015)) or size > maximum:
            return 0
        if float((np.max(np.abs(lines[size] - reference), axis=1) > 35).mean()) < .4:
            return 0
        return size

    top, bottom = band_size(a), band_size(a[::-1])
    transposed = a.transpose(1, 0, 2)
    left, right = band_size(transposed), band_size(transposed[::-1])
    box = [round(left * im.width / w), round(top * im.height / h),
           im.width - round(right * im.width / w), im.height - round(bottom * im.height / h)]
    return im.crop(box), box


def _component_count(mask: np.ndarray, minimum: int = 12) -> int:
    """Connected patches may merge multiple crowns or split one tree."""
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    total = 0
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        size = 0
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if size >= minimum:
            total += 1
    return total


def _offline_segmentation(im: Image.Image) -> dict:
    model_path = os.environ.get("REALITYFORGE_ONNX_MODEL")
    if not model_path:
        return {"status": "not_configured"}
    try:
        if not Path(model_path).is_file():
            return {"status": "unavailable", "reason": "Configured model file is missing."}
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or shape[1] not in (3, "channels", None):
            raise ValueError("Expected NCHW RGB model input")
        height = shape[2] if isinstance(shape[2], int) else 256
        width = shape[3] if isinstance(shape[3], int) else 256
        if not 16 <= width <= 2048 or not 16 <= height <= 2048:
            raise ValueError("Unsupported model input size")
        rgb = np.asarray(im.resize((width, height)), dtype=np.float32) / 255
        logits = np.asarray(session.run(None, {model_input.name: rgb.transpose(2, 0, 1)[None]})[0])
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 4 or not np.isfinite(logits).all():
            raise ValueError("Expected four-class NCHW logits")
        classes = logits[0].argmax(axis=0).astype(np.uint8)
        classes = np.asarray(Image.fromarray(classes).resize((128, 128), Image.Resampling.NEAREST))
        return {"status": "available", "estimated_counts": {
            name: min(100, _component_count(classes == label, 8))
            for name, label in (("trees", 1), ("people", 2), ("vehicles", 3))}, "confidence": .4}
    except ImportError:
        return {"status": "unavailable", "reason": "Optional onnxruntime package is not installed."}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"Model could not be used ({type(exc).__name__}); color fallback active."}


def analyze_photo(path: Path, *, metadata: dict | None = None, crop: list[float] | None = None,
                  projection: str = "auto", heading: float | None = None, fov: float = 70,
                  tree_style: str = "auto", output_path: Path | None = None) -> dict:
    with Image.open(path) as original:
        original.load()
        metadata = extract_photo_metadata(original) if metadata is None else metadata
        im = ImageOps.exif_transpose(original).convert("RGB")
    original_size = im.size
    if crop is not None:
        if len(crop) != 4 or not all(np.isfinite(v) and 0 <= v <= 1 for v in crop) or crop[2] <= crop[0] or crop[3] <= crop[1]:
            raise ValueError("Crop must be normalized [left, top, right, bottom] with positive area.")
        box = [round(crop[0] * im.width), round(crop[1] * im.height), round(crop[2] * im.width), round(crop[3] * im.height)]
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            raise ValueError("Crop must be at least 8 pixels wide and high.")
        im = im.crop(box)
    else:
        im, box = trim_uniform_borders(im)
    if output_path:
        im.save(output_path, "JPEG", quality=92)
    width, height = im.size
    segment = _offline_segmentation(im)
    sample = im.copy()
    sample.thumbnail((128, 128))
    rgb = np.asarray(sample, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vegetation = (g > r * 1.10) & (g > b * 1.08) & (g > 40)
    sky = (b > r * 1.12) & (b > g * 1.01) & (b > 100)
    autumn = (r > g * 1.08) & (g > b * 1.35) & (g > 55) & (r > 100)
    usable = (~vegetation) & (~sky) & (rgb.mean(axis=-1) > 35) & (rgb.mean(axis=-1) < 240)
    center = np.zeros_like(usable)
    center[int(len(center) * .18):max(1, int(len(center) * .80))] = True
    pixels = rgb[usable & center]
    if len(pixels) < 20:
        pixels = rgb.reshape(-1, 3)
    quantized = (pixels.astype(np.uint8) // 24).astype(np.int32)
    codes = quantized[:, 0] * 121 + quantized[:, 1] * 11 + quantized[:, 2]
    dominant_bin = int(np.bincount(codes).argmax())
    dominant = np.median(pixels[codes == dominant_bin], axis=0).astype(int)
    median = np.median(pixels, axis=0)
    red, green, blue = median
    if red > green * 1.15 and red > blue * 1.25:
        style = "brick"
    elif red > blue * 1.22 and green > blue * 1.08 and red < 185:
        style = "wood"
    elif median.mean() > 175 and float(median.max() - median.min()) < 38:
        style = "modern"
    else:
        style = "stone"
    vegetation_ratio = round(float(vegetation.mean()), 3)
    # Brown architecture can mimic fall foliage: require some green evidence.
    season = "green" if vegetation_ratio > .06 else "unknown"
    if float(autumn.mean()) > .09 and vegetation_ratio > .025:
        season = "autumn"
    gray = rgb.mean(axis=-1)
    dark = (gray < 105) & ((rgb.max(axis=-1) - rgb.min(axis=-1)) < 28)
    upper = dark[:max(1, int(len(dark) * .65))]
    edges = np.abs(np.diff(upper.astype(np.int8), axis=1)).mean() if upper.shape[1] > 1 else 0
    if vegetation_ratio < .025 and float(sky.mean()) > .15 and .015 < float(upper.mean()) < .24 and edges > .09:
        season = "bare_candidate"
    crown_mask = vegetation.copy()
    crown_mask[int(len(crown_mask) * .85):] = False
    count = min(30, _component_count(crown_mask, minimum=max(12, int(crown_mask.size * .003))))
    tree_count = count if count > 0 and .015 < vegetation_ratio < .75 else None
    colors = []
    for mask in (vegetation, autumn if season == "autumn" else np.zeros_like(autumn)):
        if mask.sum() >= 12:
            color = np.median(rgb[mask], axis=0).astype(int)
            colors.append("#" + "".join(f"{v:02x}" for v in color))
    effective_projection = projection if projection != "auto" else metadata.get("metadata_projection", "perspective")
    was_cropped = box != [0, 0, *original_size]
    if was_cropped and projection == "auto" and effective_projection == "panorama":
        effective_projection = "perspective"
    if heading is None and metadata.get("exif_heading_reference") == "true":
        heading = metadata.get("exif_heading")
    return {
        **metadata, "width": width, "height": height, "original_width": original_size[0], "original_height": original_size[1],
        "projection": effective_projection, "projection_setting": projection,
        "projection_source": "user" if projection != "auto" else ("metadata" if metadata.get("metadata_projection") and not was_cropped else "default"),
        "wide_image_candidate": width / max(height, 1) >= 1.8,
        "heading": heading, "fov": 360 if effective_projection == "panorama" else fov, "tree_style": tree_style,
        "crop": crop, "applied_crop": box, "auto_trimmed": crop is None and was_cropped,
        "dominant_color": "#" + "".join(f"{int(v):02x}" for v in dominant),
        "suggested_style": style, "material": STYLE_MATERIALS[style],
        "vegetation_ratio": vegetation_ratio, "vegetation_state": season, "foliage_colors": colors,
        "estimated_counts": segment.get("estimated_counts", {"trees": tree_count, "people": None, "vehicles": None}),
        "tree_count_confidence": segment.get("confidence", .15 if tree_count is not None else 0),
        "segmentation": segment, "sky_ratio": round(float(sky.mean()), 3),
        "method": "offline_segmentation_and_color" if segment["status"] == "available" else "local_color_heuristics",
        "confidence": .35,
        "summary": f"{style.title()} palette; {round(vegetation_ratio * 100)}% green pixels; vegetation {season.replace('_', ' ')}. Appearance estimate, not 3D reconstruction.",
        "limitations": ["No depth, calibrated geometry, or unseen facade reconstruction.",
                        "Color patches are uncertain vegetation estimates, not verified object detections.",
                        "Heading and distance place approximate objects; magnetic EXIF headings need a true-heading override.",
                        "Automatic trimming removes only uniform edge bars. Crop complex screenshot interfaces manually."],
    }
