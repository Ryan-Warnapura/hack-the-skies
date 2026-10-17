"""Small, deterministic photo appearance hints. This is not photogrammetry.

No network or model weights are needed. OSM supplies object geometry; these hints
only select the material of nearby hand-built architectural assets.
"""
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageOps


STYLE_MATERIALS = {
    "brick": "minecraft:bricks",
    "stone": "minecraft:stone_bricks",
    "wood": "minecraft:oak_planks",
    "modern": "minecraft:white_concrete",
}


def _gps_degrees(value):
    return float(value[0]) + float(value[1]) / 60 + float(value[2]) / 3600


def analyze_photo(path: Path) -> dict:
    """Read a normal JPEG/PNG/WebP and return explainable palette suggestions.

    RGB color thresholds are deliberately exposed as heuristics in the result.
    A wide image is labelled a panorama candidate, never a calibrated camera.
    """
    with Image.open(path) as original:
        original.load()
        exif = original.getexif()
        gps = {}
        try:
            gps = exif.get_ifd(34853)
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
        im = ImageOps.exif_transpose(original).convert("RGB")
        width, height = im.size
        im.thumbnail((256, 256))
        rgb = np.asarray(im, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vegetation = (g > r * 1.10) & (g > b * 1.08) & (g > 40)
    sky = (b > r * 1.12) & (b > g * 1.01) & (b > 100)
    usable = (~vegetation) & (~sky) & (rgb.mean(axis=-1) > 35) & (rgb.mean(axis=-1) < 240)
    # Prefer the middle of the frame, reducing sky and pavement influence.
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
    panorama = width / max(height, 1) >= 1.8
    result = {
        "width": width, "height": height,
        "projection": "panorama" if panorama else "perspective",
        "dominant_color": "#" + "".join(f"{int(v):02x}" for v in dominant),
        "suggested_style": style, "material": STYLE_MATERIALS[style],
        "vegetation_ratio": vegetation_ratio,
        "sky_ratio": round(float(sky.mean()), 3),
        "method": "local_color_heuristics",
        "confidence": 0.35,
        "summary": f"{style.title()} palette suggestion; {round(vegetation_ratio * 100)}% green pixels. "
                   "Appearance hint only; geometry comes from map data.",
        "limitations": ["No depth, semantic segmentation, camera calibration, or unseen facade reconstruction.",
                        "Wide aspect ratio suggests a panorama; projection and field of view are uncalibrated."],
    }
    try:
        if 2 in gps and 4 in gps:
            lat, lon = _gps_degrees(gps[2]), _gps_degrees(gps[4])
            if str(gps.get(1, "N")).upper() == "S":
                lat *= -1
            if str(gps.get(3, "E")).upper() == "W":
                lon *= -1
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                result.update({"exif_lat": lat, "exif_lon": lon})
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        pass
    return result
