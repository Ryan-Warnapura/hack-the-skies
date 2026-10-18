"""Geographic features -> semantic 3-D model -> Minecraft voxels.

The intermediate scene stores real polygon footprints, object types, heights,
materials and provenance. Both OBJ export and voxelization consume that model.
Coordinates are x=east, y=up, z=south in blocks; arrays are ordered Y,Z,X.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .photos import STYLE_MATERIALS
from .assets import BLOCK_COLORS, part_mesh, voxelize_parts


DEFAULT_MATERIALS = {
    "wall": "minecraft:bricks", "roof": "minecraft:deepslate_tiles",
    "floor": "minecraft:oak_planks", "window": "minecraft:light_blue_stained_glass",
    "trim": "minecraft:quartz_block", "road": "minecraft:gray_concrete",
    "sidewalk": "minecraft:smooth_stone", "grass": "minecraft:grass_block",
    "trunk": "minecraft:oak_log[axis=y]", "leaves": "minecraft:oak_leaves[persistent=true]",
    "water": "minecraft:water[level=0]",
}

MATERIAL_COLORS = {
    "bricks": (161, 87, 68), "stone_bricks": (130, 130, 125), "oak_planks": (171, 137, 84),
    "white_concrete": (221, 224, 215), "sandstone": (212, 195, 143),
    "deepslate_tiles": (61, 67, 73), "dark_oak_planks": (77, 55, 34),
    "red_terracotta": (151, 75, 53), "light_blue_stained_glass": (105, 169, 191),
    "quartz_block": (223, 222, 209), "gray_concrete": (65, 70, 75),
    "smooth_stone": (158, 159, 152), "grass_block": (91, 141, 64),
    "oak_log": (103, 81, 51), "oak_leaves": (65, 124, 51),
    "water": (54, 128, 192), "white_concrete_powder": (220, 220, 210),
}


def material_color(material: str) -> str:
    name = material.split(":")[-1].split("[")[0]
    if name in BLOCK_COLORS:
        return BLOCK_COLORS[name]
    return "#" + "".join(f"{v:02x}" for v in MATERIAL_COLORS.get(name, (145, 145, 140)))


def _number(value, default):
    try:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group()) if match else default
    except (TypeError, ValueError):
        return default


def _projection(bbox, scale):
    west, south, east, north = map(float, bbox)
    coslat = math.cos(math.radians((north + south) / 2))
    width = max(1, math.ceil((east - west) * 111320 * coslat / scale - 1e-7))
    depth = max(1, math.ceil((north - south) * 111320 / scale - 1e-7))

    def project(lon, lat):
        return [round((float(lon) - west) * 111320 * coslat / scale, 3),
                round((north - float(lat)) * 111320 / scale, 3)]

    return width, depth, project


def _join_rings(parts):
    """Join the outer/inner ways of simple OSM multipolygons by endpoints."""
    remaining = [list(p) for p in parts if len(p) >= 2]
    result = []
    while remaining:
        chain = remaining.pop(0)
        changed = True
        while changed and chain[0] != chain[-1]:
            changed = False
            for i, part in enumerate(remaining):
                if chain[-1] == part[0]:
                    chain.extend(part[1:])
                elif chain[-1] == part[-1]:
                    chain.extend(reversed(part[:-1]))
                elif chain[0] == part[-1]:
                    chain = part[:-1] + chain
                elif chain[0] == part[0]:
                    chain = list(reversed(part[1:])) + chain
                else:
                    continue
                remaining.pop(i)
                changed = True
                break
        if len(chain) >= 4 and chain[0] == chain[-1]:
            result.append(chain[:-1])
    return result


def _point_inside(point, poly):
    x, z = point
    inside = False
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if (a[1] > z) != (b[1] > z):
            cross = (b[0] - a[0]) * (z - a[1]) / (b[1] - a[1]) + a[0]
            if x < cross:
                inside = not inside
    return inside


def _clip_polygon(poly, width, depth):
    """Sutherland-Hodgman clipping, preserving oblique footprint edges."""
    vertices = poly
    for axis, bound, lower in [(0, 0, True), (0, width - .01, False),
                               (1, 0, True), (1, depth - .01, False)]:
        if not vertices:
            break
        output = []
        prev = vertices[-1]
        prev_inside = prev[axis] >= bound if lower else prev[axis] <= bound
        for curr in vertices:
            curr_inside = curr[axis] >= bound if lower else curr[axis] <= bound
            if prev_inside != curr_inside:
                t = (bound - prev[axis]) / (curr[axis] - prev[axis])
                output.append([round(prev[0] + t * (curr[0] - prev[0]), 3),
                               round(prev[1] + t * (curr[1] - prev[1]), 3)])
            if curr_inside:
                output.append(curr)
            prev, prev_inside = curr, curr_inside
        vertices = output
    return vertices


def _terrain_height(terrain, x, z, width, depth):
    grid = terrain["heights"]
    rows, cols = len(grid), len(grid[0])
    gx = max(0, min(cols - 1, x / max(1, width - 1) * (cols - 1)))
    gz = max(0, min(rows - 1, z / max(1, depth - 1) * (rows - 1)))
    x0, z0 = int(gx), int(gz)
    x1, z1 = min(x0 + 1, cols - 1), min(z0 + 1, rows - 1)
    tx, tz = gx - x0, gz - z0
    return (grid[z0][x0] * (1 - tx) + grid[z0][x1] * tx) * (1 - tz) + \
        (grid[z1][x0] * (1 - tx) + grid[z1][x1] * tx) * tz


def _clip_road(a, b, width, depth):
    """Liang-Barsky segment clipping keeps huge OSM ways bounded before meshing."""
    dx, dz = b[0] - a[0], b[1] - a[1]
    start, end = 0., 1.
    for p, q in [(-dx, a[0]), (dx, width - .01 - a[0]),
                 (-dz, a[1]), (dz, depth - .01 - a[1])]:
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            start = max(start, q / p)
        else:
            end = min(end, q / p)
        if start > end:
            return None
    return [[round(a[0] + t * dx, 3), round(a[1] + t * dz, 3)] for t in (start, end)]


def build_scene(osm: dict, bbox: list[float], meters_per_block: float = 1.0,
                photos: list[dict] | None = None, elevation: dict | None = None,
                annotations: list[dict] | None = None) -> dict:
    """Build a JSON-serializable semantic scene from Overpass `out geom` data."""
    if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
        raise ValueError("bbox must contain finite west,south,east,north coordinates")
    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
        raise ValueError("Choose a nonempty bounding box between latitudes -85 and 85")
    scale = float(meters_per_block)
    if not math.isfinite(scale) or scale < .5 or scale > 8:
        raise ValueError("meters_per_block must be between 0.5 and 8")
    width, depth, project = _projection(bbox, scale)
    if width > 1024 or depth > 1024:
        raise ValueError("Selected area exceeds 1024 blocks per side; increase meters per block or reduce the area")
    limitations = ["Photo hints recolor nearby procedural assets; they do not reconstruct exact photo geometry.",
                   "Unmapped buildings and unseen features cannot be recovered; missing heights use explicit estimates.",
                   "Roads, terrain and roofs are approximated on the Minecraft block grid."]
    terrain = {"kind": "flat", "heights": [[4]], "source": "flat_fallback"}
    if elevation and elevation.get("heights"):
        values = np.asarray(elevation["heights"], dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Elevation heights must be a finite rectangular grid")
        values = values - values.min()
        if values.max() / scale > 56:
            limitations.append("Terrain relief is capped at 56 blocks to fit the bounded export height.")
        terrain = {"kind": "elevation", "heights": np.round(np.minimum(values / scale, 56) + 4, 2).tolist(),
                   "source": elevation.get("source", "provided_elevation")}
    else:
        limitations.append("No elevation data supplied: terrain is flat, not a measured real-world surface.")
    elements = osm.get("elements", [])
    nodes = {el["id"]: el for el in elements if el.get("type") == "node"}
    ways = {el["id"]: el for el in elements if el.get("type") == "way"}

    def coordinates(el):
        geom = el.get("geometry", [])
        if geom:
            return [project(p["lon"], p["lat"]) for p in geom if "lon" in p and "lat" in p]
        return [project(nodes[n]["lon"], nodes[n]["lat"]) for n in el.get("nodes", []) if n in nodes]

    features, relation_members = [], set()
    for el in elements:
        if el.get("type") != "relation" or el.get("tags", {}).get("type") != "multipolygon":
            continue
        tags = el.get("tags", {})
        if not any(k in tags for k in ("building", "natural", "landuse", "leisure", "water", "amenity")):
            continue
        outer, inner = [], []
        for member in el.get("members", []):
            if member.get("type") != "way":
                continue
            points = coordinates(member) or coordinates(ways.get(member.get("ref"), {}))
            if not points:
                continue
            (inner if member.get("role") == "inner" else outer).append(points)
            relation_members.add(member.get("ref"))
        holes = _join_rings(inner)
        for i, ring in enumerate(_join_rings(outer)):
            features.append((f"relation/{el['id']}/{i}", tags, ring,
                             [hole for hole in holes if _point_inside(hole[0], ring)]))
    for el in elements:
        tags = el.get("tags", {})
        if el.get("type") == "way" and el.get("id") not in relation_members:
            points = coordinates(el)
            if len(points) >= 2:
                if len(points) > 2 and points[0] == points[-1]:
                    points = points[:-1]
                features.append((f"way/{el['id']}", tags, points, []))
        elif el.get("type") == "node" and tags.get("natural") == "tree":
            features.append((f"node/{el['id']}", tags, [project(el["lon"], el["lat"])], []))

    local_photos = []
    for index, photo in enumerate(photos or []):
        lat, lon = photo.get("lat", photo.get("exif_lat")), photo.get("lon", photo.get("exif_lon"))
        if lat is None or lon is None:
            continue
        if not math.isfinite(float(lat)) or not math.isfinite(float(lon)):
            continue
        local_photos.append({**photo, "id": photo.get("id", str(index)), "position": project(lon, lat)})

    bounded = []
    for feature_id, tags, points, holes in features:
        if tags.get("highway"):
            chain, part = [], 0
            for a, b in zip(points, points[1:]):
                clipped = _clip_road(a, b, width, depth)
                if clipped and (not chain or chain[-1] == clipped[0]):
                    chain = chain + clipped[1:] if chain else clipped
                else:
                    if len(chain) >= 2:
                        bounded.append((f"{feature_id}/part/{part}", tags, chain, []))
                        part += 1
                    chain = clipped or []
            if len(chain) >= 2:
                bounded.append((f"{feature_id}/part/{part}", tags, chain, []))
        else:
            bounded.append((feature_id, tags, points, holes))

    objects = []
    for feature_id, tags, points, holes in bounded:
        kind = None
        if tags.get("building") and tags.get("building") != "no":
            kind = "building"
        elif tags.get("highway") and tags.get("highway") not in {"proposed", "construction"}:
            kind = "road"
        elif tags.get("natural") == "tree":
            kind = "tree"
        elif tags.get("natural") == "water" or tags.get("water") or tags.get("landuse") == "reservoir":
            kind = "water"
        elif tags.get("amenity") == "parking":
            kind = "parking"
        elif tags.get("leisure") in {"park", "garden", "playground"} or tags.get("landuse") in {"forest", "grass", "meadow", "recreation_ground"} or tags.get("natural") in {"wood", "grassland"}:
            kind = "park"
        if kind is None:
            continue
        if kind == "parking" and (tags.get("parking") in {"underground", "multi-storey", "rooftop"} or _number(tags.get("layer"), 0) < 0 or _number(tags.get("level"), 0) < 0):
            continue  # A buried garage is not a surface car park over a civic plaza.
        if kind not in {"tree", "road"}:
            points = _clip_polygon(points, width, depth)
            holes = [_clip_polygon(hole, width, depth) for hole in holes]
            holes = [h for h in holes if len(h) >= 3]
            if len(points) < 3:
                continue
        if kind == "tree" and not (0 <= points[0][0] < width and 0 <= points[0][1] < depth):
            continue
        if kind == "road" and (max(p[0] for p in points) < 0 or min(p[0] for p in points) >= width or
                               max(p[1] for p in points) < 0 or min(p[1] for p in points) >= depth):
            continue
        center = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]
        base_y = round(_terrain_height(terrain, *center, width, depth)) + 1
        materials = dict(DEFAULT_MATERIALS)
        obj = {"id": feature_id, "kind": kind, "name": tags.get("name", kind.title()),
               "geometry": {"type": "point" if kind == "tree" else "polyline" if kind == "road" else "polygon",
                            "coordinates": points[0] if kind == "tree" else points,
                            "holes": holes},
               "center": center, "base_y": base_y, "height": 0,
               "materials": materials, "source": "osm", "confidence": 0.65, "tags": dict(tags)}
        if kind == "building":
            building_type = tags.get("building", "yes")
            levels = max(1, min(30, int(_number(tags.get("building:levels"), 2))))
            height_m = _number(tags.get("height"), levels * 3.2)
            if height_m / scale > 96 and "Building heights are capped at 96 blocks." not in limitations:
                limitations.append("Building heights are capped at 96 blocks.")
            height_blocks = max(1, min(96, round(height_m / scale)))
            pitched = tags.get("roof:shape", "") in {"gabled", "hipped", "pyramidal", "half-hipped"} or (
                "roof:shape" not in tags and building_type not in {"apartments", "commercial", "industrial", "retail", "office", "warehouse", "civic", "public", "government", "school", "hospital"} and levels <= 3)
            roof_height = max(1, min(6, round(2.5 / scale))) if pitched else 1
            obj.update({"levels": levels, "height": height_blocks,
                        "roof_shape": "gabled" if pitched else "flat", "roof_height": roof_height,
                        "height_source": "osm" if "height" in tags or "building:levels" in tags else "estimated",
                        "building_type": building_type})
            known = str(tags.get("building:material", "")).lower()
            if "wood" in known:
                materials["wall"] = "minecraft:oak_planks"
            elif "sandstone" in known:
                materials["wall"] = "minecraft:sandstone"
            elif "stone" in known:
                materials["wall"] = "minecraft:stone_bricks"
            elif "concrete" in known or building_type in {"apartments", "office", "commercial"}:
                materials["wall"] = "minecraft:white_concrete"
            nearest = []
            for photo in local_photos:
                px, pz = photo["position"]
                distance = math.hypot(center[0] - px, center[1] - pz) * scale
                if distance > max(0, min(500, float(photo.get("radius_m", 80)))):
                    continue
                heading = photo.get("heading")
                if heading is not None and photo.get("projection") != "panorama":
                    bearing = math.degrees(math.atan2(center[0] - px, -(center[1] - pz))) % 360
                    delta = abs((bearing - float(heading) + 180) % 360 - 180)
                    if delta > float(photo.get("fov") or 70) / 2:
                        continue
                nearest.append((distance, photo))
            if nearest:
                distance, photo = min(nearest, key=lambda item: item[0])
                style = photo.get("style", "auto")
                mat = STYLE_MATERIALS.get(style, photo.get("material", materials["wall"]))
                if mat in STYLE_MATERIALS.values():
                    materials["wall"] = mat
                obj.update({"source": "osm+photo", "photo_id": photo["id"],
                            "appearance_confidence": 0.35, "photo_distance_m": round(distance, 1),
                            "appearance_method": "nearby_photo_palette", "photo_color": photo.get("dominant_color")})
        elif kind == "road":
            highway = tags.get("highway")
            default_width = {"motorway": 16, "trunk": 13, "primary": 11, "secondary": 9,
                             "tertiary": 8, "residential": 6, "service": 4, "footway": 2,
                             "path": 2, "cycleway": 2, "pedestrian": 4}.get(highway, 6)
            obj.update({"width": max(1, min(30 / scale, _number(tags.get("width"), default_width) / scale)),
                        "road_type": highway, "sidewalk_width": max(0, round(1.5 / scale)),
                        "markings": highway not in {"footway", "path", "cycleway", "steps", "pedestrian"}})
            if not obj["markings"]:
                materials["road"] = "minecraft:smooth_stone"
        elif kind == "tree":
            obj.update({"height": max(4, min(18, round(_number(tags.get("height"), 8) / scale))),
                        "canopy_radius": max(1.5, 3 / scale)})
        elif kind == "park":
            obj["vegetation"] = "forest" if tags.get("landuse") == "forest" or tags.get("natural") == "wood" else "park"
        elif kind == "parking":
            materials["surface"] = "minecraft:gravel" if tags.get("surface") in {"gravel", "fine_gravel", "unpaved"} else "minecraft:gray_concrete"
        obj["color"] = material_color(materials["wall" if kind == "building" else "road" if kind == "road" else "water" if kind == "water" else "leaves" if kind == "tree" else "surface" if kind == "parking" else "grass"])
        objects.append(obj)

    # Deterministic hand-built trees inside mapped green areas. Their locations
    # are explicit approximations, not claimed observations from the images.
    for park in [o for o in objects if o["kind"] == "park"]:
        poly = park["geometry"]["coordinates"]
        spacing = max(4, (7 if park["vegetation"] == "forest" else 12) / scale)
        minx, maxx = max(1, min(p[0] for p in poly)), min(width - 2, max(p[0] for p in poly))
        minz, maxz = max(1, min(p[1] for p in poly)), min(depth - 2, max(p[1] for p in poly))
        count = 0
        for z in np.arange(minz + spacing / 2, maxz, spacing):
            for x in np.arange(minx + spacing / 2, maxx, spacing):
                jitter = math.sin(float(x) * 1.19 + float(z) * 2.73)
                point = [round(float(x + jitter * spacing * .16), 2), round(float(z), 2)]
                if not _point_inside(point, poly) or any(_point_inside(point, h) for h in park["geometry"]["holes"]):
                    continue
                if any(o["kind"] in {"building", "water"} and _point_inside(point, o["geometry"]["coordinates"])
                       for o in objects):
                    continue
                count += 1
                objects.append({"id": f"{park['id']}/tree/{count}", "kind": "tree", "name": "Estimated park tree",
                                "geometry": {"type": "point", "coordinates": point, "holes": []}, "center": point,
                                "base_y": round(_terrain_height(terrain, *point, width, depth)) + 1,
                                "height": max(4, round((7 + abs(jitter) * 3) / scale)),
                                "canopy_radius": max(1.5, 2.7 / scale),
                                "materials": dict(DEFAULT_MATERIALS), "color": "#417c33",
                                "source": "procedural_vegetation", "confidence": 0.25})
                if count >= 2000:
                    break
            if count >= 2000:
                break

    max_height = max([int(max(max(row) for row in terrain["heights"])) + 12] +
                     [int(o["base_y"] + o["height"] + o.get("roof_height", 0) + 3) for o in objects])
    if max_height > 128:
        limitations.append("Objects taller than the 128-block local build height are clipped.")
    counts = Counter(o["kind"] for o in objects)
    scene = {"version": 1, "coordinate_system": "x=east,y=up,z=south; units=blocks", "bbox": list(map(float, bbox)),
            "meters_per_block": scale, "bounds": {"width": width, "depth": depth, "height": min(128, max(24, max_height))},
            "ground_y": 4, "terrain": terrain, "materials": DEFAULT_MATERIALS.copy(), "objects": objects,
            "stats": {"objects": len(objects), **dict(counts), "photo_dressed_buildings": sum(o["source"] == "osm+photo" for o in objects),
                      "photos_used": len({o.get("photo_id") for o in objects if o.get("photo_id") is not None})},
            "limitations": limitations}
    # Import lazily: enrichment reuses projection/terrain helpers from this file.
    from .enrichment import enrich_scene
    enrich_scene(scene, osm, local_photos, annotations or [])
    return scene


def _polygon_mask(obj, width, depth):
    im = Image.new("1", (width, depth))
    draw = ImageDraw.Draw(im)
    draw.polygon([tuple(p) for p in obj["geometry"]["coordinates"]], fill=1)
    for hole in obj["geometry"].get("holes", []):
        if len(hole) >= 3:
            draw.polygon([tuple(p) for p in hole], fill=0)
    return np.asarray(im, dtype=bool)


def _erode(mask):
    result = mask.copy()
    result[1:] &= mask[:-1]
    result[:-1] &= mask[1:]
    result[:, 1:] &= mask[:, :-1]
    result[:, :-1] &= mask[:, 1:]
    result[0] = result[-1] = False
    result[:, 0] = result[:, -1] = False
    return result


def _terrain_grid(scene):
    width, depth = scene["bounds"]["width"], scene["bounds"]["depth"]
    grid = np.asarray(scene["terrain"]["heights"], dtype=np.float32)
    if grid.shape == (depth, width):
        return np.rint(grid).astype(np.int16)
    return np.rint(np.asarray(Image.fromarray(grid).resize((width, depth), Image.Resampling.BILINEAR))).astype(np.int16)


def scene_to_voxels(scene: dict) -> tuple[np.ndarray, list[str]]:
    """Voxelize semantic hand-built assets into a bounded YZX palette array."""
    width, depth, height = (int(scene["bounds"][k]) for k in ("width", "depth", "height"))
    if min(width, depth, height) <= 0 or width > 1024 or depth > 1024 or height > 128:
        raise ValueError("Invalid or oversized scene bounds")
    blocks = np.zeros((height, depth, width), dtype=np.uint16)
    palette = ["minecraft:air"]
    palette_ids = {palette[0]: 0}

    def block(name):
        if name not in palette_ids:
            palette_ids[name] = len(palette)
            palette.append(name)
        return palette_ids[name]

    ground = _terrain_grid(scene)
    grass, dirt, stone = block("minecraft:grass_block"), block("minecraft:dirt"), block("minecraft:stone")
    blocks[0] = block("minecraft:bedrock")
    for y in range(1, min(height, int(ground.max()) + 1)):
        blocks[y][ground > y + 2] = stone
        blocks[y][(ground >= y) & (ground <= y + 2)] = dirt
        blocks[y][ground == y] = grass
    zz, xx = np.indices((depth, width))
    occupied = np.zeros((depth, width), dtype=bool)

    for obj in scene["objects"]:
        if obj["kind"] == "parking":
            mask = _polygon_mask(obj, width, depth)
            z, x = np.where(mask)
            blocks[ground[z, x], z, x] = block(obj["materials"].get("surface", "minecraft:gray_concrete"))

    for obj in scene["objects"]:
        if obj["kind"] == "water":
            mask = _polygon_mask(obj, width, depth)
            z, x = np.where(mask)
            y = ground[z, x]
            blocks[y, z, x] = block("minecraft:water[level=0]")
            blocks[np.maximum(y - 1, 1), z, x] = block("minecraft:sand")
            occupied |= mask

    for obj in scene["objects"]:
        if obj["kind"] != "road":
            continue
        points = [tuple(p) for p in obj["geometry"]["coordinates"]]
        road_width = max(1, round(obj["width"]))
        im = Image.new("1", (width, depth))
        draw = ImageDraw.Draw(im)
        draw.line(points, fill=1, width=road_width + 2 * obj["sidewalk_width"], joint="curve")
        outer = np.asarray(im, dtype=bool)
        z, x = np.where(outer)
        blocks[ground[z, x], z, x] = block(obj["materials"]["sidewalk"])
        im = Image.new("1", (width, depth))
        draw = ImageDraw.Draw(im)
        draw.line(points, fill=1, width=road_width, joint="curve")
        mask = np.asarray(im, dtype=bool)
        z, x = np.where(mask)
        blocks[ground[z, x], z, x] = block(obj["materials"]["road"])
        # Sidewalks and walking paths are valid tree locations. The previous
        # outer mask silently discarded every sidewalk-adjacent tree.
        if obj.get("road_type") not in {"footway", "path", "cycleway", "steps", "pedestrian", "bridleway"}:
            occupied |= mask
        if obj["markings"] and road_width >= 4:
            # A dashed line follows the polyline arc, including rotated roads.
            distance_offset = 0.0
            for a, b in zip(points, points[1:]):
                length = math.dist(a, b)
                if length == 0:
                    continue
                for distance in np.arange(0, length, .6):
                    if int(distance + distance_offset) % 10 >= 4:
                        continue
                    t = distance / length
                    x, z = round(a[0] + t * (b[0] - a[0])), round(a[1] + t * (b[1] - a[1]))
                    if 0 <= x < width and 0 <= z < depth:
                        blocks[ground[z, x], z, x] = block("minecraft:white_concrete")
                distance_offset += length

    for obj in scene["objects"]:
        if obj["kind"] != "building" or "parts" in obj:
            continue
        mask = _polygon_mask(obj, width, depth)
        if not mask.any():
            continue
        inside = _erode(mask)
        walls = mask & ~inside
        occupied |= mask
        floor_y = min(height - 3, max(int(obj["base_y"]) - 1, int(ground[mask].max())))
        top_y = min(height - 2, floor_y + int(obj["height"]))
        wall, trim = block(obj["materials"]["wall"]), block(obj["materials"]["trim"])
        floor = block(obj["materials"]["floor"])
        # Solid foundation, hollow rooms, and floors every level.
        for y in range(int(ground[mask].min()), floor_y + 1):
            blocks[y][mask] = block("minecraft:stone_bricks")
        floor_step = max(3, round(obj["height"] / max(1, obj["levels"])))
        for y in range(floor_y + 1, top_y + 1):
            blocks[y][mask] = 0
            blocks[y][walls] = trim if (y - floor_y) % floor_step == 0 else wall
            if (y - floor_y) % floor_step == 0:
                blocks[y][inside] = floor
            elif (y - floor_y) % floor_step in {1, 2} and y > floor_y + 1:
                windows = walls & (((xx + zz) % 5 == 2) | ((xx + zz) % 5 == 3))
                blocks[y][windows] = block(obj["materials"]["window"])
        blocks[floor_y][mask] = floor
        # Put a real two-block oak door on the longest exterior edge.
        poly = obj["geometry"]["coordinates"]
        a, b = max(zip(poly, poly[1:] + poly[:1]), key=lambda pair: math.dist(*pair))
        door_x, door_z = int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2)
        if 0 <= door_x < width and 0 <= door_z < depth and floor_y + 2 < height:
            facing = "north" if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else "west"
            blocks[floor_y + 1, door_z, door_x] = block(f"minecraft:oak_door[facing={facing},half=lower,hinge=left,open=false,powered=false]")
            blocks[floor_y + 2, door_z, door_x] = block(f"minecraft:oak_door[facing={facing},half=upper,hinge=left,open=false,powered=false]")
        roof = block(obj["materials"]["roof"])
        if obj["roof_shape"] == "flat":
            blocks[min(height - 1, top_y + 1)][mask] = roof
        else:
            z_indices, x_indices = np.where(mask)
            spanx, spanz = max(1, int(np.ptp(x_indices))), max(1, int(np.ptp(z_indices)))
            axis = xx if spanx < spanz else zz
            minimum = x_indices.min() if spanx < spanz else z_indices.min()
            span = spanx if spanx < spanz else spanz
            rise = np.floor((1 - np.abs((axis - minimum) / span * 2 - 1)) * obj["roof_height"]).astype(int)
            for layer in range(obj["roof_height"] + 1):
                roofmask = mask & (rise >= layer)
                y = top_y + 1 + layer
                if y < height:
                    blocks[y][roofmask] = roof

    for obj in scene["objects"]:
        if obj["kind"] != "tree" or "parts" in obj:
            continue
        cx, cz = map(math.floor, obj["geometry"]["coordinates"])
        if not (0 <= cx < width and 0 <= cz < depth) or occupied[cz, cx]:
            continue
        floor_y = int(ground[cz, cx])
        tree_height = int(obj["height"])
        radius = float(obj.get("canopy_radius", 2.5))
        crown_y = floor_y + tree_height - max(1, round(radius * .6))
        minx, maxx = max(0, math.floor(cx - radius)), min(width, math.ceil(cx + radius + 1))
        minz, maxz = max(0, math.floor(cz - radius)), min(depth, math.ceil(cz + radius + 1))
        for y in range(max(floor_y + 2, math.floor(crown_y - radius)), min(height, math.ceil(crown_y + radius + 1))):
            local_z, local_x = np.ogrid[minz:maxz, minx:maxx]
            canopy = ((local_x - cx) / radius) ** 2 + ((local_z - cz) / radius) ** 2 + ((y - crown_y) / (radius * 1.1)) ** 2 <= 1.05
            available = blocks[y, minz:maxz, minx:maxx] == 0
            blocks[y, minz:maxz, minx:maxx][canopy & available] = block(obj["materials"]["leaves"])
        for y in range(floor_y + 1, min(height, crown_y + 1)):
            blocks[y, cz, cx] = block(obj["materials"]["trunk"])
    # Enriched placement has already resolved collisions and recorded drops.
    # Apply the same world-coordinate parts used by OBJ/browser rendering.
    # People remain real entities rather than duplicate block mannequins.
    for obj in scene["objects"]:
        if obj["kind"] != "person" and "parts" in obj:
            voxelize_parts(blocks, obj, block)
    return blocks, palette


def _triangulate(poly):
    """Ear-clip a simple polygon for interoperable OBJ triangle faces."""
    if len(poly) < 3:
        return []
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))
    ids = list(range(len(poly))) if area > 0 else list(reversed(range(len(poly))))
    triangles = []
    budget = len(ids) ** 2
    while len(ids) > 3 and budget > 0:
        found = False
        for k in range(len(ids)):
            a, b, c = ids[k - 1], ids[k], ids[(k + 1) % len(ids)]
            pa, pb, pc = poly[a], poly[b], poly[c]
            cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
            if cross <= 1e-8:
                continue
            if any(_point_inside(poly[j], [pa, pb, pc]) for j in ids if j not in {a, b, c}):
                continue
            triangles.append((a, b, c))
            ids.pop(k)
            found = True
            break
        budget -= 1
        if not found:
            break
    if len(ids) == 3:
        triangles.append(tuple(ids))
    return triangles


def export_obj(scene: dict, path: Path) -> Path:
    """Write a real semantic mesh and MTL, before any block rasterization.

    Building shells/roofs, terrain, streets and procedural tree meshes are
    included. Minecraft windows/doors are discrete asset details added later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = path.with_suffix(".mtl")
    vertices_count, materials = 0, {}
    width, depth = scene["bounds"]["width"], scene["bounds"]["depth"]

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# RealityForge semantic 3D model; units=blocks; x=east,y=up,z=south\nmtllib {mtl_path.name}\n")

        def mesh(name, vertices, faces, material):
            nonlocal vertices_count
            if not faces:
                return
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", material)
            materials[safe_name] = material
            stream.write(f"o {re.sub(r'[^a-zA-Z0-9_]', '_', name)}\nusemtl {safe_name}\n")
            for x, y, z in vertices:
                stream.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")
            for face in faces:
                stream.write("f " + " ".join(str(vertices_count + i + 1) for i in face) + "\n")
            vertices_count += len(vertices)

        def cuboid(name, x, y, z, sx, sy, sz, material):
            verts = [(x + dx * sx, y + dy * sy, z + dz * sz) for dy in [0, 1] for dz in [0, 1] for dx in [0, 1]]
            mesh(name, verts, [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)], material)

        # The same supplied coarse elevation grid produces a continuous mesh.
        grid = scene["terrain"]["heights"]
        if len(grid) == 1 and len(grid[0]) == 1:
            mesh("terrain", [(0, grid[0][0] + 1, 0), (width, grid[0][0] + 1, 0),
                              (width, grid[0][0] + 1, depth), (0, grid[0][0] + 1, depth)],
                 [(0, 3, 2), (0, 2, 1)], "minecraft:grass_block")
        else:
            rows, cols = len(grid), len(grid[0])
            verts = [(x * width / max(1, cols - 1), grid[z][x] + 1, z * depth / max(1, rows - 1))
                     for z in range(rows) for x in range(cols)]
            faces = []
            for z in range(rows - 1):
                for x in range(cols - 1):
                    i = z * cols + x
                    faces.extend([(i, i + cols, i + 1), (i + 1, i + cols, i + cols + 1)])
            mesh("terrain", verts, faces, "minecraft:grass_block")

        for obj in scene["objects"]:
            kind, geom = obj["kind"], obj["geometry"]
            name, base = obj["id"], obj["base_y"]
            if "parts" in obj:
                for index, part in enumerate(obj["parts"]):
                    vertices, faces = part_mesh(part)
                    mesh(name + f"_part_{index}", vertices, faces, part["material"])
            elif kind == "building":
                poly = geom["coordinates"]
                n = len(poly)
                top = base + obj["height"]
                verts = [(x, base, z) for x, z in poly] + [(x, top, z) for x, z in poly]
                faces = [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
                mesh(name, verts, faces, obj["materials"]["wall"])
                for index, hole in enumerate(geom.get("holes", [])):
                    nh = len(hole)
                    mesh(name + f"_courtyard_{index}", [(x, base, z) for x, z in hole] + [(x, top, z) for x, z in hole],
                         [(i + nh, (i + 1) % nh + nh, (i + 1) % nh, i) for i in range(nh)], obj["materials"]["wall"])
                # Courtyard buildings get polygon strip roofs, excluding holes.
                if geom.get("holes"):
                    for i, (a, b) in enumerate(zip(poly, poly[1:] + poly[:1])):
                        mesh(name + f"_roof_rim_{i}", [(a[0], top, a[1]), (b[0], top, b[1]),
                             (b[0], top + .2, b[1]), (a[0], top + .2, a[1])], [(0, 1, 2, 3)], obj["materials"]["roof"])
                else:
                    minx, maxx = min(p[0] for p in poly), max(p[0] for p in poly)
                    minz, maxz = min(p[1] for p in poly), max(p[1] for p in poly)
                    ridge_axis = 0 if maxx - minx < maxz - minz else 1
                    low, high = (minx, maxx) if ridge_axis == 0 else (minz, maxz)
                    # Split at the ridge so the roof contains its raised center.
                    halves = [poly] if obj["roof_shape"] == "flat" else [
                        _clip_half_plane(poly, ridge_axis, (low + high) / 2, True),
                        _clip_half_plane(poly, ridge_axis, (low + high) / 2, False)]
                    for index, half in enumerate(halves):
                        roof_vertices = []
                        for p in half:
                            rise = 0 if obj["roof_shape"] == "flat" else (1 - abs((p[ridge_axis] - low) / max(.01, high - low) * 2 - 1)) * obj["roof_height"]
                            roof_vertices.append((p[0], top + rise, p[1]))
                        mesh(name + f"_roof_{index}", roof_vertices, _triangulate(half), obj["materials"]["roof"])
            elif kind == "road":
                points = geom["coordinates"]
                for i, (a, b) in enumerate(zip(points, points[1:])):
                    length = math.dist(a, b)
                    if length < .01:
                        continue
                    dx, dz = -(b[1] - a[1]) / length, (b[0] - a[0]) / length
                    for label, halfwidth, material, lift in [("sidewalk", obj["width"] / 2 + obj["sidewalk_width"], obj["materials"]["sidewalk"], .05),
                                                            ("surface", obj["width"] / 2, obj["materials"]["road"], .10)]:
                        verts = []
                        for p, sign in [(a, 1), (b, 1), (b, -1), (a, -1)]:
                            x, z = p[0] + sign * dx * halfwidth, p[1] + sign * dz * halfwidth
                            y = _terrain_height(scene["terrain"], x, z, width, depth) + 1 + lift
                            verts.append((x, y, z))
                        mesh(name + f"_{label}_{i}", verts, [(0, 1, 2), (0, 2, 3)], material)
            elif kind == "tree":
                x, z = geom["coordinates"]
                h, radius = obj["height"], obj.get("canopy_radius", 2.5)
                cuboid(name + "_trunk", x - .4, base, z - .4, .8, h * .7, .8, obj["materials"]["trunk"])
                cuboid(name + "_crown", x - radius, base + h * .45, z - radius, radius * 2, h * .4, radius * 2, obj["materials"]["leaves"])
                cuboid(name + "_crown_top", x - radius * .7, base + h * .8, z - radius * .7, radius * 1.4, h * .2, radius * 1.4, obj["materials"]["leaves"])
            elif kind in {"water", "park", "parking"}:
                poly = geom["coordinates"]
                vertices = [(x, _terrain_height(scene["terrain"], x, z, width, depth) + 1.03, z) for x, z in poly]
                # Avoid filling an island/courtyard when triangle centers fall in a hole.
                faces = [t for t in _triangulate(poly) if not any(_point_inside(
                    [sum(poly[i][0] for i in t) / 3, sum(poly[i][1] for i in t) / 3], h) for h in geom.get("holes", []))]
                key = "water" if kind == "water" else "surface" if kind == "parking" else "grass"
                mesh(name, vertices, faces, obj["materials"].get(key, "minecraft:gray_concrete"))

    with mtl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for name, material in materials.items():
            color = material_color(material)
            values = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            stream.write(f"newmtl {name}\nKd {' '.join(f'{v:.4f}' for v in values)}\nKa 0.2 0.2 0.2\nd 1\n\n")
    return path


def _clip_half_plane(poly, axis, bound, keep_less):
    output = []
    if not poly:
        return output
    previous = poly[-1]
    previous_inside = previous[axis] <= bound if keep_less else previous[axis] >= bound
    for current in poly:
        inside = current[axis] <= bound if keep_less else current[axis] >= bound
        if inside != previous_inside:
            t = (bound - previous[axis]) / (current[axis] - previous[axis])
            output.append([previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1])])
        if inside:
            output.append(current)
        previous, previous_inside = current, inside
    return output
