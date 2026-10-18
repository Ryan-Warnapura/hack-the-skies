"""Seeded semantic street assets shared by the mesh, preview and voxel exports.

Population is an explicit approximation, never an observation about real people
or vehicles. Pedestrians are exported separately as Java armour-stand entities.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw


WALKING_ROADS = {"footway", "path", "pedestrian", "steps", "living_street"}
NO_PARKING_ROADS = {"motorway", "motorway_link", "trunk", "trunk_link", "cycleway"}
VEHICLE_DIMENSIONS = {
    "sedan": (1.9, 1.7, 4.6), "hatchback": (1.8, 1.7, 3.8),
    "suv": (2.0, 2.0, 4.8), "pickup": (2.0, 1.9, 5.3),
    "van": (2.0, 2.5, 5.2), "taxi": (1.9, 1.8, 4.6),
    "police": (1.9, 1.8, 4.8), "delivery_truck": (2.4, 3.3, 7.2),
    "bus": (2.5, 3.3, 11.0), "streetcar": (2.5, 3.4, 14.0),
}
BLOCK_COLORS = {
    "white": 0xD6DADB, "light_gray": 0x999D97, "gray": 0x464A4C,
    "black": 0x202124, "red": 0xA93432, "orange": 0xD57832,
    "yellow": 0xE7BB34, "lime": 0x84AC35, "green": 0x4C6938,
    "cyan": 0x218389, "light_blue": 0x54A3C7, "blue": 0x3B4E9C,
    "purple": 0x793EA7, "magenta": 0xAF51A3, "pink": 0xD889A4,
    "brown": 0x775436,
}
CAR_COLORS = ("white", "light_gray", "gray", "black", "red", "blue", "cyan", "green", "orange")


def _rng(seed):
    return random.Random(int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big"))


def _rgb(value, fallback):
    try:
        return int(str(value).lstrip("#"), 16) if isinstance(value, str) else int(value)
    except (ValueError, TypeError):
        return fallback


def _nearest_color(value):
    rgb = _rgb(value, BLOCK_COLORS["blue"])
    return min(BLOCK_COLORS, key=lambda k: sum(((rgb >> shift & 255) - (BLOCK_COLORS[k] >> shift & 255)) ** 2
                                              for shift in (16, 8, 0)))


def _ground(scene, point):
    # Lazy import keeps scene.py able to import populate_scene.
    from .scene import _terrain_height
    return round(_terrain_height(scene["terrain"], *point, scene["bounds"]["width"], scene["bounds"]["depth"])) + 1


def _base(scene, kind, point, object_id, source, confidence, heading):
    point = [round(float(v), 3) for v in point]
    return {"id": object_id or f"{source}/{kind}/{point[0]}/{point[1]}", "kind": kind,
            "name": f"Estimated {kind}" if source != "user_annotation" else f"Annotated {kind}",
            "geometry": {"type": "point", "coordinates": point, "holes": []}, "center": point,
            "base_y": _ground(scene, point), "heading": round(float(heading) % 360, 3),
            "source": source, "confidence": confidence, "height": 0,
            "materials": dict(scene.get("materials", {})), "parts": []}


def _part(obj, local_center, size, material, shape="box", yaw=0):
    """Local front is negative Z; heading is compass degrees clockwise north."""
    radians = math.radians(-obj.get("heading", 0))
    x, y, z = local_center
    cosine, sine = math.cos(radians), math.sin(radians)
    obj["parts"].append({"shape": shape,
                         "center": [round(obj["center"][0] + x * cosine + z * sine, 3),
                                    round(obj["base_y"] + y, 3),
                                    round(obj["center"][1] - x * sine + z * cosine, 3)],
                         "size": [round(float(v), 3) for v in size],
                         "material": material, "yaw": round(yaw - obj.get("heading", 0), 3)})


def make_vehicle(scene, point, vehicle_type="sedan", color=None, heading=0, object_id=None,
                 source="procedural_population", confidence=.25, placement="parked"):
    """Build one complete vehicle; geometry is in block units at the scene scale."""
    vehicle_type = vehicle_type if vehicle_type in VEHICLE_DIMENSIONS else "sedan"
    obj = _base(scene, "vehicle", point, object_id, source, confidence, heading)
    rng = _rng(obj["id"])
    scale = scene.get("meters_per_block", 1)
    physical = VEHICLE_DIMENSIONS[vehicle_type]
    # Design in metres first; scale every offset and part together at the end.
    # Minimum block dimensions would turn a car into a 25-metre vehicle at 8m.
    width, height, length = physical
    color_name = _nearest_color(color) if color else rng.choice(CAR_COLORS)
    if not color and vehicle_type == "taxi":
        color_name = "yellow"
    elif not color and vehicle_type == "police":
        color_name = "white"
    elif not color and vehicle_type == "streetcar":
        color_name = "red"
    body = f"minecraft:{color_name}_concrete"
    obj.update({"vehicle_type": vehicle_type, "placement": placement, "height": height,
                "width": width, "length": length, "color": f"#{BLOCK_COLORS[color_name]:06x}",
                "requested_color": color, "appearance_method": "hand_built_vehicle_asset"})
    obj["materials"].update({"body": body, "window": "minecraft:light_blue_stained_glass",
                             "wheel": "minecraft:black_concrete"})
    wheel = .85
    floor = max(.55, wheel * .7)
    body_top = max(1.1, height * .58)
    _part(obj, (0, (floor + body_top) / 2, 0), (width, body_top - floor, length), body)
    # Dark chassis and pale bumpers make the front/rear readable in Minecraft.
    _part(obj, (0, floor * .65, 0), (width * .8, .35, length * .9), "minecraft:polished_blackstone")
    for front in (-1, 1):
        _part(obj, (0, floor + .08, front * (length / 2 - .13)),
              (width * .92, .3, .3), "minecraft:polished_andesite")
    axle_z = length * (.34 if vehicle_type not in {"bus", "streetcar"} else .37)
    for x in (-width * .45, width * .45):
        for z in (-axle_z, axle_z):
            _part(obj, (x, wheel / 2, z), (wheel * .5, wheel, wheel), "minecraft:black_concrete")
    if vehicle_type in {"bus", "streetcar", "van", "delivery_truck"}:
        roof_start = -length * .36
        cabin_length = length * .88
        _part(obj, (0, (body_top + height) / 2, length * .025),
              (width * .94, height - body_top, cabin_length), body)
        window_y = body_top + (height - body_top) * .43
        window_h = max(.5, (height - body_top) * .65)
        _part(obj, (0, window_y, roof_start - .045 * length),
              (width * .82, window_h, .18), "minecraft:light_blue_stained_glass")
        if vehicle_type != "delivery_truck":
            bays = max(2, round(length / 1.5))
            for x in (-width * .478, width * .478):
                for index in range(bays):
                    z = -length * .29 + index / max(1, bays - 1) * length * .58
                    _part(obj, (x, window_y, z), (.17, window_h, length * .44 / bays),
                          "minecraft:light_blue_stained_glass")
        else:
            # A contrasting cargo box and double rear doors distinguish a truck.
            _part(obj, (0, (body_top + height) / 2, length * .14),
                  (width * .96, height - body_top, length * .6), "minecraft:white_concrete")
            _part(obj, (0, (floor + height) / 2, length * .485),
                  (width * .78, height - floor - .1, .14), "minecraft:iron_block")
        _part(obj, (0, height - .1, length * .025), (width, .23, cabin_length + .08), body)
    else:
        cabin_len = length * ({"pickup": .35, "hatchback": .62, "suv": .65}.get(vehicle_type, .55))
        cabin_z = -length * .11 if vehicle_type == "pickup" else length * .045
        cabin_width = width * .84
        _part(obj, (0, (body_top + height) / 2, cabin_z),
              (cabin_width, height - body_top, cabin_len), "minecraft:light_blue_stained_glass")
        _part(obj, (0, height - .08, cabin_z), (cabin_width + .05, .25, cabin_len * .9), body)
        for x in (-cabin_width / 2, cabin_width / 2):
            _part(obj, (x, (body_top + height) / 2, cabin_z), (.16, height - body_top, .3), body)
        if vehicle_type == "pickup":
            _part(obj, (0, body_top + .08, length * .28),
                  (width * .7, .18, length * .3), "minecraft:gray_concrete")
        if vehicle_type == "taxi":
            _part(obj, (0, height + .12, cabin_z), (width * .55, .35, .45), "minecraft:yellow_concrete")
        if vehicle_type == "police":
            for x, light in [(-width * .23, "red"), (width * .23, "blue")]:
                _part(obj, (x, height + .13, cabin_z), (width * .42, .35, .45), f"minecraft:{light}_concrete")
            for x in (-width / 2, width / 2):
                _part(obj, (x, body_top * .75, 0), (.15, .35, length * .45), "minecraft:black_concrete")
    for x in (-width * .32, width * .32):
        _part(obj, (x, floor + .39, -length / 2 - .035), (.45, .35, .18), "minecraft:sea_lantern")
        _part(obj, (x, floor + .39, length / 2 + .035), (.4, .35, .18), "minecraft:redstone_block")
    anchor = [obj["center"][0], obj["base_y"], obj["center"][1]]
    for part in obj["parts"]:
        part["center"] = [round(anchor[i] + (part["center"][i] - anchor[i]) / scale, 5) for i in range(3)]
        part["size"] = [round(value / scale, 5) for value in part["size"]]
    obj.update(width=width / scale, height=height / scale, length=length / scale,
               physical_size_m={"width": width, "height": height, "length": length},
               below_voxel_resolution=length / scale < 2)
    return obj


def make_person(scene, point, heading=0, object_id=None, source="procedural_population", confidence=.25,
                appearance=None):
    obj = _base(scene, "person", point, object_id, source, confidence, heading)
    rng = _rng(obj["id"])
    appearance = appearance or {}
    defaults = {"top": rng.choice([0x457EAE, 0xB0483E, 0xE2B84B, 0x4C8754, 0x8F629D, 0xE4E2D5]),
                "bottom": rng.choice([0x343B4B, 0x355B80, 0x695849, 0x3E4C3A]),
                "shoes": rng.choice([0x282829, 0xEEEEDE, 0x684D37])}
    outfit = {key: max(0, min(0xFFFFFF, _rgb(appearance.get(key), val))) for key, val in defaults.items()}
    stride = rng.uniform(-18, 18)
    obj.update({"height": 1.975, "outfit": outfit, "color": f"#{outfit['top']:06x}",
                "entity_type": "minecraft:armor_stand", "appearance_method": "dressed_armour_stand",
                "pose": {"Head": [round(rng.uniform(-8, 8), 2), round(rng.uniform(-22, 22), 2), 0.],
                         "Body": [0., 0., 0.], "LeftArm": [round(-stride - 6, 2), 0., -5.],
                         "RightArm": [round(stride - 6, 2), 0., 5.],
                         "LeftLeg": [round(stride, 2), 0., 0.], "RightLeg": [round(-stride, 2), 0., 0.]}})
    top, bottom, shoes = (f"minecraft:{_nearest_color(outfit[key])}_concrete" for key in ("top", "bottom", "shoes"))
    skin = rng.choice(["sandstone", "smooth_sandstone", "brown_terracotta", "terracotta"])
    outfit["head"] = {"sandstone": 0xD6BC87, "smooth_sandstone": 0xDDCBA3,
                      "brown_terracotta": 0x805435, "terracotta": 0xA87557}[skin]
    _part(obj, (0, 1.73, 0), (.44, .44, .44), f"minecraft:{skin}")
    _part(obj, (0, 1.94, .035), (.46, .13, .46), rng.choice(["minecraft:brown_concrete", "minecraft:black_concrete", "minecraft:yellow_terracotta"]))
    _part(obj, (0, 1.18, 0), (.55, .66, .3), top)
    for sign in (-1, 1):
        swing = math.sin(math.radians(stride)) * sign
        _part(obj, (sign * .37, 1.12, swing * .25), (.18, .67, .21), top)
        _part(obj, (sign * .15, .48, -swing * .25), (.22, .72, .25), bottom)
        _part(obj, (sign * .15, .1, -.055 - swing * .25), (.25, .2, .36), shoes)
    return obj


def decorate_street_detail(scene, obj):
    """Give OSM street furniture explicit shared geometry, keeping provenance."""
    subtype = obj.get("subtype", obj.get("detail_type", "bench"))
    obj.setdefault("center", obj["geometry"]["coordinates"])
    obj.setdefault("base_y", _ground(scene, obj["center"]))
    obj.setdefault("heading", 0)
    obj["parts"] = []
    scale = scene.get("meters_per_block", 1)
    unit = 1 / scale
    iron, oak = "minecraft:iron_block", "minecraft:oak_planks"
    if subtype == "bench":
        _part(obj, (0, .55 * unit, 0), (2 * unit, .25 * unit, .75 * unit),
              "minecraft:oak_slab[type=bottom,waterlogged=false]")
        _part(obj, (0, .95 * unit, .3 * unit), (2 * unit, .7 * unit, .2 * unit), oak)
        for x in (-.7 * unit, .7 * unit):
            _part(obj, (x, .25 * unit, 0), (.25 * unit, .5 * unit, .65 * unit), iron)
    elif subtype in {"streetlight", "lamp"}:
        height = 5.5 * unit
        _part(obj, (0, height / 2, 0), (.35 * unit, height, .35 * unit), iron)
        _part(obj, (.6 * unit, height - .15 * unit, 0), (1.5 * unit, .25 * unit, .25 * unit), iron)
        _part(obj, (1.1 * unit, height - .45 * unit, 0), (.75 * unit, .5 * unit, .75 * unit), "minecraft:sea_lantern")
    elif subtype == "bollard":
        _part(obj, (0, .6 * unit, 0), (.35 * unit, 1.2 * unit, .35 * unit), "minecraft:polished_blackstone")
        _part(obj, (0, .95 * unit, 0), (.38 * unit, .2 * unit, .38 * unit), "minecraft:quartz_block")
    elif subtype == "bus_stop":
        for x in (-1.3 * unit, 1.3 * unit):
            _part(obj, (x, 1.25 * unit, .45 * unit), (.18 * unit, 2.5 * unit, .18 * unit), iron)
        _part(obj, (0, 1.2 * unit, .52 * unit), (2.6 * unit, 2.4 * unit, .12 * unit), "minecraft:light_blue_stained_glass")
        _part(obj, (0, 2.5 * unit, 0), (3 * unit, .25 * unit, 1.8 * unit), "minecraft:gray_concrete")
        _part(obj, (1.5 * unit, 1.5 * unit, -.4 * unit), (.15 * unit, 3 * unit, .15 * unit), iron)
        _part(obj, (1.5 * unit, 2.8 * unit, -.4 * unit), (.65 * unit, .65 * unit, .15 * unit), "minecraft:yellow_concrete")
    elif subtype == "playground":
        for x in (-1.5 * unit, 1.5 * unit):
            _part(obj, (x, 1.5 * unit, 0), (.25 * unit, 3 * unit, .25 * unit), "minecraft:red_concrete")
        _part(obj, (0, 3 * unit, 0), (3.3 * unit, .25 * unit, .25 * unit), "minecraft:yellow_concrete")
        for x in (-.6 * unit, .6 * unit):
            _part(obj, (x, 1.85 * unit, 0), (.12 * unit, 2.1 * unit, .12 * unit), iron)
        _part(obj, (0, .8 * unit, 0), (1.4 * unit, .2 * unit, .55 * unit), "minecraft:blue_concrete")
    elif subtype == "crossing":
        # Root chooses its road-aligned heading/width when available.
        span = obj.get("width", 6 * unit)
        for x in np.arange(-span / 2 + .4 * unit, span / 2, 1.1 * unit):
            _part(obj, (float(x), -.42 * unit, 0), (.5 * unit, .18 * unit, 3 * unit), "minecraft:white_concrete")
    elif subtype == "fence":
        coordinates = obj["geometry"]["coordinates"]
        if obj["geometry"]["type"] == "polyline":
            original_center, original_base, original_heading = obj["center"], obj["base_y"], obj["heading"]
            for a, b in zip(coordinates, coordinates[1:]):
                distance = math.dist(a, b)
                if distance < .01:
                    continue
                steps = max(1, math.ceil(distance / (3 * unit)))
                heading = math.degrees(math.atan2(b[0] - a[0], -(b[1] - a[1]))) % 360
                for index in range(steps):
                    start = [a[i] + (b[i] - a[i]) * index / steps for i in (0, 1)]
                    end = [a[i] + (b[i] - a[i]) * (index + 1) / steps for i in (0, 1)]
                    obj["center"] = [(start[i] + end[i]) / 2 for i in (0, 1)]
                    obj["base_y"], obj["heading"] = _ground(scene, obj["center"]), heading
                    segment = distance / steps
                    _part(obj, (0, .7 * unit, segment / 2), (.25 * unit, 1.4 * unit, .25 * unit), "minecraft:oak_log[axis=y]")
                    for y in (.5, 1.1):
                        _part(obj, (0, y * unit, 0), (.18 * unit, .2 * unit, segment), oak)
            obj["center"], obj["base_y"], obj["heading"] = original_center, original_base, original_heading
    obj["height"] = max((p["center"][1] + p["size"][1] / 2 - obj["base_y"] for p in obj["parts"]), default=1)


def _mask(obj, width, depth, extra=0):
    im = Image.new("1", (width, depth))
    draw = ImageDraw.Draw(im)
    geom = obj["geometry"]
    points = [tuple(p) for p in geom["coordinates"]]
    if geom["type"] == "polyline":
        draw.line(points, fill=1, width=max(1, round(obj.get("width", 2) + extra)), joint="curve")
    elif geom["type"] == "polygon":
        draw.polygon(points, fill=1)
        for hole in geom.get("holes", []):
            draw.polygon([tuple(p) for p in hole], fill=0)
    return np.asarray(im, dtype=bool)


def _footprint(obj, margin=.2):
    """Pixel cells touched by a rotated vehicle, including lights and bumpers."""
    w, length = obj["width"] / 2 + margin, obj["length"] / 2 + margin
    theta = math.radians(-obj["heading"])
    c, s = math.cos(theta), math.sin(theta)
    return [(obj["center"][0] + x * c + z * s, obj["center"][1] - x * s + z * c)
            for x, z in [(-w, -length), (w, -length), (w, length), (-w, length)]]


def _footprint_cells(obj, width, depth, margin=.2):
    corners = _footprint(obj, margin)
    if any(not (0 <= x < width and 0 <= z < depth) for x, z in corners):
        return None
    x0, z0 = math.floor(min(p[0] for p in corners)), math.floor(min(p[1] for p in corners))
    x1, z1 = math.ceil(max(p[0] for p in corners)), math.ceil(max(p[1] for p in corners))
    image = Image.new("1", (x1 - x0 + 1, z1 - z0 + 1))
    ImageDraw.Draw(image).polygon([(x - x0, z - z0) for x, z in corners], fill=1)
    z, x = np.where(np.asarray(image))
    return z + z0, x + x0


def _photo_population_candidates(scene, photos):
    """Convert supplied model counts to uncertain bearings, never infer counts."""
    from .scene import _projection
    _, _, project = _projection(scene["bbox"], scene["meters_per_block"])
    candidates, dropped = [], []
    for index, photo in enumerate(photos or []):
        counts = photo.get("estimated_counts") or {}
        photo_id = str(photo.get("id") or index)
        for plural, kind in (("people", "person"), ("vehicles", "vehicle")):
            value = counts.get(plural)
            if value is None or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                continue
            count = min(100, int(value))
            heading = photo.get("heading")
            position = photo.get("position")
            if position is None:
                lat = photo.get("lat") if photo.get("lat") is not None else photo.get("exif_lat")
                lon = photo.get("lon") if photo.get("lon") is not None else photo.get("exif_lon")
                if lat is not None and lon is not None:
                    position = project(lon, lat)
            reason = None
            if heading is None or not math.isfinite(float(heading)):
                reason = "photo_heading_unknown"
            elif position is None or len(position) != 2 or not all(math.isfinite(float(p)) for p in position):
                reason = "photo_location_unknown"
            if reason:
                dropped.extend({"id": f"photo/{photo_id}/{kind}/{i}", "reason": reason} for i in range(count))
                continue
            fov = 360 if photo.get("projection") == "panorama" else min(360, max(1, float(photo.get("fov") or 70)))
            distance_m = min(150, max(5, float(photo.get("estimated_distance_m") or 25)))
            distance = distance_m / scene["meters_per_block"]
            for i in range(count):
                bearing = float(heading) + ((i + .5) / count - .5) * fov
                angle = math.radians(bearing)
                point = [float(position[0]) + math.sin(angle) * distance,
                         float(position[1]) - math.cos(angle) * distance]
                candidates.append({"id": f"photo/{photo_id}/{kind}/{i}", "kind": kind,
                    "position": point, "photo_id": photo_id, "photo_heading": float(heading),
                    "photo_fov": fov, "estimated_distance_m": distance_m,
                    "heading": (bearing + 180) % 360, "source": "photo_estimate", "confidence": .2})
    return candidates, dropped


def populate_scene(scene, annotations=None, photos=None):
    """Add plausible, repeatable street life; placement checks use whole vehicles."""
    width, depth = (scene["bounds"][key] for key in ("width", "depth"))
    scale = scene["meters_per_block"]
    rng = _rng((scene.get("bbox"), "population-v2"))
    objects = scene["objects"]
    forbidden = np.zeros((depth, width), dtype=bool)
    carriageway = np.zeros_like(forbidden)
    walkable = np.zeros_like(forbidden)
    parking = np.zeros_like(forbidden)
    overlap = np.zeros((depth, width), dtype=np.uint16)
    road_masks = {}
    roads, tracks = [], []
    for obj in objects:
        kind = obj["kind"]
        if kind in {"building", "water"}:
            forbidden |= _mask(obj, width, depth)
        elif kind == "park":
            walkable |= _mask(obj, width, depth)
        elif kind == "parking":
            parking |= _mask(obj, width, depth)
        elif kind == "tram_track":
            tracks.append(obj)
        elif kind == "road":
            mask = _mask(obj, width, depth)
            road_type = obj.get("road_type", "residential")
            if road_type in WALKING_ROADS:
                walkable |= mask
            elif road_type != "cycleway":
                carriageway |= mask
                overlap += mask.astype(np.uint16)
                road_masks[obj["id"]] = mask
                if road_type not in NO_PARKING_ROADS:
                    roads.append(obj)
                    if obj.get("sidewalk_width", 0) > 0:
                        walkable |= _mask(obj, width, depth, 2 * obj["sidewalk_width"]) & ~mask
        elif kind == "street_detail":
            decorate_street_detail(scene, obj)
    walkable &= ~forbidden & ~carriageway
    street_obstacles = forbidden.copy()
    # Reserve trunks, large shrubs and street furniture, without excluding tree canopies.
    for obj in objects:
        if obj["kind"] in {"tree", "shrub", "street_detail"}:
            center = obj.get("center")
            if not center or not isinstance(center[0], (float, int)):
                continue
            x, z = [round(v) for v in center]
            if 0 <= x < width and 0 <= z < depth:
                radius = 1 if obj["kind"] == "tree" else 0
                street_obstacles[max(0, z-radius):min(depth, z+radius+1), max(0, x-radius):min(width, x+radius+1)] = True
    # Suppress vehicles for several metres around every crossing of separate ways.
    intersection = overlap > 1
    dilation = max(2, round(4 / scale))
    intersection_image = Image.fromarray(intersection.astype(np.uint8) * 255)
    from PIL import ImageFilter
    intersection = np.asarray(intersection_image.filter(ImageFilter.MaxFilter(dilation * 2 + 1))) > 0
    used = np.zeros_like(forbidden)
    vehicle_counts = Counter()
    rejected = Counter()

    def accept_vehicle(vehicle, allowed, avoid_intersection=True):
        if vehicle.get("below_voxel_resolution"):
            rejected["below_voxel_resolution"] += 1
            return False
        cells = _footprint_cells(vehicle, width, depth)
        if cells is None:
            rejected["out_of_bounds"] += 1
            return False
        z, x = cells
        if not np.all(allowed[z, x]) or np.any(street_obstacles[z, x]):
            rejected["unsuitable_surface"] += 1
            return False
        if avoid_intersection and np.any(intersection[z, x]):
            rejected["intersection"] += 1
            return False
        if np.any(used[z, x]):
            rejected["occupied"] += 1
            return False
        objects.append(vehicle)
        used[z, x] = True
        vehicle_counts[vehicle["vehicle_type"]] += 1
        return True

    def refresh_vehicle_usage():
        used[:] = False
        vehicle_counts.clear()
        for vehicle in objects:
            if vehicle["kind"] == "vehicle":
                cells = _footprint_cells(vehicle, width, depth)
                if cells is not None:
                    used[cells] = True
                vehicle_counts[vehicle["vehicle_type"]] += 1

    types = ["sedan"] * 8 + ["hatchback"] * 6 + ["suv"] * 5 + ["pickup"] * 2 + ["van"] * 3 + ["taxi"] * 2 + ["police", "delivery_truck", "bus"]
    for road in sorted(roads, key=lambda obj: obj["id"]):
        if sum(vehicle_counts.values()) >= 400:
            break
        coordinates = road["geometry"]["coordinates"]
        for segment, (a, b) in enumerate(zip(coordinates, coordinates[1:])):
            length = math.dist(a, b)
            if length < max(7, 12 / scale):
                continue
            direction = [(b[i] - a[i]) / length for i in (0, 1)]
            heading = math.degrees(math.atan2(direction[0], -direction[1])) % 360
            spacing = max(8, 22 / scale)
            for index, distance in enumerate(np.arange(max(5, 9 / scale), length - max(5, 9 / scale), spacing)):
                if sum(vehicle_counts.values()) >= 400:
                    break
                kind = rng.choice(types)
                identity = f"population/{road['id']}/{segment}/car/{index}"
                # Rare large vehicles need a wider road and extra longitudinal space.
                if kind in {"bus", "delivery_truck"} and road["width"] * scale < 8:
                    kind = "van"
                vehicle = make_vehicle(scene, a, kind, heading=heading, object_id=identity)
                if vehicle["below_voxel_resolution"]:
                    rejected["below_voxel_resolution"] += 1
                    continue
                side = -1 if rng.random() < .5 else 1
                traffic = rng.random() < .15 and road["width"] > vehicle["width"] * 2.3
                # Pixel-rasterized road edges need a one-block inset to protect sidewalks.
                offset = max(0, (road["width"] - vehicle["width"]) / 2 - 1.2)
                if traffic:
                    offset = min(offset, road["width"] * .22)
                if road["width"] < vehicle["width"] + 1.1:
                    continue
                point = [float(a[0] + direction[0] * distance + direction[1] * offset * side),
                         float(a[1] + direction[1] * distance - direction[0] * offset * side)]
                vehicle = make_vehicle(scene, point, kind, heading=heading if side > 0 else heading + 180,
                                       object_id=identity, placement="traffic" if traffic else "parked")
                vehicle["road_id"] = road["id"]
                accept_vehicle(vehicle, road_masks[road["id"]])

    for lot in sorted((o for o in objects if o["kind"] == "parking"), key=lambda obj: obj["id"]):
        lot_mask = _mask(lot, width, depth)
        zs, xs = np.where(lot_mask)
        if not len(xs):
            continue
        lot_heading = 0
        polygon = lot["geometry"]["coordinates"]
        if len(polygon) > 1:
            a, b = max(zip(polygon, polygon[1:] + polygon[:1]), key=lambda pair: math.dist(*pair))
            lot_heading = math.degrees(math.atan2(b[0] - a[0], -(b[1] - a[1]))) % 360
        for z in range(int(zs.min()) + 3, int(zs.max()) - 2, max(6, round(8 / scale))):
            for x in range(int(xs.min()) + 3, int(xs.max()) - 2, max(4, round(4 / scale))):
                if rng.random() > .6 or sum(vehicle_counts.values()) >= 400:
                    continue
                vehicle = make_vehicle(scene, [x, z], rng.choice(types[:-3]), heading=lot_heading,
                                       object_id=f"population/{lot['id']}/car/{x}/{z}", placement="parking_lot")
                accept_vehicle(vehicle, lot_mask, avoid_intersection=False)

    # A streetcar is only plausible where OSM contains actual tram rails.
    for track in tracks:
        for index, (a, b) in enumerate(zip(track["geometry"]["coordinates"], track["geometry"]["coordinates"][1:])):
            if math.dist(a, b) < max(18, 30 / scale) or rng.random() > .25:
                continue
            point = [(a[i] + b[i]) / 2 for i in (0, 1)]
            heading = math.degrees(math.atan2(b[0] - a[0], -(b[1] - a[1]))) % 360
            vehicle = make_vehicle(scene, point, "streetcar", heading=heading,
                                   object_id=f"population/{track['id']}/tram/{index}", placement="tram_track")
            rail_mask = _mask({**track, "width": max(3.5, 3.2 / scale)}, width, depth)
            accept_vehicle(vehicle, (rail_mask & ~walkable) | carriageway)

    annotation_drops = []
    manual_people = []
    replaced_people = set()
    for index, annotation in enumerate(annotations or []):
        if annotation.get("kind") not in {"vehicle", "person"}:
            continue
        from .scene import _projection
        _, _, project = _projection(scene["bbox"], scale)
        point = annotation.get("position") or project(annotation["lon"], annotation["lat"])
        if not (0 <= point[0] < width and 0 <= point[1] < depth):
            annotation_drops.append({"id": annotation.get("id") or str(index), "reason": "outside_generation_bounds"})
            continue
        identity = f"annotation/{annotation.get('id') or index}"
        heading = annotation.get("heading", 0)
        if annotation["kind"] == "vehicle":
            nearest = None
            for road in roads:
                for a, b in zip(road["geometry"]["coordinates"], road["geometry"]["coordinates"][1:]):
                    dx, dz = b[0] - a[0], b[1] - a[1]
                    denominator = dx * dx + dz * dz
                    if denominator <= 0:
                        continue
                    t = max(0, min(1, ((point[0]-a[0])*dx+(point[1]-a[1])*dz)/denominator))
                    dist = math.dist(point, [a[0]+t*dx, a[1]+t*dz])
                    if nearest is None or dist < nearest[0]:
                        nearest = (dist, math.degrees(math.atan2(dx, -dz)) % 360)
            if nearest is not None and "heading" not in annotation:
                heading = nearest[1]
            vehicle = make_vehicle(scene, point, annotation.get("vehicle_type", "sedan"), annotation.get("color"),
                                   heading, identity, "user_annotation", 1., "annotated")
            if vehicle["below_voxel_resolution"]:
                rejected["below_voxel_resolution"] += 1
                annotation_drops.append({"id": annotation.get("id") or str(index), "reason": "vehicle_below_two_blocks_at_selected_scale"})
                continue
            # Explicit annotations take precedence over invented traffic. Clear
            # conflicting procedural cars only after validating the new footprint.
            cells = _footprint_cells(vehicle, width, depth)
            original_objects = None
            if cells is not None:
                z, x = cells
                valid = np.all((carriageway | parking)[z, x]) and not np.any((street_obstacles | intersection)[z, x])
                if valid:
                    original_objects = list(objects)
                    manual_cells = set(zip(z.tolist(), x.tolist()))
                    for old in list(objects):
                        if old["kind"] != "vehicle":
                            continue
                        if old["id"] == annotation.get("replace_id"):
                            objects.remove(old)
                            continue
                        if old["source"] != "procedural_population":
                            continue
                        old_cells = _footprint_cells(old, width, depth)
                        if old_cells is not None and manual_cells.intersection(zip(old_cells[0].tolist(), old_cells[1].tolist())):
                            objects.remove(old)
                    refresh_vehicle_usage()
            if not accept_vehicle(vehicle, carriageway | parking):
                if original_objects is not None:
                    objects[:] = original_objects
                    refresh_vehicle_usage()
                annotation_drops.append({"id": annotation.get("id") or str(index), "reason": "vehicle_requires_clear_road_or_parking_footprint"})
        else:
            x, z = min(width-1, round(point[0])), min(depth-1, round(point[1]))
            if forbidden[z, x] or carriageway[z, x]:
                annotation_drops.append({"id": annotation.get("id") or str(index), "reason": "pedestrian_on_building_water_or_carriageway"})
                continue
            if annotation.get("replace_id"):
                replaced_people.add(annotation["replace_id"])
                objects[:] = [o for o in objects if not (o["kind"] == "person" and o["id"] == annotation["replace_id"])]
                manual_people[:] = [o for o in manual_people if o["id"] != annotation["replace_id"]]
            manual_people.append(make_person(scene, point, heading, identity, "user_annotation", 1., annotation.get("appearance")))

    walkable_area = int(walkable.sum()) * scale * scale
    limit = min(300, math.floor(walkable_area / 400))
    # Manual annotations explicitly identify a person, but still respect the hard
    # scene cap. Their positions do not need an OSM-mapped walking surface.
    objects.extend(manual_people[:300])
    for person in manual_people[300:]:
        annotation_drops.append({"id": person["id"], "reason": "pedestrian_hard_limit_300"})

    photo_candidates, photo_drops = _photo_population_candidates(scene, photos)
    photo_placed = Counter()
    for candidate in photo_candidates:
        point = candidate["position"]
        reason = None
        if not (0 <= point[0] < width and 0 <= point[1] < depth):
            reason = "outside_generation_bounds"
        elif candidate["kind"] == "vehicle":
            nearest = None
            for road in roads:
                for a, b in zip(road["geometry"]["coordinates"], road["geometry"]["coordinates"][1:]):
                    dx, dz = b[0] - a[0], b[1] - a[1]
                    denominator = dx * dx + dz * dz
                    if denominator <= 0:
                        continue
                    t = max(0, min(1, ((point[0]-a[0])*dx+(point[1]-a[1])*dz)/denominator))
                    distance = math.dist(point, [a[0]+t*dx, a[1]+t*dz])
                    if nearest is None or distance < nearest[0]:
                        nearest = (distance, math.degrees(math.atan2(dx, -dz)) % 360)
            vehicle = make_vehicle(scene, point, heading=nearest[1] if nearest else candidate["heading"],
                                   object_id=candidate["id"], source="photo_estimate", confidence=.2,
                                   placement="photo_estimate")
            if sum(vehicle_counts.values()) >= 400:
                reason = "vehicle_population_limit"
            elif not accept_vehicle(vehicle, carriageway | parking):
                reason = "vehicle_below_two_blocks_at_selected_scale" if vehicle["below_voxel_resolution"] else "vehicle_requires_unoccupied_road_or_parking_footprint"
            else:
                obj = vehicle
        else:
            x, z = math.floor(point[0]), math.floor(point[1])
            people = [o for o in objects if o["kind"] == "person"]
            if len(people) >= limit:
                reason = "pedestrian_density_limit"
            elif not walkable[z, x] or street_obstacles[z, x] or used[z, x]:
                reason = "pedestrian_requires_clear_walkable_surface"
            elif any(math.dist(point, person["center"]) < max(2, 5 / scale) for person in people):
                reason = "pedestrian_position_occupied"
            else:
                obj = make_person(scene, point, candidate["heading"], candidate["id"], "photo_estimate", .2)
                objects.append(obj)
        if reason:
            photo_drops.append({"id": candidate["id"], "reason": reason})
        else:
            photo_placed[candidate["kind"]] += 1
            obj.update({key: candidate[key] for key in ("photo_id", "photo_heading", "photo_fov", "estimated_distance_m")})
            obj["appearance_method"] = "photo_count_bearing_estimate"

    candidate_z, candidate_x = np.where(walkable & ~street_obstacles & ~used)
    order = list(range(len(candidate_x)))
    rng.shuffle(order)
    separation = max(2, 5 / scale)
    placed_people = [(p["center"][0], p["center"][1]) for p in objects if p["kind"] == "person"]
    person_index = 0
    for candidate in order:
        if len(placed_people) >= limit:
            break
        x, z = int(candidate_x[candidate]), int(candidate_z[candidate])
        if any(math.hypot(x-px, z-pz) < separation for px, pz in placed_people):
            continue
        identity = f"population/person/{person_index}"
        person_index += 1
        if identity in replaced_people:
            continue
        person = make_person(scene, [x + .1, z + .1], rng.uniform(0, 360), identity)
        objects.append(person)
        placed_people.append((x, z))

    scene.setdefault("stats", {}).update({"pedestrians": len(placed_people),
        "pedestrian_walkable_area_m2": round(walkable_area, 2), "pedestrian_procedural_limit": limit,
        "vehicles": sum(vehicle_counts.values()), "vehicle_types": dict(sorted(vehicle_counts.items())),
        "vehicle_placements": dict(sorted(Counter(o.get("placement", "unknown") for o in objects if o["kind"] == "vehicle").items())),
        "vehicle_placement_rejections": dict(sorted(rejected.items())),
        "photo_estimates_placed": sum(o.get("source") == "photo_estimate" for o in objects),
        "photo_estimates_dropped": len(photo_drops) + sum(r.get("source") == "photo_estimate" and r.get("status") == "dropped" for r in scene.get("tree_placement", [])),
        "photo_population_estimates_placed": dict(sorted(photo_placed.items())),
        "photo_population_estimate_drops": photo_drops,
        "annotations_dropped": scene.get("stats", {}).get("annotations_dropped", []) + annotation_drops,
        "street_details": sum(o["kind"] == "street_detail" for o in objects)})
    scene["stats"].update(dict(Counter(o["kind"] for o in objects)))
    scene["stats"]["objects"] = len(objects)
    scene.setdefault("limitations", []).append("Pedestrians and vehicles are seeded decorative estimates; their identities, counts and positions are not observed real-world traffic.")
    if not math.isclose(scale, 1):
        scene["limitations"].append("Minecraft armour stands remain approximately two blocks tall regardless of map scale; pedestrians are decorative markers, not physically scaled humans at this resolution.")
    if rejected["below_voxel_resolution"]:
        scene["limitations"].append("Vehicles shorter than two blocks at the selected map scale are omitted instead of enlarging their real-world dimensions; omissions are counted in vehicle placement rejections.")
    if photo_candidates or photo_drops:
        scene["limitations"].append("Photo people/vehicle counts require supplied segmentation estimates; positions use camera heading, field of view and estimated distance with low confidence, not recovered 3D geometry.")
