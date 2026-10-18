"""Check semantic placement rules, not just the existence of generated assets."""
import copy
import math

import numpy as np
import pytest

from forge.population import (
    VEHICLE_DIMENSIONS, _footprint_cells, _mask, decorate_street_detail,
    make_person, make_vehicle, populate_scene,
)


def _scene(diagonal=False):
    road = {"id": "road/east-west", "kind": "road", "source": "osm", "width": 10,
            "road_type": "residential", "sidewalk_width": 2, "base_y": 5,
            "geometry": {"type": "polyline", "coordinates": [[0, 60], [120, 60]]}}
    second = {**road, "id": "road/north-south",
              "geometry": {"type": "polyline", "coordinates": [[60, 0], [60, 120]]}}
    if diagonal:
        road["geometry"]["coordinates"] = [[4, 116], [116, 4]]
    park = {"id": "park", "kind": "park", "source": "osm", "base_y": 5,
            "geometry": {"type": "polygon", "coordinates": [[5, 5], [45, 5], [45, 45], [5, 45]], "holes": []}}
    return {"bbox": [-79.38, 43.65, -79.37851, 43.65108], "meters_per_block": 1,
            "bounds": {"width": 120, "depth": 120, "height": 32}, "terrain": {"heights": [[4]]},
            "materials": {}, "objects": [road, park] if diagonal else [road, second, park], "stats": {}}


def test_population_is_deterministic_and_density_is_bounded():
    first, second = _scene(), _scene()
    populate_scene(first)
    populate_scene(second)
    assert first == second
    pedestrians = [o for o in first["objects"] if o["kind"] == "person"]
    assert 2 <= len(pedestrians) <= min(300, first["stats"]["pedestrian_walkable_area_m2"] // 400)
    assert len({o["outfit"]["top"] for o in pedestrians}) > 1
    assert len({tuple(o["pose"]["LeftLeg"]) for o in pedestrians}) > 1
    assert all(o["entity_type"] == "minecraft:armor_stand" for o in pedestrians)
    assert all(o["source"] == "procedural_population" and o["confidence"] < .5 for o in pedestrians)


def test_people_avoid_carriageways_and_buildings():
    scene = _scene()
    building = {"id": "obstacle", "kind": "building", "geometry": {
        "type": "polygon", "coordinates": [[10, 10], [35, 10], [35, 35], [10, 35]]}}
    scene["objects"].append(building)
    disallowed = _mask(building, 120, 120).copy()
    for road in scene["objects"]:
        if road["kind"] == "road":
            disallowed |= _mask(road, 120, 120)
    populate_scene(scene)
    people = [o for o in scene["objects"] if o["kind"] == "person"]
    assert people
    for person in people:
        x, z = person["center"]
        assert not disallowed[int(z), int(x)]


def test_vehicle_full_footprints_avoid_sidewalks_and_intersection():
    scene = _scene()
    roads = {o["id"]: o for o in scene["objects"] if o["kind"] == "road"}
    populate_scene(scene)
    vehicles = [o for o in scene["objects"] if o["kind"] == "vehicle"]
    assert len(vehicles) >= 4
    occupied = np.zeros((120, 120), dtype=bool)
    crossing = _mask(roads["road/east-west"], 120, 120) & _mask(roads["road/north-south"], 120, 120)
    for vehicle in vehicles:
        z, x = _footprint_cells(vehicle, 120, 120)
        assert np.all(_mask(roads[vehicle["road_id"]], 120, 120)[z, x])
        assert not np.any(crossing[z, x])
        assert not np.any(occupied[z, x])
        occupied[z, x] = True
        expected = 90 if vehicle["road_id"] == "road/east-west" else 0
        assert math.isclose((vehicle["heading"] - expected) % 180, 0, abs_tol=.001)
    assert not any(o["vehicle_type"] == "streetcar" for o in vehicles)


def test_vehicles_align_to_oblique_road_and_parts_share_heading():
    scene = _scene(diagonal=True)
    populate_scene(scene)
    vehicles = [o for o in scene["objects"] if o["kind"] == "vehicle"]
    assert vehicles
    for vehicle in vehicles:
        assert math.isclose((vehicle["heading"] - 45) % 180, 0, abs_tol=.001)
        assert all(math.isclose(p["yaw"], -vehicle["heading"], abs_tol=.001) for p in vehicle["parts"])
        front = [p for p in vehicle["parts"] if p["material"] == "minecraft:sea_lantern"][0]
        vx, vz = front["center"][0] - vehicle["center"][0], front["center"][2] - vehicle["center"][1]
        theta = math.radians(vehicle["heading"])
        assert vx * math.sin(theta) - vz * math.cos(theta) > 0


def test_all_ten_vehicle_assets_have_distinct_silhouettes_and_materials():
    scene = _scene()
    assets = [make_vehicle(scene, [30, 30], name, object_id=name) for name in VEHICLE_DIMENSIONS]
    assert len(assets) == 10
    assert len({(o["width"], o["height"], o["length"], len(o["parts"])) for o in assets}) >= 8
    for vehicle in assets:
        palette = {p["material"] for p in vehicle["parts"]}
        assert {"minecraft:black_concrete", "minecraft:light_blue_stained_glass", "minecraft:sea_lantern", "minecraft:redstone_block"} <= palette
        assert all(p["shape"] in {"box", "ellipsoid"} for p in vehicle["parts"])
    coarse = copy.deepcopy(scene)
    coarse["meters_per_block"] = 4
    assert make_vehicle(coarse, [30, 30])["length"] < make_vehicle(scene, [30, 30])["length"]


def test_annotations_keep_provenance_and_reject_invalid_vehicle_surfaces():
    scene = _scene()
    populate_scene(scene, [
        {"id": "car", "kind": "vehicle", "position": [30, 60], "vehicle_type": "taxi", "color": "#EDB900"},
        {"id": "person", "kind": "person", "position": [20, 20], "appearance": {"top": "#AB1234"}},
        {"id": "badcar", "kind": "vehicle", "position": [20, 20]},
    ])
    manual = {o["id"]: o for o in scene["objects"] if o["source"] == "user_annotation"}
    assert set(manual) == {"annotation/car", "annotation/person"}
    assert all(o["confidence"] == 1 for o in manual.values())
    assert manual["annotation/person"]["outfit"]["top"] == 0xAB1234
    assert manual["annotation/car"]["vehicle_type"] == "taxi"
    assert manual["annotation/car"]["heading"] == 90
    assert any(item["id"] == "badcar" and "road_or_parking" in item["reason"] for item in scene["stats"]["annotations_dropped"])


def test_street_furniture_has_shared_nonempty_geometry():
    scene = _scene()
    for subtype in ["bench", "streetlight", "bollard", "bus_stop", "crossing", "playground", "fence"]:
        obj = {"id": subtype, "kind": "street_detail", "subtype": subtype,
               "center": [25, 25], "geometry": {"type": "point", "coordinates": [25, 25]}}
        if subtype == "fence":
            obj["geometry"] = {"type": "polyline", "coordinates": [[10, 10], [25, 25]]}
        decorate_street_detail(scene, obj)
        assert obj["parts"], subtype
        assert all(len(part["center"]) == len(part["size"]) == 3 for part in obj["parts"])


def test_person_outfits_accept_hex_and_generate_stable_poses():
    first = make_person(_scene(), [20, 20], object_id="test", appearance={"top": "#29AA33", "bottom": 0x304080})
    second = make_person(_scene(), [20, 20], object_id="test", appearance={"top": "#29AA33", "bottom": 0x304080})
    assert first == second
    assert first["outfit"]["top"] == 0x29AA33
    assert first["outfit"]["bottom"] == 0x304080
    assert len(first["pose"]) == 6


def test_corrections_replace_only_after_validating_new_location():
    baseline = _scene()
    populate_scene(baseline)
    original = next(o for o in baseline["objects"] if o["kind"] == "vehicle")
    invalid = _scene()
    populate_scene(invalid, [{"id": "bad_move", "kind": "vehicle", "position": [20, 20],
                             "replace_id": original["id"], "vehicle_type": "taxi"}])
    assert original in invalid["objects"]
    corrected = _scene()
    populate_scene(corrected, [{"id": "new_car", "kind": "vehicle", "position": original["center"],
                               "replace_id": original["id"], "vehicle_type": "sedan"},
                              {"id": "new_person", "kind": "person", "position": [20, 20],
                               "replace_id": "population/person/0"}])
    assert not any(o["id"] == original["id"] for o in corrected["objects"])
    assert not any(o["id"] == "population/person/0" for o in corrected["objects"])
    assert any(o["id"] == "annotation/new_car" for o in corrected["objects"])
    assert any(o["id"] == "annotation/new_person" for o in corrected["objects"])


def test_parking_lot_vehicles_stay_within_lot_and_do_not_create_streetcars():
    scene = _scene()
    scene["objects"] = [{"id": "lot", "kind": "parking", "source": "osm", "geometry": {
        "type": "polygon", "coordinates": [[10, 10], [80, 10], [80, 90], [10, 90]]}}]
    surface = _mask(scene["objects"][0], 120, 120)
    populate_scene(scene)
    cars = [o for o in scene["objects"] if o["kind"] == "vehicle"]
    assert cars
    for car in cars:
        z, x = _footprint_cells(car, 120, 120)
        assert np.all(surface[z, x])
        assert car["placement"] == "parking_lot"
        assert car["vehicle_type"] != "streetcar"


@pytest.mark.parametrize("scale", [.5, 2, 4, 8])
def test_vehicle_geometry_preserves_real_dimensions_at_every_scale(scale):
    fine, coarse = _scene(), _scene()
    coarse["meters_per_block"] = scale
    for name, dimensions in VEHICLE_DIMENSIONS.items():
        original = make_vehicle(fine, [25, 25], name, heading=37, object_id=name)
        scaled = make_vehicle(coarse, [25, 25], name, heading=37, object_id=name)
        for key, metres in zip(("width", "height", "length"), dimensions):
            assert math.isclose(scaled[key] * scale, metres)
        anchor = [25, 5, 25]
        for reference_part, scaled_part in zip(original["parts"], scaled["parts"]):
            assert np.allclose(np.array(scaled_part["size"]) * scale, reference_part["size"], atol=.00005)
            assert np.allclose((np.array(scaled_part["center"]) - anchor) * scale,
                               np.array(reference_part["center"]) - anchor, atol=.00005)
        assert scaled["below_voxel_resolution"] == (dimensions[2] / scale < 2)


def test_coarse_population_omits_tiny_vehicles_and_reports_why():
    scene = _scene()
    scene["meters_per_block"] = 8
    populate_scene(scene, [{"id": "tiny", "kind": "vehicle", "position": [30, 60]}])
    assert not any(o["kind"] == "vehicle" for o in scene["objects"])
    assert scene["stats"]["vehicle_placement_rejections"]["below_voxel_resolution"] > 0
    assert {"id": "tiny", "reason": "vehicle_below_two_blocks_at_selected_scale"} in scene["stats"]["annotations_dropped"]
    assert any("armour stands" in text and "map scale" in text for text in scene["limitations"])
    assert any("omitted instead of enlarging" in text for text in scene["limitations"])


@pytest.mark.parametrize("subtype", ["bench", "streetlight", "bollard", "bus_stop", "playground", "crossing"])
def test_street_furniture_scales_without_minimum_size_inflation(subtype):
    fine, coarse = _scene(), _scene()
    coarse["meters_per_block"] = 8
    make_object = lambda: {"id": subtype, "kind": "street_detail", "subtype": subtype,
                           "center": [25, 25], "geometry": {"type": "point", "coordinates": [25, 25]}}
    original, scaled = make_object(), make_object()
    decorate_street_detail(fine, original)
    decorate_street_detail(coarse, scaled)
    assert len(original["parts"]) == len(scaled["parts"])
    for reference_part, scaled_part in zip(original["parts"], scaled["parts"]):
        assert np.allclose(np.array(scaled_part["size"]) * 8, reference_part["size"], atol=.005)
        assert np.allclose((np.array(scaled_part["center"]) - [25, 5, 25]) * 8,
                           np.array(reference_part["center"]) - [25, 5, 25], atol=.005)


def test_fence_height_and_thickness_preserve_metres_in_coarse_world():
    scene = _scene()
    scene["meters_per_block"] = 8
    fence = {"id": "fence", "kind": "street_detail", "subtype": "fence", "center": [25, 25],
             "geometry": {"type": "polyline", "coordinates": [[10, 10], [25, 25]]}}
    decorate_street_detail(scene, fence)
    post = next(p for p in fence["parts"] if "oak_log" in p["material"])
    assert post["size"] == [.031, .175, .031]
    assert math.isclose(post["size"][1] * 8, 1.4)


def test_optional_annotation_ids_produce_unique_semantic_ids():
    scene = _scene()
    populate_scene(scene, [{"id": None, "kind": "person", "position": [20, 20]},
                           {"id": None, "kind": "person", "position": [30, 30]}])
    assert {o["id"] for o in scene["objects"] if o["source"] == "user_annotation"} == {"annotation/0", "annotation/1"}


def test_photo_estimates_use_bearing_and_record_invalid_placements():
    photos = [{"id": "people", "position": [20, 30], "heading": 0, "fov": 70,
               "estimated_distance_m": 10, "estimated_counts": {"people": 1, "vehicles": None}},
              {"id": "car", "position": [20, 80], "heading": 0, "fov": 70,
               "estimated_distance_m": 20, "estimated_counts": {"people": None, "vehicles": 1}},
              {"id": "intersection", "position": [60, 80], "heading": 0, "fov": 70,
               "estimated_distance_m": 20, "estimated_counts": {"vehicles": 1}}]
    first, second = _scene(), _scene()
    populate_scene(first, photos=photos)
    populate_scene(second, photos=photos)
    assert first == second
    estimates = {o["kind"]: o for o in first["objects"] if o["source"] == "photo_estimate"}
    assert set(estimates) == {"person", "vehicle"}
    assert estimates["person"]["center"] == [20, 20]
    assert estimates["vehicle"]["center"] == [20, 60]
    assert estimates["vehicle"]["heading"] == 90
    assert all(o["confidence"] == .2 and o["photo_id"] for o in estimates.values())
    assert first["stats"]["photo_estimates_placed"] == 2
    assert first["stats"]["photo_estimates_dropped"] == 1
    assert first["stats"]["pedestrians"] <= first["stats"]["pedestrian_procedural_limit"]
    assert "road_or_parking" in first["stats"]["photo_population_estimate_drops"][0]["reason"]


def test_photo_fallback_counts_do_not_invent_people_or_vehicles():
    scene = _scene()
    populate_scene(scene, photos=[{"id": "fallback", "position": [20, 30], "heading": 0,
                                  "estimated_counts": {"trees": 8, "people": None, "vehicles": None}}])
    assert not any(o["source"] == "photo_estimate" for o in scene["objects"])
    assert scene["stats"]["photo_estimates_placed"] == 0
    assert scene["stats"]["photo_estimates_dropped"] == 0


def test_photo_estimates_do_not_replace_manual_annotations():
    scene = _scene()
    populate_scene(scene, annotations=[{"id": "explicit", "kind": "person", "position": [20, 20]}],
                   photos=[{"id": "uncertain", "position": [20, 30], "heading": 0,
                            "estimated_distance_m": 10, "estimated_counts": {"people": 1}}])
    manual = next(o for o in scene["objects"] if o["id"] == "annotation/explicit")
    assert manual["source"] == "user_annotation" and manual["confidence"] == 1
    assert scene["stats"]["photo_estimates_placed"] == 0
    assert scene["stats"]["photo_population_estimate_drops"][0]["reason"] == "pedestrian_position_occupied"


def test_bench_seat_uses_a_real_java_slab_block_state():
    scene = _scene()
    bench = {"id": "bench", "kind": "street_detail", "subtype": "bench", "center": [25, 25],
             "geometry": {"type": "point", "coordinates": [25, 25]}}
    decorate_street_detail(scene, bench)
    assert bench["parts"][0]["material"] == "minecraft:oak_slab[type=bottom,waterlogged=false]"
