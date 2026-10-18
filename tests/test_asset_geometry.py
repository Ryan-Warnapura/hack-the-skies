"""Canonical asset parts feed both semantic meshes and block exports."""
from pathlib import Path

import numpy as np

from forge.assets import BLOCK_COLORS, part_mesh
from forge.scene import DEFAULT_MATERIALS, export_obj, material_color, scene_to_voxels


def minimal_scene(objects):
    return {"bounds": {"width": 16, "depth": 16, "height": 16},
            "terrain": {"heights": [[4]]}, "objects": objects}


def object_with_parts(kind, parts, identity="asset"):
    return {"kind": kind, "id": identity, "geometry": {"coordinates": [8, 8], "type": "point"},
            "base_y": 5, "parts": parts}


def test_rotated_vehicle_has_matching_obj_bounds_voxel_orientation_and_material(tmp_path):
    part = {"shape": "box", "center": [8, 5.5, 8], "size": [2, 1, 4], "yaw": 90,
            "material": "minecraft:red_concrete"}
    scene = minimal_scene([object_with_parts("vehicle", [part])])
    blocks, palette = scene_to_voxels(scene)
    actual = np.argwhere(blocks == palette.index(part["material"]))
    assert len(actual) == 8
    assert actual.min(axis=0).tolist() == [5, 7, 6]
    assert actual.max(axis=0).tolist() == [5, 8, 9]
    # The scene has east-west length four, regardless of its original local Z.
    path = export_obj(scene, tmp_path / "car.obj")
    text = path.read_text()
    asset_text = text.split("o asset_part_0\n", 1)[1]
    verts = np.array([[float(v) for v in line.split()[1:]] for line in asset_text.splitlines() if line.startswith("v ")])
    np.testing.assert_allclose(verts.min(axis=0), [6, 5, 7])
    np.testing.assert_allclose(verts.max(axis=0), [10, 6, 9])
    assert "usemtl minecraft_red_concrete" in asset_text
    color = BLOCK_COLORS["red_concrete"]
    kd = " ".join(f"{int(color[i:i + 2], 16)/255:.4f}" for i in (1, 3, 5))
    assert f"newmtl minecraft_red_concrete\nKd {kd}" in path.with_suffix(".mtl").read_text()


def test_thin_details_survive_and_people_are_meshes_but_not_duplicate_blocks(tmp_path):
    pole = {"shape": "box", "center": [3.5, 6.5, 3.5], "size": [.12, .12, .12],
            "material": "minecraft:iron_bars"}
    person = {"shape": "box", "center": [12.5, 5.5, 12.5], "size": [1, 1, 1],
              "material": "minecraft:purple_concrete"}
    scene = minimal_scene([object_with_parts("street_detail", [pole], "thin_pole"),
                           object_with_parts("person", [person], "pedestrian")])
    blocks, palette = scene_to_voxels(scene)
    assert palette[blocks[6, 3, 3]] == "minecraft:iron_bars"
    assert palette[blocks[5, 12, 12]] == "minecraft:air"
    assert "minecraft:purple_concrete" not in palette
    text = export_obj(scene, tmp_path / "details.obj").read_text()
    assert "o thin_pole_part_0" in text and "o pedestrian_part_0" in text
    assert "usemtl minecraft_purple_concrete" in text


def test_tree_parts_override_legacy_canopy_and_share_leaf_color(tmp_path):
    bare_trunk = {"shape": "box", "center": [8.5, 7, 8.5], "size": [1, 4, 1],
                  "material": "minecraft:birch_log[axis=y]"}
    tree = object_with_parts("tree", [bare_trunk], "winter_birch")
    tree.update({"height": 9, "canopy_radius": 3, "materials": dict(DEFAULT_MATERIALS)})
    blocks, palette = scene_to_voxels(minimal_scene([tree]))
    assert np.count_nonzero(blocks == palette.index(bare_trunk["material"])) == 4
    assert not any("leaves" in name for name in palette)
    text = export_obj(minimal_scene([tree]), tmp_path / "winter.obj").read_text()
    assert "winter_birch_part_0" in text and "_crown" not in text
    assert material_color("minecraft:cherry_leaves[persistent=true]") == BLOCK_COLORS["cherry_leaves"]
    assert material_color("minecraft:birch_log[axis=y]") == BLOCK_COLORS["birch_log"]


def test_legacy_sidewalk_tree_survives_carriageway_mask():
    road = {"id": "road", "kind": "road", "geometry": {"coordinates": [[8, 0], [8, 15]]},
            "base_y": 5, "width": 4, "sidewalk_width": 4, "road_type": "residential",
            "markings": False, "materials": dict(DEFAULT_MATERIALS)}
    trees = [{"id": f"tree/{x}", "kind": "tree", "geometry": {"coordinates": [x, 8]},
              "base_y": 5, "height": 5, "canopy_radius": 1, "materials": dict(DEFAULT_MATERIALS)}
             for x in [3.5, 8.5]]
    blocks, palette = scene_to_voxels(minimal_scene([road, *trees]))
    assert palette[blocks[5, 8, 3]] == "minecraft:oak_log[axis=y]"
    assert palette[blocks[5, 8, 8]] == "minecraft:air"
    assert palette[blocks[4, 8, 3]] == "minecraft:smooth_stone"


def test_parking_surface_is_present_in_voxels_and_mesh(tmp_path):
    lot = {"id": "parking", "kind": "parking", "base_y": 5,
           "geometry": {"coordinates": [[1, 1], [6, 1], [6, 6], [1, 6]], "holes": []},
           "materials": {"surface": "minecraft:gray_concrete"}}
    scene = minimal_scene([lot])
    blocks, palette = scene_to_voxels(scene)
    assert palette[blocks[4, 3, 3]] == "minecraft:gray_concrete"
    assert palette[blocks[5, 3, 3]] == "minecraft:air"
    text = export_obj(scene, tmp_path / "lot.obj").read_text()
    assert "o parking\nusemtl minecraft_gray_concrete" in text


def test_ellipsoid_mesh_and_voxels_have_consistent_world_location():
    crown = {"shape": "ellipsoid", "center": [8.5, 9.5, 8.5], "size": [5, 5, 5],
             "material": "minecraft:cherry_leaves[persistent=true]"}
    vertices, _ = part_mesh(crown)
    vertices = np.asarray(vertices)
    assert vertices[:, 1].min() == 7 and vertices[:, 1].max() == 12
    blocks, palette = scene_to_voxels(minimal_scene([object_with_parts("tree", [crown])]))
    cells = np.argwhere(blocks == palette.index(crown["material"]))
    assert len(cells) > 40
    assert cells.min(axis=0).tolist() == [7, 6, 6]
    assert cells.max(axis=0).tolist() == [11, 10, 10]
    assert palette[blocks[9, 8, 8]] == crown["material"]
    assert palette[blocks[7, 6, 6]] == "minecraft:air"
