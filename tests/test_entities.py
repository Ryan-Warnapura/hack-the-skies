"""Read entity NBT independently from writer helpers, including region boundaries."""
import io
import json
import math
import struct
import uuid
import zlib

import nbtlib
import numpy as np
import pytest

from forge.minecraft import read_schematic, scene_entities, write_java_world, write_schematic


def read_entity_chunk(world, cx, cz):
    raw = (world / "entities" / f"r.{cx // 32}.{cz // 32}.mca").read_bytes()
    slot = cx % 32 + cz % 32 * 32
    location = struct.unpack_from(">I", raw, slot * 4)[0]
    sector, count = location >> 8, location & 255
    assert sector >= 2 and count > 0
    offset = sector * 4096
    size = struct.unpack_from(">I", raw, offset)[0]
    assert size + 4 <= count * 4096 and raw[offset + 4] == 2
    root = nbtlib.File.parse(io.BytesIO(zlib.decompress(raw[offset + 5:offset + 4 + size])))
    assert int(root["DataVersion"]) == 3700
    assert list(map(int, root["Position"])) == [cx, cz]
    return root["Entities"]


def scene():
    return {
        "bbox": [-79.38, 43.65, -79.37, 43.66],
        "objects": [
            {"kind": "person", "id": "pedestrian/1", "geometry": {"coordinates": [15.999, 2.5]},
             "base_y": 5, "yaw": 90,
             "outfit": {"top": 0xBB5522, "bottom": 0x223399, "shoes": 0x181818},
             "pose": {"Head": [5, -12, 0], "LeftArm": [15, 0, -4], "RightLeg": [-8, 0, 0]}},
            {"kind": "person", "id": "pedestrian/2", "geometry": {"coordinates": [16.0, 2.5]},
             "base_y": 5, "yaw": 0,
             "outfit": {"top": 0x448866, "bottom": 0x444444, "shoes": 0x999999},
             "pose": {"RightArm": [-25, 0, 6]}},
            {"kind": "person", "id": "pedestrian/3", "geometry": {"coordinates": [512, 16]},
             "base_y": 6, "yaw": 270, "outfit": {}, "pose": {}},
            {"kind": "tree", "id": "ignore-this"},
        ],
    }


def blocks():
    result = np.zeros((9, 17, 513), dtype=np.uint16)
    result[:5] = 1
    return result


def uuid_from_tag(tag):
    return str(uuid.UUID(bytes=struct.pack(">4i", *map(int, tag["UUID"]))))


def test_semantic_people_export_real_sponge_entities_and_stable_uuids(tmp_path):
    entities = scene_entities(scene())
    assert len(entities) == 3
    assert entities == scene_entities({**scene(), "job_id": "a-different-generation"})
    assert [e["yaw"] for e in entities] == [-90, -180, 90]
    assert len({e["uuid"] for e in entities}) == 3
    path = write_schematic(tmp_path / "people.schem", blocks(), ["minecraft:air", "minecraft:stone"],
                           {"name": "People"}, entities=entities)
    root = nbtlib.load(path, gzipped=True)
    assert len(root["Entities"]) == 3
    first = root["Entities"][0]
    assert str(first["Id"]) == "minecraft:armor_stand" and "id" not in first
    assert first["Pos"].subtype is nbtlib.Double
    assert first["Rotation"].subtype is nbtlib.Float
    assert [float(v) for v in first["Pos"]] == [15.999, 5.0, 2.5]
    assert [float(v) for v in first["Rotation"]] == [-90, 0]
    assert uuid_from_tag(first) == entities[0]["uuid"]
    assert all(int(first[name]) == 1 for name in ("ShowArms", "NoBasePlate", "NoGravity", "Invulnerable"))
    assert len(first["ArmorItems"]) == 4 and len(first["HandItems"]) == 2
    for item, name, color in zip(first["ArmorItems"], ("boots", "leggings", "chestplate"), (0x181818, 0x223399, 0xBB5522)):
        assert str(item["id"]) == f"minecraft:leather_{name}"
        assert isinstance(item["Count"], nbtlib.Byte) and int(item["Count"]) == 1
        assert int(item["tag"]["display"]["color"]) == color
    assert list(map(float, first["Pose"]["Head"])) == [5, -12, 0]
    assert first["Pose"]["Head"].subtype is nbtlib.Float
    _, _, metadata = read_schematic(path)
    assert metadata["name"] == "People"
    assert metadata["entities"][0]["outfit"]["top"] == 0xBB5522
    assert metadata["entities"][0]["pose"]["Head"] == [5, -12, 0]


def test_java_entity_regions_across_chunks_and_regions_read_actual_schematic(tmp_path):
    entities = scene_entities(scene())
    path = write_schematic(tmp_path / "people.schem", blocks(), ["minecraft:air", "minecraft:stone"], entities=entities)
    root = nbtlib.load(path, gzipped=True)
    # Deliberately contradict the real NBT using JSON metadata and stale RAM.
    root["Metadata"]["RealityForge"] = nbtlib.String(json.dumps({"entities": [{"pos": [1, 1, 1]}]}))
    root["Entities"][0]["ArmorItems"][2]["tag"]["display"]["color"] = nbtlib.Int(0xFFFF00)
    root["Entities"][0]["Pos"][1] = nbtlib.Double(6)
    root.save(path)
    entities[0]["pos"] = [0, 0, 0]
    _, _, metadata = read_schematic(path)
    assert metadata["entities"][0]["pos"] == [15.999, 6, 2.5]
    world = write_java_world(path, tmp_path / "world")
    assert sorted(p.name for p in (world / "entities").glob("*.mca")) == ["r.0.0.mca", "r.1.0.mca"]
    all_entities = []
    for cx, cz in [(0, 0), (1, 0), (32, 1)]:
        tagged = read_entity_chunk(world, cx, cz)
        assert len(tagged) == 1
        entity = tagged[0]
        assert str(entity["id"]) == "minecraft:armor_stand" and "Id" not in entity
        x, y, z = map(float, entity["Pos"])
        assert math.floor(x / 16) == cx and math.floor(z / 16) == cz
        assert y == (65 if cx == 1 else 66)
        all_entities.append(entity)
    assert int(all_entities[0]["ArmorItems"][2]["tag"]["display"]["color"]) == 0xFFFF00
    assert list(map(float, all_entities[0]["Pose"]["Head"])) == [5, -12, 0]
    assert [uuid_from_tag(e) for e in all_entities] == [e["uuid"] for e in scene_entities(scene())]
    # Entity NBT stays the same between jobs (file timestamps intentionally do not).
    second = write_java_world(path, tmp_path / "second")
    assert read_entity_chunk(second, 0, 0).snbt() == read_entity_chunk(world, 0, 0).snbt()


def test_empty_entities_preserve_legacy_metadata_and_world(tmp_path):
    metadata = {"name": "Old contract", "bbox": [1, 2, 3, 4]}
    path = write_schematic(tmp_path / "empty.schem", np.ones((2, 2, 2), dtype=np.uint16),
                           ["minecraft:air", "minecraft:stone"], metadata)
    assert len(nbtlib.load(path, gzipped=True)["Entities"]) == 0
    assert read_schematic(path)[2] == metadata
    world = write_java_world(path, tmp_path / "world")
    assert not (world / "entities").exists()


@pytest.mark.parametrize("pos", [[float("nan"), 1, 0], [-0.01, 1, 0], [2, 1, 0], [0, 1, 2], [0, -1, 0], [0, 3, 0]])
def test_bad_entity_coordinates_fail_before_output(tmp_path, pos):
    path = tmp_path / "invalid.schem"
    with pytest.raises(ValueError, match="finite|outside"):
        write_schematic(path, np.zeros((2, 2, 2), dtype=np.uint16), ["minecraft:stone"],
                        entities=[{"id": "minecraft:armor_stand", "pos": pos}])
    assert not path.exists()


def test_duplicate_entity_uuids_rejected(tmp_path):
    entity = {"id": "minecraft:armor_stand", "pos": [0, 1, 0], "uuid": str(uuid.UUID(int=1))}
    with pytest.raises(ValueError, match="unique"):
        write_schematic(tmp_path / "duplicate.schem", np.zeros((2, 2, 2), dtype=np.uint16),
                        ["minecraft:stone"], entities=[entity, entity])
