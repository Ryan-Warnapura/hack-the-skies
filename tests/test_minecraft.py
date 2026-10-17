"""Read exported binary formats independently of the writer's packing helpers."""
import io
import struct
import zlib

import nbtlib
import numpy as np
import pytest

from forge.minecraft import DATA_VERSION, WORLD_BASE_Y, read_schematic, write_java_world, write_schematic


def decode_varints(raw):
    output, current, shift = [], 0, 0
    for value in bytes(np.asarray(raw, dtype=np.int8).view(np.uint8)):
        current |= (value & 127) << shift
        if value & 128:
            shift += 7
        else:
            output.append(current)
            current = shift = 0
    assert shift == 0
    return output


def read_chunk(world, cx, cz):
    data = (world / "region" / f"r.{cx // 32}.{cz // 32}.mca").read_bytes()
    index = cx % 32 + cz % 32 * 32
    location = struct.unpack_from(">I", data, index * 4)[0]
    sector, sectors = location >> 8, location & 255
    assert sector >= 2 and sectors > 0
    offset = sector * 4096
    size = struct.unpack_from(">I", data, offset)[0]
    assert 4 + size <= sectors * 4096
    assert data[offset + 4] == 2  # Anvil zlib compression
    root = nbtlib.File.parse(io.BytesIO(zlib.decompress(data[offset + 5:offset + 4 + size])))
    assert int(root["xPos"]) == cx and int(root["zPos"]) == cz
    assert int(root["DataVersion"]) == 3700
    assert str(root["Status"]) == "minecraft:full"
    return root


def unpack_longs(tag, bits, count):
    words = [int(value) & ((1 << 64) - 1) for value in tag]
    per_word = 64 // bits
    mask = (1 << bits) - 1
    return [(words[index // per_word] >> ((index % per_word) * bits)) & mask for index in range(count)]


def block_at(root, x, y, z):
    section = next(section for section in root["sections"] if int(section["Y"]) == y // 16)
    states = section["block_states"]
    palette = states["palette"]
    index = (y % 16) * 256 + z * 16 + x
    state_id = unpack_longs(states["data"], max(4, (len(palette) - 1).bit_length()), 4096)[index] if "data" in states else 0
    state = palette[state_id]
    name = str(state["Name"])
    if "Properties" in state:
        properties = ",".join(f"{key}={value}" for key, value in sorted(state["Properties"].items()))
        name += f"[{properties}]"
    return name


def test_schematic_order_metadata_and_multibyte_varints(tmp_path):
    palette = [f"minecraft:test_block_{index}" for index in range(16385)]
    blocks = np.array([[[0, 1, 127], [128, 255, 16383]], [[16384, 1, 128], [127, 0, 255]]], dtype=np.uint16)
    metadata = {"name": "Intersection", "bbox": [-79.4, 43.6, -79.3, 43.7], "photos": 2}
    path = write_schematic(tmp_path / "test.schem", blocks, palette, metadata)
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    raw = nbtlib.load(str(path), gzipped=True)
    assert raw.root_name == "Schematic"
    assert [int(raw[key]) for key in ("Width", "Height", "Length")] == [3, 2, 2]
    assert int(raw["Version"]) == 2 and int(raw["DataVersion"]) == DATA_VERSION
    assert decode_varints(raw["BlockData"]) == [0, 1, 127, 128, 255, 16383, 16384, 1, 128, 127, 0, 255]
    assert int(raw["Palette"]["minecraft:test_block_16384"]) == 16384
    actual, actual_palette, actual_metadata = read_schematic(path)
    np.testing.assert_array_equal(actual, blocks)
    assert actual.dtype == np.uint16 and actual_palette == palette and actual_metadata == metadata


def test_schematic_validation_and_corrupt_varints(tmp_path):
    with pytest.raises(ValueError, match="outside the palette"):
        write_schematic(tmp_path / "bad.schem", np.ones((1, 1, 1), dtype=np.uint16), ["minecraft:air"])
    with pytest.raises(ValueError, match="integers"):
        write_schematic(tmp_path / "bad.schem", np.zeros((1, 1, 1)), ["minecraft:air"])
    with pytest.raises(ValueError, match="unique"):
        write_schematic(tmp_path / "bad.schem", np.zeros((1, 1, 1), dtype=np.uint16), ["minecraft:air", "minecraft:air"])
    path = write_schematic(tmp_path / "bad.schem", np.zeros((1, 1, 1), dtype=np.uint16), ["minecraft:air"])
    raw = nbtlib.load(str(path), gzipped=True)
    raw["BlockData"] = nbtlib.ByteArray(np.array([128], dtype=np.uint8).view(np.int8))
    raw.save(path)
    with pytest.raises(ValueError, match="varints"):
        read_schematic(path)


def test_world_crosses_chunk_and_region_bounds_and_preserves_source(tmp_path):
    palette = ["minecraft:air", "minecraft:stone", "minecraft:grass_block", "minecraft:oak_log[axis=y]"]
    blocks = np.zeros((9, 17, 513), dtype=np.uint16)
    blocks[:4] = 1
    blocks[4] = 2
    blocks[5:8, 8, 256] = 3
    schematic = write_schematic(tmp_path / "neighbourhood.schem", blocks, palette)
    # World export must reread the schematic, regardless of later RAM changes.
    blocks[:] = 0
    world = write_java_world(schematic, tmp_path / "world", "Test Neighbourhood")
    assert sorted(p.name for p in (world / "region").glob("*.mca")) == ["r.0.0.mca", "r.1.0.mca"]
    first = read_chunk(world, 0, 0)
    assert block_at(first, 0, 60, 0) == "minecraft:stone"
    assert block_at(first, 15, 64, 15) == "minecraft:grass_block"
    assert block_at(first, 0, 59, 0) == "minecraft:air"
    assert block_at(first, 0, 65, 0) == "minecraft:air"
    across_chunk = read_chunk(world, 1, 0)
    assert block_at(across_chunk, 0, 64, 0) == "minecraft:grass_block"
    across_region = read_chunk(world, 32, 1)
    assert block_at(across_region, 0, 64, 0) == "minecraft:grass_block"
    assert block_at(across_region, 1, 64, 0) == "minecraft:air"
    assert block_at(across_region, 0, 64, 1) == "minecraft:air"
    tree = read_chunk(world, 16, 0)
    assert block_at(tree, 0, 67, 8) == "minecraft:oak_log[axis=y]"
    # All build-height sections carry biomes. The engine must calculate light.
    assert len(first["sections"]) == 24 and int(first["isLightOn"]) == 0
    assert all(str(section["biomes"]["palette"][0]) == "minecraft:plains" for section in first["sections"])
    heights = unpack_longs(first["Heightmaps"]["WORLD_SURFACE"], 9, 256)
    assert heights == [129] * 256  # top grass 64 + 1 - minimum world Y (-64)
    edge_heights = unpack_longs(across_region["Heightmaps"]["WORLD_SURFACE"], 9, 256)
    assert edge_heights[0] == 129 and edge_heights[1:] == [0] * 255
    level = nbtlib.load(str(world / "level.dat"), gzipped=True)["Data"]
    assert str(level["LevelName"]) == "Test Neighbourhood"
    assert int(level["DataVersion"]) == 3700 and int(level["GameType"]) == 1
    assert int(level["SpawnY"]) == 65
    assert (int(level["SpawnX"]), int(level["SpawnZ"])) != (256, 8)
    assert int(level["Difficulty"]) == 0
    settings = level["WorldGenSettings"]["dimensions"]["minecraft:overworld"]["generator"]
    assert str(settings["type"]) == "minecraft:flat" and len(settings["settings"]["layers"]) == 0
    with pytest.raises(FileExistsError):
        write_java_world(schematic, world)


def test_nonzero_air_index_and_five_bit_padded_block_states(tmp_path):
    colors = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]
    palette = [f"minecraft:{color}_concrete" for color in colors] + ["minecraft:stone", "minecraft:air"]
    blocks = np.empty((16, 16, 16), dtype=np.uint16)
    blocks[:] = (np.arange(blocks.size).reshape(blocks.shape) % 17).astype(np.uint16)
    path = write_schematic(tmp_path / "palette.schem", blocks, palette)
    world = write_java_world(path, tmp_path / "world")
    chunk = read_chunk(world, 0, 0)
    section = next(section for section in chunk["sections"] if int(section["Y"]) == 4)
    assert len(section["block_states"]["palette"]) == 18
    assert len(section["block_states"]["data"]) == 342  # ceil(4096 / floor(64 / 5))
    for local_y, z, x in [(4, 0, 11), (4, 0, 12), (5, 15, 15), (15, 12, 13), (0, 0, 0)]:
        expected = palette[int(blocks[local_y, z, x])]
        assert block_at(chunk, x, WORLD_BASE_Y + local_y, z) == expected
    assert block_at(chunk, 0, 76, 0) == "minecraft:air"


def test_single_entry_schematic_gets_air_for_padding(tmp_path):
    path = write_schematic(tmp_path / "solid.schem", np.zeros((1, 1, 1), dtype=np.uint16), ["minecraft:stone"])
    world = write_java_world(path, tmp_path / "world")
    chunk = read_chunk(world, 0, 0)
    assert block_at(chunk, 0, 60, 0) == "minecraft:stone"
    assert block_at(chunk, 1, 60, 0) == "minecraft:air"


def test_void_and_overheight_worlds_fail_before_creation(tmp_path):
    path = write_schematic(tmp_path / "void.schem", np.zeros((1, 1, 1), dtype=np.uint16), ["minecraft:air"])
    with pytest.raises(ValueError, match="safe"):
        write_java_world(path, tmp_path / "void")
    assert not (tmp_path / "void").exists()
    path = write_schematic(tmp_path / "tall.schem", np.zeros((261, 1, 1), dtype=np.uint16), ["minecraft:stone"])
    with pytest.raises(ValueError, match="too tall"):
        write_java_world(path, tmp_path / "tall")
    assert not (tmp_path / "tall").exists()
