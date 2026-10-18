"""Sponge v2 schematics and Minecraft Java 1.20.4 Anvil worlds.

The intermediate schematic is the source of truth for world export. Arrays use
``(y, z, x)`` order, matching Sponge's x-fastest block ordering. No Java, Minecraft
installation, external editor, or Rust executable is needed to write a world.

References: SpongePowered/Schematic-Specification/versions/schematic-2.md;
Mojang's official 1.20.4 server ``version.json`` (world_version=3700), and its
ChunkSerializer / SimpleBitStorage implementation. Modern packed entries never
cross a 64-bit word boundary.
"""
from __future__ import annotations

import io
import json
import math
import re
import struct
import time
import uuid
import zlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import nbtlib
import numpy as np
from nbtlib import Byte, ByteArray, Compound, Double, Float, Int, IntArray, List, Long, LongArray, Short, String

DATA_VERSION = 3700
MINECRAFT_VERSION = "1.20.4"
WORLD_BASE_Y = 60
WORLD_MIN_Y = -64
WORLD_MAX_Y = 319
MAX_BLOCKS = 100_000_000
AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
_BLOCK_STATE = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+(?:\[[a-z0-9_=,.-]+\])?$")
_ENTITY_NAMESPACE = uuid.UUID("f454726c-6974-5908-b599-85c5eba2ba38")
_POSE_PARTS = {"Head", "Body", "LeftArm", "RightArm", "LeftLeg", "RightLeg"}


def scene_entities(scene: dict) -> list[dict]:
    """Convert semantic people into static Java armour stands.

    Scene yaw is a geographic bearing (north=0, east=90). Minecraft yaw faces
    south at zero and west at 90, so subtract 180. Feet stay at local base_y;
    world export applies its Y offset only after reading the schematic.
    UUIDs depend on the scene location and object identity, never the job ID.
    """
    entities = []
    location = json.dumps(scene.get("bbox", []), separators=(",", ":"))
    for index, obj in enumerate(scene.get("objects", [])):
        if obj.get("kind") != "person":
            continue
        x, z = obj["geometry"]["coordinates"]
        pos = [float(x), float(obj["base_y"]), float(z)]
        identity = str(obj.get("id", f"person/{index}"))
        entities.append({
            "id": "minecraft:armor_stand", "pos": pos,
            "yaw": (float(obj.get("heading", obj.get("yaw", 0))) % 360) - 180,
            "pitch": 0.0, "outfit": dict(obj.get("outfit", {})),
            "pose": dict(obj.get("pose", {})),
            "small": bool(obj.get("entity_small", False)),
            "uuid": str(uuid.uuid5(_ENTITY_NAMESPACE, f"{location}/{identity}/{pos}")),
        })
    return entities


def _finite_vector(value, length: int, label: str) -> list[float]:
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Entity {label} must contain {length} finite numbers.") from exc
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError(f"Entity {label} must contain {length} finite numbers.")
    return result


def _entity_uuid(entity: dict, index: int) -> uuid.UUID:
    if entity.get("uuid"):
        return uuid.UUID(str(entity["uuid"]))
    stable = json.dumps(entity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return uuid.uuid5(_ENTITY_NAMESPACE, f"{index}/{stable}")


def _armor_item(item: str, color: int) -> Compound:
    if not 0 <= int(color) <= 0xFFFFFF:
        raise ValueError("Armour colors must be RGB integers between 0 and 16777215.")
    return Compound({
        "id": String(f"minecraft:leather_{item}"), "Count": Byte(1),
        "tag": Compound({"display": Compound({"color": Int(int(color))})}),
    })


def _entity_tag(entity: dict, index: int) -> Compound:
    """Sponge v2 uses Id, Java uses id; all extra tags share the compound."""
    if entity.get("id", "minecraft:armor_stand") != "minecraft:armor_stand":
        raise ValueError("RealityForge currently creates only armour stand entities.")
    pos = _finite_vector(entity.get("pos"), 3, "position")
    rotation = _finite_vector([entity.get("yaw", 0), entity.get("pitch", 0)], 2, "rotation")
    outfit = entity.get("outfit", {})
    armor = [_armor_item("boots", outfit.get("shoes", 0x292929)),
             _armor_item("leggings", outfit.get("bottom", 0x294B70)),
             _armor_item("chestplate", outfit.get("top", 0xC95746)),
             _armor_item("helmet", outfit.get("head", outfit.get("skin", 0xB98666)))]
    poses = entity.get("pose", {})
    if any(part not in _POSE_PARTS for part in poses):
        raise ValueError("Unsupported armour stand pose part.")
    return Compound({
        "Id": String("minecraft:armor_stand"),
        "Pos": List[Double](pos), "Rotation": List[Float](rotation),
        "Motion": List[Double]([0.0, 0.0, 0.0]),
        "UUID": IntArray(struct.unpack(">4i", _entity_uuid(entity, index).bytes)),
        "ShowArms": Byte(1), "NoBasePlate": Byte(1), "NoGravity": Byte(1),
        "Invulnerable": Byte(1), "Silent": Byte(1), "Invisible": Byte(0),
        "Small": Byte(int(bool(entity.get("small", False)))), "OnGround": Byte(1), "Health": Float(20),
        "Fire": Short(-1), "Air": Short(300), "FallDistance": Float(0),
        "ArmorItems": List[Compound](armor),
        "HandItems": List[Compound]([Compound({}), Compound({})]),
        "Pose": Compound({part: List[Float](_finite_vector(angles, 3, "pose")) for part, angles in sorted(poses.items())}),
        "Tags": List[String](["realityforge", "pedestrian"]),
    })


def _entity_descriptor(tag: Compound) -> dict:
    """Read semantic descriptors from real entity NBT, never metadata JSON."""
    entity = {"id": str(tag.get("Id", tag.get("id", ""))),
              "pos": _finite_vector(tag.get("Pos"), 3, "position")}
    rotation = _finite_vector(tag.get("Rotation", [0, 0]), 2, "rotation")
    entity.update(yaw=rotation[0], pitch=rotation[1])
    if "UUID" in tag:
        parts = [int(v) for v in tag["UUID"]]
        if len(parts) != 4:
            raise ValueError("Entity UUID must contain four integers.")
        entity["uuid"] = str(uuid.UUID(bytes=struct.pack(">4i", *parts)))
    if entity["id"] == "minecraft:armor_stand":
        outfit = {}
        for key, item in zip(("shoes", "bottom", "top", "head"), tag.get("ArmorItems", [])):
            color = item.get("tag", {}).get("display", {}).get("color")
            if color is not None:
                outfit[key] = int(color)
        entity["outfit"] = outfit
        entity["pose"] = {part: _finite_vector(angles, 3, "pose") for part, angles in tag.get("Pose", {}).items()}
    return entity


def _validate_blocks(blocks: np.ndarray, palette: list[str]) -> None:
    if not isinstance(blocks, np.ndarray) or blocks.ndim != 3:
        raise ValueError("Blocks must be a numpy array in (Y, Z, X) order.")
    if blocks.dtype.kind not in "ui":
        raise ValueError("Block indices must be integers.")
    if any(size < 1 or size > 65535 for size in blocks.shape):
        raise ValueError("Schematic dimensions must each be between 1 and 65535.")
    if blocks.size > MAX_BLOCKS:
        raise ValueError(f"Schematic exceeds the {MAX_BLOCKS:,} block memory limit.")
    if not palette or len(palette) > 65536 or len(set(palette)) != len(palette):
        raise ValueError("Palette must contain 1–65536 unique block states.")
    if any(not isinstance(state, str) or not _BLOCK_STATE.fullmatch(state) for state in palette):
        raise ValueError("Palette entries must be namespaced Minecraft block states.")
    if int(blocks.min()) < 0 or int(blocks.max()) >= len(palette):
        raise ValueError("A block index is outside the palette.")


def _encode_varints(values: np.ndarray) -> np.ndarray:
    """Encode uint16 IDs, using bounded temporary arrays for large schematics."""
    flat = np.asarray(values, dtype=np.uint16).reshape(-1)
    if int(flat.max(initial=0)) < 128:
        return flat.astype(np.uint8).view(np.int8)
    result = bytearray()
    for start in range(0, flat.size, 1_000_000):
        batch = flat[start:start + 1_000_000]
        lengths = 1 + (batch >= 128).astype(np.int8) + (batch >= 16384).astype(np.int8)
        ends = np.cumsum(lengths, dtype=np.int64)
        starts = ends - lengths
        encoded = np.empty(int(ends[-1]), dtype=np.uint8)
        encoded[starts] = (batch & 127) | ((lengths > 1).astype(np.uint16) << 7)
        more = lengths > 1
        encoded[starts[more] + 1] = ((batch[more] >> 7) & 127) | ((lengths[more] > 2).astype(np.uint16) << 7)
        more = lengths > 2
        encoded[starts[more] + 2] = batch[more] >> 14
        result.extend(encoded.tobytes())
    return np.frombuffer(result, dtype=np.int8)


def _decode_varints(data: np.ndarray, count: int) -> np.ndarray:
    raw = np.asarray(data, dtype=np.int8).view(np.uint8)
    if raw.size == count and not np.any(raw & 128):
        return raw.astype(np.uint16)
    # Decode in bounded batches: one byte ends each varint, at most three bytes
    # represent a uint16 palette ID. Malformed files fail before allocation.
    if raw.size < count or raw.size > count * 3 or raw.size == 0 or raw[-1] & 128:
        raise ValueError("Schematic contains truncated or invalid block varints.")
    output = np.empty(count, dtype=np.uint16)
    input_pos = output_pos = 0
    while input_pos < raw.size:
        stop = min(input_pos + 2_000_000, raw.size)
        while stop < raw.size and raw[stop - 1] & 128:
            stop += 1
        batch = raw[input_pos:stop]
        ends = np.flatnonzero(batch < 128)
        starts = np.empty_like(ends)
        starts[0] = 0
        starts[1:] = ends[:-1] + 1
        lengths = ends - starts + 1
        if np.any(lengths > 3) or output_pos + ends.size > count:
            raise ValueError("Schematic contains invalid block varints.")
        decoded = (batch[starts] & 127).astype(np.uint32)
        more = lengths > 1
        decoded[more] |= (batch[starts[more] + 1] & 127).astype(np.uint32) << 7
        more = lengths > 2
        decoded[more] |= batch[starts[more] + 2].astype(np.uint32) << 14
        if np.any(decoded > 65535):
            raise ValueError("Schematic block index exceeds the uint16 palette limit.")
        output[output_pos:output_pos + ends.size] = decoded
        output_pos += ends.size
        input_pos = stop
    if output_pos != count:
        raise ValueError("Schematic dimensions do not match its block data.")
    return output


def _unsigned_short(value: int) -> Short:
    return Short(value if value <= 32767 else value - 65536)


def write_schematic(path: Path, blocks: np.ndarray, palette: list[str], metadata: dict | None = None,
                    entities: list[dict] | None = None) -> Path:
    """Write Sponge v2 blocks and optional real entities, in local coordinates."""
    _validate_blocks(blocks, palette)
    height, length, width = blocks.shape
    metadata = dict(metadata or {})
    # Entity state belongs in the actual Entities list, never just a JSON blob.
    metadata.pop("entities", None)
    entity_tags = [_entity_tag(entity, i) for i, entity in enumerate(entities or [])]
    _validate_entity_tags(entity_tags, width, length, height)
    root = nbtlib.File({
        "Version": Int(2),
        "DataVersion": Int(DATA_VERSION),
        "Width": _unsigned_short(width),
        "Height": _unsigned_short(height),
        "Length": _unsigned_short(length),
        "Offset": IntArray([0, 0, 0]),
        "PaletteMax": Int(len(palette)),
        "Palette": Compound({state: Int(index) for index, state in enumerate(palette)}),
        "BlockData": ByteArray(_encode_varints(blocks)),
        "BlockEntities": List[Compound]([]),
        "Entities": List[Compound](entity_tags),
        "Metadata": Compound({
            "Name": String(str(metadata.get("name", metadata.get("Name", "RealityForge")))),
            "Author": String("RealityForge"),
            "Date": Long(int(time.time() * 1000)),
            "RealityForge": String(json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))),
        }),
    }, root_name="Schematic", gzipped=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root.save(path)
    return path


def read_schematic(path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Read blocks and metadata; actual entities appear in metadata['entities'].

    Empty entity lists keep the original three-tuple metadata contract intact.
    """
    root = nbtlib.load(str(path), gzipped=True)
    return _read_schematic_root(root)


def _read_schematic_root(root: nbtlib.File) -> tuple[np.ndarray, list[str], dict]:
    if int(root.get("Version", 0)) != 2:
        raise ValueError("RealityForge reads Sponge v2 schematics.")
    shape = tuple(int(root[name]) & 65535 for name in ("Height", "Length", "Width"))
    count = math.prod(shape)
    if min(shape) < 1 or count > MAX_BLOCKS:
        raise ValueError("Invalid or oversized schematic dimensions.")
    mapping = root.get("Palette", {})
    if not mapping or len(mapping) > 65536:
        raise ValueError("Schematic needs a local palette of at most 65536 entries.")
    ids = [int(value) for value in mapping.values()]
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("Schematic palette IDs must be contiguous starting at zero.")
    palette = [""] * len(mapping)
    for state, index in mapping.items():
        palette[int(index)] = str(state)
    blocks = _decode_varints(root["BlockData"], count).reshape(shape)
    _validate_blocks(blocks, palette)
    meta = root.get("Metadata", {})
    metadata = json.loads(str(meta["RealityForge"])) if "RealityForge" in meta else {key: value.unpack() for key, value in meta.items()}
    if not isinstance(metadata, dict):
        raise ValueError("RealityForge schematic metadata must be an object.")
    metadata.pop("entities", None)
    entities = [_entity_descriptor(entity) for entity in root.get("Entities", [])]
    if entities:
        metadata["entities"] = entities
    return blocks, palette, metadata


def _validate_entity_tags(entities, width: int, length: int, height: int) -> None:
    seen = set()
    for entity in entities:
        if not re.fullmatch(r"[a-z0-9_.-]+:[a-z0-9_./-]+", str(entity.get("Id", entity.get("id", "")))):
            raise ValueError("Entity Id must be a namespaced Minecraft entity type.")
        x, y, z = _finite_vector(entity.get("Pos"), 3, "position")
        # Feet may sit exactly atop the final block layer; entity bodies are not
        # voxel data and therefore need not fit inside the block array height.
        if not (0 <= x < width and 0 <= z < length and 0 <= y <= height):
            raise ValueError("Entity position is outside the schematic bounds.")
        identity = tuple(int(v) for v in entity.get("UUID", []))
        if identity:
            if identity in seen:
                raise ValueError("Entity UUIDs must be unique within a schematic.")
            seen.add(identity)


def _pack_longs(values: np.ndarray, bits: int) -> LongArray:
    """Pack non-overlapping little-endian entries inside big-endian NBT longs."""
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    entries_per_long = 64 // bits
    padded = np.zeros(math.ceil(values.size / entries_per_long) * entries_per_long, dtype=np.uint64)
    padded[:values.size] = values
    matrix = padded.reshape(-1, entries_per_long)
    shifts = np.arange(entries_per_long, dtype=np.uint64) * bits
    words = np.bitwise_or.reduce(matrix << shifts, axis=1)
    # NBT longs are signed; reinterpret the bits rather than overflowing casts.
    return LongArray(words.view(np.int64))


def _state_tag(state: str) -> Compound:
    name, separator, properties = state.partition("[")
    tag = Compound({"Name": String(name)})
    if separator:
        pairs = [item.split("=", 1) for item in properties.rstrip("]").split(",")]
        if any(len(pair) != 2 or not all(pair) for pair in pairs):
            raise ValueError(f"Invalid block properties: {state}")
        tag["Properties"] = Compound({key: String(value) for key, value in pairs})
    return tag


def _section(blocks: np.ndarray, y: int, state_tags: list[Compound]) -> Compound:
    unique, inverse = np.unique(blocks, return_inverse=True)
    states = Compound({"palette": List[Compound]([state_tags[int(index)] for index in unique])})
    if unique.size > 1:
        states["data"] = _pack_longs(inverse, max(4, (int(unique.size) - 1).bit_length()))
    return Compound({
        "Y": Byte(y),
        "block_states": states,
        "biomes": Compound({"palette": List[String]([String("minecraft:plains")])}),
    })


def _column_top(blocks: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    matches = accepted[blocks]
    height = blocks.shape[0]
    # Returns local top occupied Y, or -1 in an empty column.
    top = height - 1 - np.argmax(matches[::-1], axis=0)
    return np.where(np.any(matches, axis=0), top, -1)


def _spawn(blocks: np.ndarray, palette: list[str], air_ids: set[int]) -> tuple[int, int, int]:
    """Prefer open ground near the center and reject water, foliage and roofs."""
    height, length, width = blocks.shape
    air_lookup = np.array([index in air_ids for index in range(len(palette))])
    top = _column_top(blocks, ~air_lookup)
    z, x = np.indices((length, width))
    top_ids = blocks[np.maximum(top, 0), z, x]
    names = [state.split("[", 1)[0] for state in palette]
    preferred = np.array([name in {"minecraft:grass_block", "minecraft:dirt", "minecraft:coarse_dirt", "minecraft:stone", "minecraft:gravel", "minecraft:sand", "minecraft:gray_concrete", "minecraft:black_concrete", "minecraft:light_gray_concrete", "minecraft:stone_bricks", "minecraft:cobblestone", "minecraft:smooth_stone"} for name in names])
    safe = np.array([name not in AIR_NAMES and not any(part in name for part in ("water", "lava", "leaves", "log", "fence", "wall", "pane", "glass", "slab", "stairs", "flower", "grass", "bush", "carpet", "torch")) or name == "minecraft:grass_block" for name in names])
    candidates = (top >= 0) & (WORLD_BASE_Y + top + 2 <= WORLD_MAX_Y) & safe[top_ids]
    if not np.any(candidates):
        raise ValueError("World needs at least one safe, solid surface to spawn on.")
    score = (x - (width - 1) / 2) ** 2 + (z - (length - 1) / 2) ** 2
    score += (~preferred[top_ids]) * (width * width + length * length)
    score[~candidates] = np.inf
    spawn_z, spawn_x = np.unravel_index(np.argmin(score), score.shape)
    return int(spawn_x), WORLD_BASE_Y + int(top[spawn_z, spawn_x]) + 1, int(spawn_z)


def _chunk(cx: int, cz: int, blocks: np.ndarray, palette: list[str], state_tags: list[Compound], air_id: int) -> bytes:
    height, length, width = blocks.shape
    x0, z0 = cx * 16, cz * 16
    sx, sz = min(16, width - x0), min(16, length - z0)
    sections = []
    air = np.full((16, 16, 16), air_id, dtype=np.uint16)
    for section_y in range(WORLD_MIN_Y // 16, (WORLD_MAX_Y + 1) // 16):
        source_y0 = max(0, section_y * 16 - WORLD_BASE_Y)
        source_y1 = min(height, (section_y + 1) * 16 - WORLD_BASE_Y)
        section_blocks = air
        if source_y1 > source_y0:
            section_blocks = air.copy()
            local_y = WORLD_BASE_Y + source_y0 - section_y * 16
            section_blocks[local_y:local_y + source_y1 - source_y0, :sz, :sx] = blocks[source_y0:source_y1, z0:z0 + sz, x0:x0 + sx]
        sections.append(_section(section_blocks, section_y, state_tags))
    non_air = np.array([state.split("[", 1)[0] not in AIR_NAMES for state in palette])
    no_leaves = np.array([valid and "leaves" not in state for valid, state in zip(non_air, palette)])
    no_fluid = np.array([valid and not state.startswith(("minecraft:water", "minecraft:lava")) for valid, state in zip(non_air, palette)])
    maps = {}
    source = blocks[:, z0:z0 + sz, x0:x0 + sx]
    for name, predicate in (("WORLD_SURFACE", non_air), ("OCEAN_FLOOR", no_fluid), ("MOTION_BLOCKING", non_air), ("MOTION_BLOCKING_NO_LEAVES", no_leaves)):
        tops = _column_top(source, predicate)
        values = np.zeros((16, 16), dtype=np.uint16)
        values[:sz, :sx] = np.where(tops >= 0, tops + WORLD_BASE_Y + 1 - WORLD_MIN_Y, 0)
        maps[name] = _pack_longs(values, 9)
    root = nbtlib.File({
        "DataVersion": Int(DATA_VERSION), "xPos": Int(cx), "yPos": Int(-4), "zPos": Int(cz),
        "Status": String("minecraft:full"), "LastUpdate": Long(0), "InhabitedTime": Long(0),
        # Minecraft computes skylight and emissive block light when these chunks
        # are first loaded; invented light arrays cause dark seams and interiors.
        "isLightOn": Byte(0), "sections": List[Compound](sections),
        "block_entities": List[Compound]([]), "block_ticks": List[Compound]([]), "fluid_ticks": List[Compound]([]),
        "PostProcessing": List[List[Short]]([List[Short]([]) for _ in range(24)]),
        "Heightmaps": Compound(maps), "structures": Compound({"starts": Compound({}), "References": Compound({})}),
    })
    stream = io.BytesIO()
    root.write(stream)
    return zlib.compress(stream.getvalue(), level=6)


def _level_data(world_name: str, spawn: tuple[int, int, int], width: int, length: int) -> nbtlib.File:
    dimensions = Compound({
        "minecraft:overworld": Compound({
            "type": String("minecraft:overworld"),
            "generator": Compound({"type": String("minecraft:flat"), "settings": Compound({
                "biome": String("minecraft:plains"), "features": Byte(0), "lakes": Byte(0),
                "layers": List[Compound]([]), "structure_overrides": List[String]([]),
            })}),
        }),
    })
    for name, setting, biome_preset in (("the_nether", "nether", "nether"), ("the_end", "end", "end")):
        biome_source = Compound({"type": String("minecraft:the_end")}) if name == "the_end" else Compound({"type": String("minecraft:multi_noise"), "preset": String("minecraft:nether")})
        dimensions[f"minecraft:{name}"] = Compound({
            "type": String(f"minecraft:{name}"),
            "generator": Compound({"type": String("minecraft:noise"), "settings": String(f"minecraft:{setting}"), "biome_source": biome_source}),
        })
    border = float(max(width, length) + 32)
    data = Compound({
        "DataVersion": Int(DATA_VERSION), "version": Int(19133),
        "Version": Compound({"Id": Int(DATA_VERSION), "Name": String(MINECRAFT_VERSION), "Snapshot": Byte(0), "Series": String("main")}),
        "LevelName": String(world_name), "GameType": Int(1), "allowCommands": Byte(1), "hardcore": Byte(0),
        "Difficulty": Byte(0), "DifficultyLocked": Byte(1), "initialized": Byte(1), "MapFeatures": Byte(0),
        "SpawnX": Int(spawn[0]), "SpawnY": Int(spawn[1]), "SpawnZ": Int(spawn[2]), "SpawnAngle": Float(0),
        "Time": Long(0), "DayTime": Long(6000), "LastPlayed": Long(int(time.time() * 1000)),
        "raining": Byte(0), "thundering": Byte(0), "clearWeatherTime": Int(2_000_000_000), "rainTime": Int(0), "thunderTime": Int(0),
        "GameRules": Compound({name: String(value) for name, value in {
            "doDaylightCycle": "false", "doWeatherCycle": "false", "doMobSpawning": "false", "doFireTick": "false",
            "randomTickSpeed": "0", "spawnRadius": "0", "keepInventory": "true", "mobGriefing": "false",
        }.items()}),
        "WorldGenSettings": Compound({"seed": Long(0), "generate_features": Byte(0), "bonus_chest": Byte(0), "dimensions": dimensions}),
        "DataPacks": Compound({"Enabled": List[String]([String("vanilla")]), "Disabled": List[String]([])}),
        "enabled_features": List[String]([String("minecraft:vanilla")]),
        "BorderCenterX": Double(width / 2), "BorderCenterZ": Double(length / 2), "BorderSize": Double(border),
        "BorderSizeLerpTarget": Double(border), "BorderSizeLerpTime": Long(0),
        "BorderSafeZone": Double(5), "BorderDamagePerBlock": Double(0), "BorderWarningBlocks": Double(0), "BorderWarningTime": Double(0),
        "WasModded": Byte(0), "ServerBrands": List[String]([String("vanilla")]),
    })
    return nbtlib.File({"Data": data}, gzipped=True)


def _write_entity_regions(source_entities, output_dir: Path) -> None:
    """Java 1.20.4 stores entities in separate Anvil regions, not block chunks.

    Each populated entity chunk is {DataVersion, Position:[I;cx,cz], Entities}.
    Keep every vanilla extra field from the source schematic NBT intact.
    """
    if not source_entities:
        return
    regions = {}
    for index, original in enumerate(source_entities):
        entity = deepcopy(original)
        identifier = str(entity.pop("Id", entity.get("id", "")))
        if not identifier:
            raise ValueError("Schematic entity is missing its Id.")
        entity["id"] = String(identifier)
        x, y, z = _finite_vector(entity["Pos"], 3, "position")
        entity["Pos"] = List[Double]([x, y + WORLD_BASE_Y, z])
        if "UUID" not in entity:
            identity = _entity_uuid(_entity_descriptor(original), index)
            entity["UUID"] = IntArray(struct.unpack(">4i", identity.bytes))
        cx, cz = math.floor(x / 16), math.floor(z / 16)
        regions.setdefault((cx // 32, cz // 32), {}).setdefault((cx, cz), []).append(entity)
    entity_dir = output_dir / "entities"
    entity_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    for (rx, rz), chunks in sorted(regions.items()):
        header, sector = bytearray(8192), 2
        with (entity_dir / f"r.{rx}.{rz}.mca").open("wb") as region:
            region.write(header)
            for (cx, cz), entities in sorted(chunks.items()):
                root = nbtlib.File({
                    "DataVersion": Int(DATA_VERSION), "Position": IntArray([cx, cz]),
                    "Entities": List[Compound](entities),
                })
                stream = io.BytesIO()
                root.write(stream)
                compressed = zlib.compress(stream.getvalue(), level=6)
                payload = struct.pack(">I", len(compressed) + 1) + b"\x02" + compressed
                sectors = math.ceil(len(payload) / 4096)
                if sectors > 255:
                    raise ValueError("An entity chunk exceeds the Anvil region allocation limit.")
                slot = (cx % 32) + (cz % 32) * 32
                struct.pack_into(">I", header, slot * 4, (sector << 8) | sectors)
                struct.pack_into(">I", header, 4096 + slot * 4, timestamp)
                region.write(payload)
                region.write(b"\0" * (sectors * 4096 - len(payload)))
                sector += sectors
            region.seek(0)
            region.write(header)


def write_java_world(schematic_path: Path, output_dir: Path, world_name: str = "RealityForge") -> Path:
    """Read the actual schematic and write a ready-to-import Java world folder.

    The provided ``output_dir`` itself is the world folder. Existing Minecraft
    world files are never overwritten. Schematic local Y=0 becomes world Y=60;
    generated terrain with its grass at local Y=4 therefore sits at world Y=64.
    """
    root = nbtlib.load(str(schematic_path), gzipped=True)
    blocks, palette, metadata = _read_schematic_root(root)
    source_entities = root.get("Entities", [])
    del root  # Release compressed-block NBT arrays before allocating chunks.
    height, length, width = blocks.shape
    _validate_entity_tags(source_entities, width, length, height)
    if blocks.shape[0] + WORLD_BASE_Y > WORLD_MAX_Y + 1:
        raise ValueError("This schematic is too tall for the Java world build limit.")
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in ("level.dat", "region", "entities")):
        raise FileExistsError("The output directory already contains a Minecraft world.")
    if "minecraft:air" not in palette:
        if len(palette) == 65536:
            raise ValueError("The world palette needs room for minecraft:air.")
        palette.append("minecraft:air")
    air_id = palette.index("minecraft:air")
    air_ids = {index for index, state in enumerate(palette) if state in AIR_NAMES}
    spawn = _spawn(blocks, palette, air_ids)
    state_tags = [_state_tag(state) for state in palette]
    _, length, width = blocks.shape
    region_dir = output_dir / "region"
    region_dir.mkdir(parents=True, exist_ok=True)
    chunks_x, chunks_z = math.ceil(width / 16), math.ceil(length / 16)
    timestamp = int(time.time())
    for rz in range(math.ceil(chunks_z / 32)):
        for rx in range(math.ceil(chunks_x / 32)):
            header = bytearray(8192)
            with (region_dir / f"r.{rx}.{rz}.mca").open("wb") as region:
                region.write(header)
                sector = 2
                for cz in range(rz * 32, min(chunks_z, (rz + 1) * 32)):
                    for cx in range(rx * 32, min(chunks_x, (rx + 1) * 32)):
                        compressed = _chunk(cx, cz, blocks, palette, state_tags, air_id)
                        payload = struct.pack(">I", len(compressed) + 1) + b"\x02" + compressed
                        sectors = math.ceil(len(payload) / 4096)
                        if sectors > 255:
                            raise ValueError("A chunk exceeds the Anvil region allocation limit.")
                        index = (cx % 32) + (cz % 32) * 32
                        struct.pack_into(">I", header, index * 4, (sector << 8) | sectors)
                        struct.pack_into(">I", header, 4096 + index * 4, timestamp)
                        region.write(payload)
                        region.write(b"\0" * (sectors * 4096 - len(payload)))
                        sector += sectors
                region.seek(0)
                region.write(header)
    _write_entity_regions(source_entities, output_dir)
    _level_data(world_name, spawn, width, length).save(output_dir / "level.dat")
    return output_dir
