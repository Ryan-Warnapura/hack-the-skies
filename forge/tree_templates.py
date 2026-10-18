"""Optional local tree schematics -> bounded canonical asset parts.

Sponge v2/v3 references are read locally, never copied into distributable source.
Default config is data/tree_templates.json (gitignored), or the path in
REALITYFORGE_TREE_TEMPLATES. Example config:
{"version": 1, "templates": [{"path": "C:/trees.zip", "member": "oak.schem",
  "allowed_species": ["oak", "maple", "dark_oak"]}]}

These are artistic silhouettes, not evidence of a real tree species. Unsupported
blocks and display bases are excluded. Source metadata and absence of a supplied
license are reported, never interpreted as permission to redistribute an asset.
Format: https://github.com/SpongePowered/Schematic-Specification
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import nbtlib
import numpy as np

from .minecraft import _decode_varints

MAX_SOURCE_BLOCKS = 16_000_000
MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_NBT_BYTES = 96 * 1024 * 1024
MAX_TEMPLATE_PARTS = 4096
MAX_TARGET_DIMENSION = 64
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'data' / 'tree_templates.json'
WOODS = ('oak', 'spruce', 'birch', 'jungle', 'acacia', 'dark_oak', 'mangrove', 'cherry')
SPECIES = {'oak', 'maple', 'birch', 'spruce', 'pine', 'dark_oak', 'bare', 'cherry', 'shrub'}
AIR = {'minecraft:air', 'minecraft:cave_air', 'minecraft:void_air'}


def _canonical_material(state):
    """Whitelist Java 1.20.4 tree blocks; preserve visual block-state properties."""
    match = re.fullmatch(r'minecraft:([a-z0-9_]+)(?:\[([a-z0-9_=,]+)\])?', state)
    if not match:
        return None
    name, values = match.groups()
    if 'minecraft:' + name in AIR:
        return 'minecraft:air'
    properties = dict(item.split('=', 1) for item in (values or '').split(',') if '=' in item)
    family = None
    for wood in WOODS:
        if name in (wood + '_log', wood + '_wood', 'stripped_' + wood + '_log', 'stripped_' + wood + '_wood'):
            family = 'wood'
        elif name == wood + '_leaves':
            family = 'leaves'
        elif name == wood + '_fence':
            family = 'fence'
        elif name == wood + '_slab':
            family = 'slab'
        elif name == wood + '_stairs':
            family = 'stairs'
    if name in {'azalea_leaves', 'flowering_azalea_leaves'}:
        family = 'leaves'
    if name == 'moss_block':
        family = 'solid'
    if family is None:
        return None
    if family == 'wood':
        properties = {'axis': properties.get('axis', 'y') if properties.get('axis', 'y') in {'x', 'y', 'z'} else 'y'}
    elif family == 'leaves':
        # Distance has no visual effect. Persistent leaves avoid decay after paste.
        properties = {'distance': '7', 'persistent': 'true', 'waterlogged': 'false'}
    elif family == 'fence':
        properties = {direction: 'true' if properties.get(direction) == 'true' else 'false'
                      for direction in ('east', 'north', 'south', 'west')}
        properties['waterlogged'] = 'false'
    elif family == 'slab':
        properties = {'type': properties.get('type', 'bottom') if properties.get('type', 'bottom') in {'top', 'bottom', 'double'} else 'bottom',
                      'waterlogged': 'false'}
    elif family == 'stairs':
        properties = {'facing': properties.get('facing', 'north') if properties.get('facing', 'north') in {'north', 'south', 'east', 'west'} else 'north',
                      'half': 'top' if properties.get('half') == 'top' else 'bottom',
                      'shape': properties.get('shape', 'straight') if properties.get('shape', 'straight') in {'straight', 'inner_left', 'inner_right', 'outer_left', 'outer_right'} else 'straight',
                      'waterlogged': 'false'}
    else:
        properties = {}
    return 'minecraft:' + name + ('[' + ','.join(f'{key}={value}' for key, value in sorted(properties.items())) + ']' if properties else '')


def _source_signature(path: Path):
    path = path.resolve()
    stat = path.stat()
    if stat.st_size > MAX_COMPRESSED_BYTES:
        raise ValueError('Tree source exceeds the 16 MB compressed-file limit.')
    return str(path), stat.st_mtime_ns, stat.st_size


def _source_bytes(path: str, member: str | None):
    license_present = False
    if member is not None:
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist()) > 500:
                raise ValueError('Tree archive contains too many entries.')
            info = archive.getinfo(member)
            if info.file_size > MAX_COMPRESSED_BYTES or not member.lower().endswith('.schem'):
                raise ValueError('Select a .schem member no larger than 16 MB.')
            raw = archive.read(info)
            license_present = any(re.search(r'(^|/)(licen[cs]e|copying)(\.|$)', name, re.I) for name in archive.namelist())
    else:
        raw = Path(path).read_bytes()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            data = stream.read(MAX_NBT_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise ValueError('Tree reference is not a readable gzip Sponge schematic.') from exc
    if len(data) > MAX_NBT_BYTES:
        raise ValueError('Tree reference expands beyond the 96 MB NBT limit.')
    return raw, data, license_present


@lru_cache(maxsize=6)
def _read_reference(path: str, modified: int, size: int, member: str | None):
    raw, data, license_present = _source_bytes(path, member)
    root = nbtlib.File.parse(io.BytesIO(data))
    root = root.get('Schematic', root)
    version = int(root.get('Version', 0))
    if version not in (2, 3):
        raise ValueError('Tree references must use Sponge schematic v2 or v3.')
    shape = tuple(int(root.get(name, 0)) & 65535 for name in ('Height', 'Length', 'Width'))
    if min(shape) < 1 or max(shape) > 2048 or math.prod(shape) > MAX_SOURCE_BLOCKS:
        raise ValueError('Tree reference has invalid or oversized dimensions.')
    container = root.get('Blocks', {}) if version == 3 else root
    mapping = container.get('Palette', {})
    if not mapping or len(mapping) > 2048:
        raise ValueError('Tree reference must have 1..2048 block states.')
    ids = [int(v) for v in mapping.values()]
    if min(ids) < 0 or max(ids) > 65535 or len(set(ids)) != len(ids):
        raise ValueError('Invalid or repeated tree palette indices.')
    decoded = _decode_varints(container.get('Data' if version == 3 else 'BlockData', []), math.prod(shape)).reshape(shape)
    valid = np.zeros(65536, dtype=bool)
    valid[ids] = True
    if not valid[decoded].all():
        raise ValueError('Tree blocks reference a missing palette entry.')
    palette = ['minecraft:air']
    remap = np.zeros(65536, dtype=np.uint16)
    ignored, source_states = {}, {}
    counts = np.bincount(decoded.ravel(), minlength=max(ids) + 1)
    for state, old_id in mapping.items():
        state, old_id = str(state), int(old_id)
        source_states[state] = int(counts[old_id])
        material = _canonical_material(state)
        if material is None:
            ignored[state] = int(counts[old_id])
            continue
        if material not in palette:
            palette.append(material)
        remap[old_id] = palette.index(material)
    blocks = remap[decoded]
    occupied = blocks != 0
    if not occupied.any():
        raise ValueError('The schematic contains no supported Java tree blocks.')
    ys, zs, xs = np.nonzero(occupied)
    crop = [int(xs.min()), int(ys.min()), int(zs.min()), int(xs.max()) + 1, int(ys.max()) + 1, int(zs.max()) + 1]
    blocks = blocks[crop[1]:crop[4], crop[2]:crop[5], crop[0]:crop[3]].copy()
    blocks.flags.writeable = False
    metadata = root.get('Metadata', {})
    source_nonair = sum(count for state, count in source_states.items() if state.split('[')[0] not in AIR)
    leaf_species = {wood: sum(count for state, count in source_states.items() if state.startswith('minecraft:' + wood + '_leaves')) for wood in WOODS}
    inferred = max(leaf_species, key=leaf_species.get)
    inferred = inferred if leaf_species[inferred] else 'generic'
    collection = max(blocks.shape[1:]) > blocks.shape[0] * 2.2 or (blocks.shape[1] > blocks.shape[0] * 1.5 and blocks.shape[2] > blocks.shape[0] * 1.5)
    report = {
        'filename': Path(path).name, 'member': member, 'sha256': hashlib.sha256(raw).hexdigest(),
        'schematic_version': version, 'data_version': int(root.get('DataVersion', 0)),
        'source_dimensions': {'width': shape[2], 'height': shape[0], 'depth': shape[1]},
        'trimmed_dimensions': {'width': blocks.shape[2], 'height': blocks.shape[0], 'depth': blocks.shape[1]},
        'source_nonair_blocks': source_nonair, 'retained_tree_blocks': int(np.count_nonzero(blocks)),
        'source_palette_states': len(mapping), 'normalized_palette_states': len(palette),
        'ignored_blocks': sum(ignored.values()), 'ignored_states': ignored, 'trim_box': crop,
        'author': str(metadata.get('Author', 'Unknown')), 'name': str(metadata.get('Name', Path(member or path).stem)),
        'license_status': 'license_file_present_not_reviewed' if license_present else 'not_found',
        'material_inferred_species': inferred, 'species_verified': False,
        'collection_candidate': bool(collection), 'source_units': 'Minecraft blocks; not surveyed physical metres',
        'compatibility': 'Only a Java 1.20.4 tree-block allowlist is retained; game loading is unverified.',
        'ignored_entities': len(root.get('Entities', [])), 'ignored_block_entities': len(container.get('BlockEntities', [])),
    }
    return blocks, palette, report


def inspect_reference(path: Path | str, member: str | None = None) -> dict:
    """Read dimensions, material counts and provenance without writing an asset."""
    return dict(_read_reference(*_source_signature(Path(path)), member)[2])


def _resize(blocks, palette, target_height):
    shape = blocks.shape
    dimensions = tuple(max(1, round(length * target_height / shape[0])) for length in shape)
    if max(dimensions) > MAX_TARGET_DIMENSION or math.prod(dimensions) > 128_000:
        raise ValueError('Scaled tree exceeds the bounded 64-block asset dimensions.')
    if target_height >= shape[0]:
        indices = [np.minimum(length - 1, ((np.arange(out) + .5) * length / out).astype(int)) for length, out in zip(shape, dimensions)]
        return blocks[np.ix_(*indices)].copy()
    y, z, x = np.nonzero(blocks)
    destination = ((y * dimensions[0] // shape[0]) * dimensions[1] + z * dimensions[1] // shape[1]) * dimensions[2] + x * dimensions[2] // shape[2]
    states = blocks[y, z, x]
    total = math.prod(dimensions)
    result = np.zeros(total, dtype=np.uint16)
    best = np.zeros(total, dtype=np.int32)
    rank = np.zeros(total, dtype=np.uint8)
    cell_volume = math.prod(shape) / total
    for index, material in enumerate(palette[1:], 1):
        counts = np.bincount(destination[states == index], minlength=total)
        structural = any(name in material for name in ('_log', '_wood', '_fence', '_slab', '_stairs'))
        priority = 2 if structural else 1
        # Preserve thin branches while requiring meaningful foliage coverage.
        minimum = 1 if structural else max(1, math.ceil(cell_volume * .08))
        selected = (counts >= minimum) & ((priority > rank) | ((priority == rank) & (counts > best)))
        result[selected], best[selected], rank[selected] = index, counts[selected], priority
    return result.reshape(dimensions)


def _merged_parts(blocks, palette):
    """Greedy identical-material cuboids, avoiding one primitive per source voxel."""
    remaining = blocks.copy()
    height, depth, width = blocks.shape
    parts = []
    for y in range(height):
        for z in range(depth):
            x = 0
            while x < width:
                value = int(remaining[y, z, x])
                if not value:
                    x += 1
                    continue
                material = palette[value]
                right = x + 1
                while right < width and remaining[y, z, right] == value:
                    right += 1
                back = z + 1
                while back < depth and np.all(remaining[y, back, x:right] == value):
                    back += 1
                top = y + 1
                half_slab = '_slab[' in material and 'type=double' not in material
                if not half_slab:
                    while top < height and np.all(remaining[top, z:back, x:right] == value):
                        top += 1
                remaining[y:top, z:back, x:right] = 0
                cy, sy = (y + top) / 2, top - y
                if half_slab:
                    cy = y + (.75 if 'type=top' in material else .25)
                    sy = .5
                parts.append({'shape': 'box', 'center': [(x + right - width) / 2, cy, (z + back - depth) / 2],
                              'size': [right - x, sy, back - z], 'material': material})
                if len(parts) > MAX_TEMPLATE_PARTS:
                    raise ValueError('Tree requires too many geometry parts at this resolution.')
                x = right
    return parts


@lru_cache(maxsize=64)
def _prepare(path, modified, size, member, target_height):
    blocks, palette, source = _read_reference(path, modified, size, member)
    if source['collection_candidate']:
        raise ValueError('Reference appears to contain a tree collection; select an individual tree schematic.')
    scaled = _resize(blocks, palette, target_height)
    if not np.any(scaled):
        raise ValueError('No tree geometry remained at the requested height.')
    parts = _merged_parts(scaled, palette)
    return {'parts': parts, 'height': scaled.shape[0], 'width': scaled.shape[2], 'depth': scaled.shape[1],
            'source': source, 'target_blocks': int(np.count_nonzero(scaled)), 'part_count': len(parts),
            'scale_factor': round(target_height / blocks.shape[0], 6)}


def prepare_template(path: Path | str, target_height=12, member: str | None = None) -> dict:
    """Build local centred parts at an explicit world-scale height (3..32 blocks)."""
    if not math.isfinite(target_height) or not 3 <= target_height <= 32:
        raise ValueError('Tree target height must be between 3 and 32 blocks.')
    return _prepare(*_source_signature(Path(path)), member, round(target_height))


@lru_cache(maxsize=4)
def _read_config(path, modified, size):
    if size > 128 * 1024:
        raise ValueError('Tree template configuration exceeds 128 KB.')
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('version') != 1 or not isinstance(config.get('templates'), list) or len(config['templates']) > 12:
        raise ValueError('Tree config must contain version 1 and at most 12 templates.')
    return config['templates']


def apply_tree_template(obj: dict, config_path: Path | str | None = None) -> bool:
    """Replace a dressed tree's parts when a matching optional local asset exists.

    Missing, malformed or incompatible local references leave the procedural tree
    intact and expose a short template_warning. Green broadleaf templates are not
    applied to bare, autumn, conifer or flowering geometry by default.
    """
    if obj.get('species') not in {'oak', 'maple', 'dark_oak'} or obj.get('vegetation_state', 'green') != 'green' or obj.get('height', 0) < 3:
        return False
    config_path = Path(config_path or os.environ.get('REALITYFORGE_TREE_TEMPLATES') or DEFAULT_CONFIG)
    if not config_path.is_file():
        return False
    try:
        refs = _read_config(*_source_signature(config_path))
        eligible = [ref for ref in refs if obj['species'] in ref.get('allowed_species', ['oak'])]
        if not eligible:
            return False
        number = int.from_bytes(hashlib.sha256(str(obj.get('id', obj['center'])).encode()).digest()[:8], 'big')
        ref = eligible[number % len(eligible)]
        path = Path(ref['path'])
        if not path.is_absolute():
            path = config_path.parent / path
        target = prepare_template(path, min(32, max(3, obj['height'])), ref.get('member'))
        center, base_y = obj['center'], obj['base_y']
        # Anchor cell centres to the voxel grid. All three renderers see the same parts.
        x_origin = round(center[0] - target['width'] / 2) + target['width'] / 2
        z_origin = round(center[1] - target['depth'] / 2) + target['depth'] / 2
        parts = []
        for original in target['parts']:
            cx, cy, cz = original['center']
            material = original['material']
            if obj['species'] == 'dark_oak':
                material = material.replace('minecraft:oak_', 'minecraft:dark_oak_')
            elif obj['species'] == 'maple' and '_leaves[' in material:
                material = 'minecraft:azalea_leaves[distance=7,persistent=true,waterlogged=false]'
            if obj.get('photo_id') and '_leaves' in material:
                material = obj.get('materials', {}).get('leaves', material)
            parts.append({**original, 'center': [x_origin + cx, base_y + cy, z_origin + cz], 'material': material})
        obj.update(parts=parts, height=target['height'], canopy_radius=max(target['width'], target['depth']) / 2,
                   appearance_method='local_schematic_tree_template', template_source={
                       **target['source'], 'target_height_blocks': target['height'],
                       'target_blocks': target['target_blocks'], 'part_count': target['part_count'],
                       'scale_factor': target['scale_factor'], 'silhouette_species_is_approximate': True})
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile, EOFError) as exc:
        obj['template_warning'] = f'Local reference unavailable; procedural tree used ({type(exc).__name__}).'
        return False
