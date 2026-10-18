import gzip
import io
import json
import zipfile

import nbtlib
import numpy as np
import pytest
from nbtlib import ByteArray, Compound, Int, Short, String

from forge.assets import voxelize_parts
from forge.minecraft import _encode_varints, write_schematic
from forge.tree_templates import apply_tree_template, inspect_reference, prepare_template


def _reference(tmp_path, version=3):
    blocks = np.zeros((18, 14, 16), dtype=np.uint16)
    blocks[2:14, 6:8, 7:9] = 1
    blocks[10:15, 4:10, 4:12] = 2
    blocks[12, 6:8, 7:11] = 3
    blocks[15, 6, 7] = 4
    palette = ['minecraft:air', 'minecraft:oak_wood[axis=y]',
               'minecraft:oak_leaves[distance=2,persistent=false,waterlogged=false]',
               'minecraft:oak_log[axis=x]', 'minecraft:oak_slab[type=top,waterlogged=false]']
    path = tmp_path / f'tree-v{version}.schem'
    if version == 2:
        write_schematic(path, blocks, palette, {'name': 'Original test fixture'})
    else:
        root = Compound({'Version': Int(3), 'DataVersion': Int(3955), 'Width': Short(16), 'Height': Short(18), 'Length': Short(14),
                         'Metadata': Compound({'Author': String('RealityForge test fixture')}),
                         'Blocks': Compound({'Palette': Compound({state: Int(i) for i, state in enumerate(palette)}),
                                             'Data': ByteArray(_encode_varints(blocks.ravel()))})})
        nbtlib.File({'Schematic': root}, gzipped=True).save(path)
    return path


@pytest.mark.parametrize('version', [2, 3])
def test_bounded_import_trims_air_preserves_materials_and_merges(tmp_path, version):
    path = _reference(tmp_path, version)
    report = inspect_reference(path)
    assert report['schematic_version'] == version
    assert report['source_dimensions'] == {'width': 16, 'height': 18, 'depth': 14}
    assert report['trimmed_dimensions'] == {'width': 8, 'height': 14, 'depth': 6}
    assert report['material_inferred_species'] == 'oak'
    assert report['license_status'] == 'not_found'
    result = prepare_template(path, 14)
    assert result['part_count'] < result['target_blocks'] / 4
    assert result == prepare_template(path, 14)
    materials = {part['material'] for part in result['parts']}
    assert 'minecraft:oak_log[axis=x]' in materials
    assert any('persistent=true' in material for material in materials)
    slab = next(part for part in result['parts'] if '_slab[' in part['material'])
    assert slab['size'][1] == .5 and slab['center'][1] == 13.75
    assert min(p['center'][1] - p['size'][1] / 2 for p in result['parts']) == 0


def test_rescale_keeps_trunk_and_limits_parts(tmp_path):
    path = _reference(tmp_path)
    small = prepare_template(path, 7)
    assert small['height'] == 7
    assert small['width'] == 4 and small['depth'] == 3
    assert any('wood' in part['material'] for part in small['parts'])
    assert any('leaves' in part['material'] for part in small['parts'])
    for invalid in (0, 2, 33, float('inf')):
        with pytest.raises(ValueError):
            prepare_template(path, invalid)


def test_local_zip_configuration_reuses_one_geometry_for_voxels(tmp_path):
    path = _reference(tmp_path)
    archive = tmp_path / 'local-trees.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.write(path, 'trees/oak.schem')
    config = tmp_path / 'tree_templates.json'
    config.write_text(json.dumps({'version': 1, 'templates': [{'path': str(archive), 'member': 'trees/oak.schem', 'allowed_species': ['oak']}]}))
    obj = {'id': 'tree/test', 'kind': 'tree', 'species': 'oak', 'height': 14, 'center': [15.25, 15.75], 'base_y': 2,
           'vegetation_state': 'green', 'parts': [{'original': True}]}
    assert apply_tree_template(obj, config)
    assert obj['appearance_method'] == 'local_schematic_tree_template'
    palette = ['minecraft:air']
    def block_id(material):
        if material not in palette:
            palette.append(material)
        return palette.index(material)
    voxels = np.zeros((30, 32, 32), dtype=np.uint16)
    voxelize_parts(voxels, obj, block_id)
    assert int(np.count_nonzero(voxels)) == obj['template_source']['target_blocks']
    assert any('_slab[' in material for material in palette)
    assert all(0 <= min(part['center']) for part in obj['parts'])
    previous = list(obj['parts'])
    obj['vegetation_state'] = 'bare'
    assert not apply_tree_template(obj, config)
    assert obj['parts'] == previous


def test_missing_reference_keeps_procedural_tree(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'version': 1, 'templates': [{'path': 'missing.schem'}]}))
    obj = {'id': 'tree/test', 'species': 'oak', 'height': 10, 'center': [15, 15], 'base_y': 2, 'parts': [{'original': True}]}
    assert not apply_tree_template(obj, config)
    assert obj['parts'] == [{'original': True}]
    assert 'template_warning' in obj


def test_display_bases_excluded_and_collection_not_instantiated(tmp_path):
    blocks = np.zeros((8, 32, 32), dtype=np.uint16)
    blocks[0] = 1
    blocks[1:5, 2:4, 2:4] = 2
    blocks[4:7, 1:6, 1:6] = 3
    blocks[1:5, 28:30, 28:30] = 2
    blocks[4:7, 26:31, 26:31] = 3
    path = tmp_path / 'display.schem'
    write_schematic(path, blocks, ['minecraft:air', 'minecraft:grass_block', 'minecraft:oak_log[axis=y]', 'minecraft:oak_leaves[persistent=true]'])
    report = inspect_reference(path)
    assert report['ignored_blocks'] == 1024
    assert report['collection_candidate']
    with pytest.raises(ValueError, match='collection'):
        prepare_template(path, 10)
