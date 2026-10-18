import json
import math
from pathlib import Path

import numpy as np
import pytest

from forge.geography import centered_bbox, select_osm_features, validate_bbox
from forge.scene import build_scene, scene_to_voxels, _projection

BBOX = centered_bbox(43.65, -79.38, 64)


def location(x,z):
    w,s,e,n=BBOX
    return {'lon':w+(e-w)*x/64,'lat':n-(n-s)*z/64}


def tree(identifier,x,z,**tags):
    return {'type':'node','id':identifier,**location(x,z),'tags':{'natural':'tree',**tags}}


def road(width=6):
    return {'type':'way','id':10,'tags':{'highway':'residential','width':str(width)},'geometry':[location(32,0),location(32,64)]}


def test_sidewalk_and_snapped_trees_reach_actual_schematic_voxels():
    scene=build_scene({'elements':[road(),tree(1,36,20),tree(2,32,40)]},BBOX)
    records={r['id']:r for r in scene['tree_placement']}
    assert records['node/1']['status']=='placed'
    assert records['node/2']['status']=='snapped'
    assert math.dist(records['node/2']['position'],records['node/2']['original_position'])<=3
    assert scene['stats']['tree_osm_total']==scene['stats']['tree_osm_placed']==2
    blocks,palette=scene_to_voxels(scene)
    for obj in scene['objects']:
        if obj['kind']=='tree':
            x,z=map(math.floor,obj['center'])
            # Root must be physically present; semantic counts alone missed the old bug.
            neighbourhood=blocks[obj['base_y']:obj['base_y']+2,max(0,z-1):z+2,max(0,x-1):x+2]
            assert any('_log' in palette[int(v)] or '_wood' in palette[int(v)] for v in neighbourhood.ravel())


def test_unsnappable_carriageway_tree_has_explicit_drop_reason():
    scene=build_scene({'elements':[road(20),tree(1,32,20)]},BBOX)
    assert scene['stats']['trees_placed']==0
    assert scene['stats']['tree_osm_dropped']==1
    assert scene['tree_placement'][0]['reason']=='no_free_verge_within_3_blocks'


def test_toronto_has_diverse_deterministic_trees_and_no_surface_underground_garage():
    osm=json.loads((Path(__file__).parent/'fixtures/toronto-osm.json').read_text())
    bbox=[-79.3847,43.6521,-79.3817,43.6543]
    a=build_scene(osm,bbox);b=build_scene(osm,bbox)
    assert a==b
    assert len(a['stats']['tree_species'])>=5
    assert a['stats']['tree_osm_total']==34
    assert a['stats']['tree_osm_placed']==34
    assert not any(o['kind']=='parking' and o.get('tags',{}).get('parking')=='underground' for o in a['objects'])
    va,pa=scene_to_voxels(a);vb,pb=scene_to_voxels(b)
    assert pa==pb and np.array_equal(va,vb)
    assert len(pa)>=40
    assert a['stats']['pedestrians']<=min(300,a['stats']['pedestrian_walkable_area_m2']//400)


@pytest.mark.parametrize('scale',[1,2,4,8])
def test_tree_height_uses_osm_metres_without_giant_minimum(scale):
    scene=build_scene({'elements':[tree(1,20,20,height='12',genus='Betula')]},BBOX,scale)
    obj=next(o for o in scene['objects'] if o['kind']=='tree')
    assert obj['species']=='birch'
    assert obj['height']*scale==pytest.approx(12,abs=.01)
    physical_height=max(p['center'][1]+p['size'][1]/2 for p in obj['parts'])-obj['base_y']
    assert physical_height*scale<15


def test_heading_fov_and_explicit_bare_photo_style():
    photo={**location(20,32),'id':'photo','heading':0,'fov':60,'radius_m':60,'tree_style':'bare'}
    scene=build_scene({'elements':[tree(1,20,12),tree(2,20,50)]},BBOX,photos=[photo])
    trees={o['id']:o for o in scene['objects'] if o['kind']=='tree'}
    assert trees['node/1']['vegetation_state']=='bare'
    assert not any('leaves' in p['material'] for p in trees['node/1']['parts'])
    assert 'photo_id' not in trees['node/2']


def test_manual_tree_correction_and_photo_positions_have_honest_provenance():
    annotation={'kind':'tree',**location(20,20),'tree_species':'cherry','replace_id':'node/1','id':None}
    photo={**location(45,45),'id':'photo','heading':0,'fov':70,'estimated_distance_m':10,'estimated_counts':{'trees':2},'radius_m':20}
    scene=build_scene({'elements':[tree(1,20,20)]},BBOX,photos=[photo],annotations=[annotation])
    annotated=[o for o in scene['objects'] if o['source']=='user_annotation']
    assert len(annotated)==1 and annotated[0]['confidence']==1 and annotated[0]['species']=='cherry'
    assert not any(o['id']=='node/1' for o in scene['objects'])
    estimates=[o for o in scene['objects'] if o['source']=='photo_estimate']
    assert estimates and all(o['confidence']<=.3 for o in estimates)


def test_expanded_query_fallback_keeps_same_features_and_dependencies():
    entries=[{'type':'node','id':1,'lat':0,'lon':0},
             {'type':'way','id':2,'nodes':[1],'tags':{'natural':'tree_row'}},
             {'type':'node','id':3,'tags':{'highway':'street_lamp'}},
             {'type':'node','id':4,'tags':{'amenity':'restaurant'}},
             {'type':'way','id':5,'tags':{'amenity':'parking'}}]
    result=select_osm_features({'elements':entries})
    assert result['query_version']=='details-v2'
    assert {e['id'] for e in result['elements']}=={1,2,3,5}


def test_five_kilometre_bounds_and_overview_scale():
    bbox=centered_bbox(43.65,-79.38,5000)
    assert validate_bbox(bbox)==bbox
    scene=build_scene({'elements':[]},bbox,8)
    assert scene['bounds']['width']==scene['bounds']['depth']==625
    with pytest.raises(ValueError,match='1024'):
        build_scene({'elements':[]},bbox,1)
    with pytest.raises(ValueError,match='5,000'):
        validate_bbox(centered_bbox(43.65,-79.38,5100))


def test_coarse_people_use_small_stands_or_are_omitted(tmp_path):
    import nbtlib
    from forge.geography import demo_osm
    from forge.minecraft import scene_entities, write_schematic
    bbox=centered_bbox(43.65,-79.38,250)
    scene=build_scene(demo_osm(bbox),bbox,2)
    people=[o for o in scene['objects'] if o['kind']=='person']
    assert people and all(o['entity_small'] and o['height']<1 for o in people)
    blocks,palette=scene_to_voxels(scene)
    path=tmp_path/'small.schem'
    write_schematic(path,blocks,palette,entities=scene_entities(scene))
    assert all(int(e['Small'])==1 for e in nbtlib.load(path)['Entities'])
    coarse=build_scene(demo_osm(bbox),bbox,8)
    assert coarse['stats']['pedestrians']==0
    assert not any(o['kind']=='person' for o in coarse['objects'])
