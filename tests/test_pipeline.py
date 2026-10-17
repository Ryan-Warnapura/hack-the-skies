import io
import json
import time
import zipfile

import nbtlib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from forge.geography import centered_bbox, demo_osm, parse_location, validate_bbox
from forge.minecraft import read_schematic
from forge.photos import analyze_photo
from forge.pipeline import generate
from forge.scene import build_scene, export_obj, scene_to_voxels


def test_location_formats_and_limits():
    assert parse_location('43.65, -79.38') == {'lat': 43.65, 'lon': -79.38}
    assert parse_location('https://www.google.com/maps/place/Test/@43.65,-79.38,17z/data=!3d43.66!4d-79.39') == {'lat': 43.66, 'lon': -79.39}
    assert parse_location('https://www.google.com/maps/search/?api=1&query=43.65%2C-79.38')['lat'] == 43.65
    for invalid in ['https://evil.example/@43.65,-79.38', 'https://maps.app.goo.gl/abc', 'nan,20', '90,2']:
        with pytest.raises(ValueError):
            parse_location(invalid)
    with pytest.raises(ValueError):
        validate_bbox([-80, 43, -79, 44])


def test_photo_changes_only_nearby_building_materials(tmp_path):
    photo_path = tmp_path / 'brick.png'
    Image.new('RGB', (800, 300), (170, 75, 55)).save(photo_path)
    photo = analyze_photo(photo_path)
    assert photo['projection'] == 'panorama'
    bbox = centered_bbox(43.65, -79.38, 100)
    osm = demo_osm(bbox)
    plain = build_scene(osm, bbox)
    styled = build_scene(osm, bbox, photos=[{**photo, 'id': 'test', 'lat': 43.65, 'lon': -79.38, 'radius_m': 25, 'style': 'wood'}])
    changed = [o for o in styled['objects'] if o['source'] == 'osm+photo']
    assert 0 < len(changed) < plain['stats']['building']
    for before, after in zip(plain['objects'], styled['objects']):
        assert before['geometry'] == after['geometry']
        if after['source'] == 'osm+photo':
            assert after['materials']['wall'] == 'minecraft:oak_planks'
        else:
            assert before['materials'] == after['materials']


def test_scene_semantics_terrain_and_mesh(tmp_path):
    bbox = centered_bbox(43.65, -79.38, 64)
    scene = build_scene(demo_osm(bbox), bbox, elevation={'heights': [[0, 4], [6, 9]], 'source': 'test DEM'})
    assert scene['terrain']['kind'] == 'elevation'
    blocks, palette = scene_to_voxels(scene)
    assert blocks.shape == (scene['bounds']['height'], scene['bounds']['depth'], scene['bounds']['width'])
    assert (blocks[0] == palette.index('minecraft:bedrock')).all()
    assert any('oak_leaves' in p for p in palette)
    assert any('oak_door' in p for p in palette)
    assert any('glass' in p for p in palette)
    path = export_obj(scene, tmp_path / 'model.obj')
    text = path.read_text()
    assert '\nv ' in text and '\nf ' in text and '\nusemtl ' in text
    assert path.with_suffix('.mtl').exists()


def test_pipeline_writes_real_intermediates_and_world(tmp_path):
    events = []
    result = generate({'bbox': centered_bbox(43.65, -79.38, 64), 'source': 'demo', 'name': 'Integration'}, tmp_path / 'out', tmp_path / 'cache', lambda p,s,m: events.append(s))
    out = tmp_path / 'out'
    assert events.index('model') < events.index('schematic') < events.index('world')
    blocks, palette, metadata = read_schematic(out / 'realityforge.schem')
    assert blocks.size > 1000 and len(palette) > 10
    assert metadata['name'] == 'Integration'
    data = nbtlib.load(out / 'RealityForge' / 'level.dat')['Data']
    assert int(data['DataVersion']) == 3700
    assert str(data['LevelName']) == 'Integration'
    with zipfile.ZipFile(out / 'RealityForge-world.zip') as archive:
        assert 'RealityForge/level.dat' in archive.namelist()
        assert any(name.endswith('.mca') for name in archive.namelist())
    assert any('Synthetic demo' in s for s in result['report']['limitations'])


def test_http_photo_to_download_and_rejections(tmp_path, monkeypatch):
    from forge import app as module
    monkeypatch.setattr(module, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(module, 'UPLOADS', tmp_path / 'uploads')
    monkeypatch.setattr(module, 'DATA', tmp_path)
    module.JOBS.mkdir(); module.UPLOADS.mkdir()
    with TestClient(module.app) as client:
        assert client.get('/').status_code == 200
        assert client.post('/api/location', json={'value':'43.65,-79.38'}, headers={'origin':'https://untrusted.example'}).status_code == 403
        assert client.post('/api/photos', files={'files': ('bad.jpg', b'not an image', 'image/jpeg')}).status_code == 422
        image = io.BytesIO(); Image.new('RGB', (600, 200), (160, 80, 50)).save(image, 'JPEG')
        upload = client.post('/api/photos', files={'files': ('reference.jpg', image.getvalue(), 'image/jpeg')})
        assert upload.status_code == 200
        photo = upload.json()['photos'][0]
        request = {'name':'API world', 'bbox':centered_bbox(43.65,-79.38,64), 'source':'demo', 'photos':[{'id':photo['id'],'lat':43.65,'lon':-79.38,'style':'wood','radius_m':80}]}
        response = client.post('/api/generate', json=request)
        assert response.status_code == 202
        for _ in range(100):
            job = client.get('/api/jobs/' + response.json()['id']).json()
            if job['status'] in {'complete','failed'}:
                break
            time.sleep(.03)
        assert job['status'] == 'complete', job
        assert job['stats']['photo_dressed_buildings'] > 0
        for url in job['files'].values():
            assert client.get(url).status_code == 200
        monkeypatch.setattr(module.sys, 'platform', 'win32')
        monkeypatch.setenv('APPDATA', str(tmp_path / 'fake-appdata'))
        installed = client.post('/api/jobs/' + job['id'] + '/install')
        assert installed.status_code == 200
        installed_path = tmp_path / 'fake-appdata/.minecraft/saves' / ('RealityForge-' + job['id'][:8])
        assert (installed_path / 'level.dat').exists()
        assert client.post('/api/jobs/' + job['id'] + '/install').status_code == 409
        request['bbox'] = [-80,40,-79,44]
        assert client.post('/api/generate', json=request).status_code == 422
        assert client.get('/api/jobs/not-a-real-id/files/world').status_code == 404


def test_roads_are_clipped_before_expensive_geometry():
    from forge.scene import _clip_road
    assert _clip_road([-100000, 16], [100000, 16], 64, 64) == [[0., 16.], [63.99, 16.]]
    assert _clip_road([-30, -20], [-10, -10], 64, 64) is None
