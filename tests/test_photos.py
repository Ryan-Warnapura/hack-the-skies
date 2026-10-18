import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from pydantic import ValidationError

from forge.photos import analyze_photo, extract_photo_metadata, trim_uniform_borders


def test_wide_image_is_not_automatically_a_panorama(tmp_path):
    path = tmp_path / 'wide.png'
    Image.new('RGB', (1000, 300), (160, 75, 50)).save(path)
    result = analyze_photo(path)
    assert result['wide_image_candidate'] is True
    assert result['projection'] == 'perspective'
    assert result['estimated_counts']['people'] is None
    assert result['estimated_counts']['vehicles'] is None
    assert result['estimated_counts']['trees'] is None
    assert analyze_photo(path, projection='panorama')['fov'] == 360
    assert analyze_photo(path, metadata={'metadata_projection': 'panorama'})['projection'] == 'panorama'
    cropped = analyze_photo(path, metadata={'metadata_projection': 'panorama'}, crop=[.1, 0, .9, 1])
    assert cropped['projection'] == 'perspective'


def test_true_and_magnetic_heading_are_not_confused(tmp_path):
    for reference in ('T', 'M'):
        exif = Image.Exif()
        exif[34853] = {16: reference, 17: 123.5}
        path = tmp_path / f'{reference}.jpg'
        Image.new('RGB', (200, 150), (30, 80, 120)).save(path, exif=exif)
        data = analyze_photo(path)
        assert data['exif_heading'] == 123.5
        assert data['heading'] == (123.5 if reference == 'T' else None)
        assert analyze_photo(path, heading=44)['heading'] == 44
    with Image.open(path) as image:
        image.info['xmp'] = b'<rdf GPano:ProjectionType="equirectangular"/>'
        assert extract_photo_metadata(image)['metadata_projection'] == 'panorama'


def test_crop_removes_overlay_and_is_deterministic(tmp_path):
    path = tmp_path / 'screenshot.png'
    image = Image.new('RGB', (300, 200), (170, 75, 55))
    ImageDraw.Draw(image).rectangle((0, 0, 299, 19), fill='white')
    image.save(path)
    automatic = analyze_photo(path)
    assert automatic['auto_trimmed'] is True
    assert automatic['height'] == 180
    assert automatic['applied_crop'] == [0, 20, 300, 200]
    manual = analyze_photo(path, crop=[.1, .2, .9, .9], output_path=tmp_path / 'crop.jpg')
    assert (manual['width'], manual['height']) == (240, 140)
    assert manual == analyze_photo(path, crop=[.1, .2, .9, .9])
    with pytest.raises(ValueError):
        analyze_photo(path, crop=[.8, 0, .2, 1])
    with pytest.raises(ValueError):
        analyze_photo(path, crop=[0, 0, .001, .001])


def test_trim_does_not_destroy_white_black_sky_or_green_images(tmp_path):
    for color in ('white', 'black', '#3399cc', '#339933'):
        image = Image.new('RGB', (200, 100), color)
        trimmed, box = trim_uniform_borders(image)
        assert trimmed.size == image.size and box == [0, 0, 200, 100]
        path = tmp_path / 'solid.png'
        image.save(path)
        result = analyze_photo(path)
        assert result['estimated_counts']['trees'] is None


def test_vegetation_estimates_are_explicitly_uncertain(tmp_path, monkeypatch):
    image = Image.new('RGB', (200, 200), '#789fcb')
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 50, 65, 110), fill='#386e29')
    draw.ellipse((120, 30, 175, 95), fill='#478f32')
    path = tmp_path / 'vegetation.png'
    image.save(path)
    monkeypatch.setenv('REALITYFORGE_ONNX_MODEL', str(tmp_path / 'unavailable.onnx'))
    result = analyze_photo(path)
    assert result['vegetation_state'] == 'green'
    assert result['estimated_counts'] == {'trees': 2, 'people': None, 'vehicles': None}
    assert result['tree_count_confidence'] < .3
    assert result['foliage_colors']
    assert result['segmentation']['status'] == 'unavailable'


def test_http_upload_preserves_metadata_and_crop_uses_original(tmp_path, monkeypatch):
    from forge import app as module
    monkeypatch.setattr(module, 'UPLOADS', tmp_path)
    exif = Image.Exif()
    exif[34853] = {16: 'T', 17: 92.5}
    image = io.BytesIO()
    Image.new('RGB', (400, 200), '#ab6644').save(image, 'JPEG', exif=exif)
    with TestClient(module.app) as client:
        response = client.post('/api/photos', files={'files': ('view.jpg', image.getvalue(), 'image/jpeg')})
        assert response.status_code == 200
        photo = response.json()['photos'][0]
        assert photo['heading'] == 92.5
        assert photo['projection'] == 'perspective'
        assert client.get(photo['original_url']).status_code == 200
        endpoint = f"/api/photos/{photo['id']}/analyze"
        crop = client.post(endpoint, json={'crop': [.25, 0, .75, 1], 'tree_style': 'bare'})
        assert crop.status_code == 200 and crop.json()['width'] == 200
        assert crop.json()['tree_style'] == 'bare'
        restore = client.post(endpoint, json={'crop': [0, 0, 1, 1]})
        assert restore.status_code == 200 and restore.json()['width'] == 400
        assert restore.json()['heading'] == 92.5
        assert client.post(endpoint, json={'crop': [0, 0, 2, 1]}).status_code == 422
        assert client.post(endpoint, json={'heading': 360}).status_code == 422


def test_annotation_schema_rejects_untrusted_fields_and_bad_colors():
    from forge.app import GenerationRequest
    request = {'bbox': [-79.381, 43.649, -79.379, 43.651], 'annotations': [
        {'kind': 'person', 'lat': 43.65, 'lon': -79.38, 'appearance': {'top': '#fa8042'}},
        {'kind': 'tree', 'lat': 43.65, 'lon': -79.38, 'tree_species': 'maple'},
    ]}
    result = GenerationRequest.model_validate(request)
    assert result.annotations[0].appearance.top == '#fa8042'
    for bad in ({'source': 'osm'}, {'color': 'red'}, {'tree_species': 'imaginary'}, {'lat': float('nan')}):
        request['annotations'][1].update(bad)
        with pytest.raises(ValidationError):
            GenerationRequest.model_validate(request)
        request['annotations'][1] = {'kind': 'tree', 'lat': 43.65, 'lon': -79.38, 'tree_species': 'maple'}


def test_generation_forwards_verified_annotation_provenance_and_photo_metadata(tmp_path, monkeypatch):
    from forge import app as module
    from types import SimpleNamespace
    monkeypatch.setattr(module, 'UPLOADS', tmp_path)
    monkeypatch.setattr(module, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(module, 'states', {})
    module.JOBS.mkdir()
    captured = []
    monkeypatch.setattr(module, 'executor', SimpleNamespace(submit=lambda function, job_id, payload: captured.append(payload)))
    photo_id = 'a' * 32
    (tmp_path / f'{photo_id}.json').write_text(json.dumps({'projection': 'panorama', 'heading': 85, 'fov': 360}))
    with TestClient(module.app) as client:
        response = client.post('/api/generate', json={
            'bbox': [-79.381, 43.649, -79.379, 43.651],
            'photos': [{'id': photo_id, 'lat': 43.65, 'lon': -79.38}],
            'annotations': [{'kind': 'person', 'lat': 43.65, 'lon': -79.38, 'appearance': {'top': '#fa8042'}}],
        })
        assert response.status_code == 202
    assert captured[0]['annotations'][0]['source'] == 'user_annotation'
    assert captured[0]['annotations'][0]['confidence'] == 1.0
    assert captured[0]['photos'][0]['projection'] == 'panorama'
    assert captured[0]['photos'][0]['heading'] == 85
    assert captured[0]['photos'][0]['fov'] == 360
