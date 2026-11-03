"""Validate the actual shipped pilot, not just synthetic API fixtures."""
import gzip
import json
from pathlib import Path
from fastapi.testclient import TestClient
from race_strategy.service import create_app
from race_strategy.report import validate_report

ROOT = Path(__file__).resolve().parents[1] / 'artifacts'


def test_published_reports_and_fallback_match():
    catalog = json.loads((ROOT / 'catalog.json').read_text())
    samples = ROOT.parent / 'frontend/public/samples'
    fallback = json.loads((samples / 'catalog.json').read_text())
    assert fallback['weekends'] == catalog['weekends']
    for entry in catalog['weekends']:
        raw = (ROOT / 'reports' / entry['file']).read_bytes()
        assert raw == (samples / 'reports' / entry['file']).read_bytes()
        report = validate_report(json.loads(raw))
        assert (report['year'], report['round']) == (entry['year'], entry['round'])
        assert len(gzip.compress(raw)) < 1024 * 1024


def test_replay_chunks_reconstruct_the_published_driver_field():
    with TestClient(create_app(ROOT)) as client:
        catalog = client.get('/api/strategy/weekends').json()
        for replay in catalog['replays']:
            path = f"/api/replay/{replay['year']}/{replay['round']}"
            manifest = client.get(path + '/manifest').json()
            drivers = {}
            for chunk in manifest['chunks']:
                response = client.get(path + '/chunks/' + chunk['id'])
                assert response.status_code == 200
                assert response.headers['content-encoding'] == 'gzip'
                assert chunk['bytes'] < 2 * 1024 * 1024
                values = response.json()['telemetry']
                assert not set(drivers).intersection(values)
                drivers.update(values)
            assert set(drivers) == {d['driver'] for d in manifest['session']['leaderboard']}
            for values in drivers.values():
                assert len(values['Distance']) == len(values['Trel'])
                assert len(values['Trel']) > 0


def test_cors_preflight_allows_only_configured_frontend(monkeypatch):
    origin = 'https://f1-telemetry-sooty.vercel.app'
    monkeypatch.setenv('CORS_ORIGINS', origin)
    with TestClient(create_app(ROOT)) as client:
        good = client.options('/api/strategy/weekends', headers={'Origin':origin, 'Access-Control-Request-Method':'GET'})
        assert good.headers['access-control-allow-origin'] == origin
        bad = client.options('/api/strategy/weekends', headers={'Origin':'https://unrelated.example', 'Access-Control-Request-Method':'GET'})
        assert bad.status_code == 400
        assert 'access-control-allow-origin' not in bad.headers
