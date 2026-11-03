import json
import sys
from fastapi.testclient import TestClient
from race_strategy.service import create_app
from race_strategy.weekend import atomic_json
from tests.test_report import report


def setup_store(tmp_path):
    r=report()
    atomic_json(tmp_path/"reports"/"test.json",r)
    atomic_json(tmp_path/"catalog.json",dict(schema_version=1,weekends=[dict(year=2026,round=3,event=r["event"],file="test.json")],replays=[]))
    return tmp_path


def test_reads_missing_ids_and_old_public_routes(tmp_path):
    app=create_app(setup_store(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").json()["mode"] == "prepared-only"
        assert client.get("/api/strategy/weekends").status_code == 200
        assert client.get("/api/strategy/prerace/2026/3").json()["year"] == 2026
        assert client.get("/api/strategy/prerace/2026/99").status_code == 404
        assert client.get("/api/session?year=2026&gp=Unknown").status_code == 404
        assert client.get("/api/time-delta").status_code == 409
        assert client.get("/api/replay/2026/3/chunks/unknown").status_code == 404
        assert client.post("/api/strategy/scenarios",json={}).status_code == 409
    assert "fastf1" not in sys.modules


def test_corrupt_artifact_and_rate_limit(tmp_path):
    root=setup_store(tmp_path)
    (root/"reports/test.json").write_text("{}")
    with TestClient(create_app(root,read_limit=2)) as client:
        assert client.get("/api/strategy/prerace/2026/3").status_code == 503
        client.get("/api/strategy/weekends")
        response=client.get("/api/strategy/weekends")
        assert response.status_code == 429
        assert response.headers["retry-after"] == "60"


def test_report_conditional_get(tmp_path):
    with TestClient(create_app(setup_store(tmp_path))) as client:
        r=client.get("/api/strategy/prerace/2026/3")
        assert client.get("/api/strategy/prerace/2026/3",headers={"If-None-Match":r.headers["etag"]}).status_code==304
