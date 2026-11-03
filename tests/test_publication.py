import json
import pytest
from tools.publish_reports import publish_reports
from race_strategy.weekend import atomic_json
from tests.test_report import report


def test_catalog_atomic_and_no_raw_cache(tmp_path):
    source=tmp_path/"source"; target=tmp_path/"public"
    atomic_json(source/"2026-3.json",report())
    (source/"secret.sqlite").write_text("not public")
    catalog=publish_reports(source,target)
    before=catalog.read_bytes()
    assert not (target/"secret.sqlite").exists()
    assert json.loads(before)["weekends"][0]["year"] == 2026
    atomic_json(source/"bad.json",{})
    with pytest.raises(ValueError): publish_reports(source,target)
    assert catalog.read_bytes()==before
