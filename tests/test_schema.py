import json
from typing import cast

from stoquant.schema import (
    CaseRecord,
    MatrixManifest,
    read_case,
    read_manifest,
    read_snapshot,
    write_case,
    write_manifest,
)


def test_typed_case_and_manifest_round_trip(tmp_path):
    case = cast(
        CaseRecord,
        {
            "case_id": "case_cpu_bits8_n1024",
            "count": 1024,
            "bits": 8,
            "input_provenance": {"input_family": "dense", "count": 1024},
        },
    )
    manifest = cast(MatrixManifest, {"manifest_version": "3.2", "cases": [case["case_id"]]})

    write_case(tmp_path / f"{case['case_id']}.json", case)
    write_manifest(tmp_path / "manifest.json", manifest)

    assert read_case(tmp_path / f"{case['case_id']}.json") == case
    assert read_manifest(tmp_path / "manifest.json") == manifest
    assert read_snapshot(tmp_path) == (manifest, [case])
    assert (tmp_path / "manifest.json").read_bytes().endswith(b"}")


def test_typed_readers_reject_non_object_json(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(["not", "a", "manifest"]), encoding="utf-8")
    try:
        read_manifest(path)
    except TypeError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("a JSON list was accepted as a manifest")
