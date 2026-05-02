from __future__ import annotations

from pathlib import Path

from mtplx.hf_loader import (
    cached_model_path,
    hf_cache_report,
    list_cached_models,
    remove_cached_model,
    repo_id_from_model_ref,
    resolve_model_path,
    safe_model_name,
)


def test_repo_id_from_model_ref_accepts_hf_url_and_repo_id():
    assert repo_id_from_model_ref("mtplx/example") == "mtplx/example"
    assert (
        repo_id_from_model_ref("https://huggingface.co/mtplx/example/tree/main")
        == "mtplx/example"
    )
    assert repo_id_from_model_ref("models/local-model") is None


def test_safe_model_name_and_cache_path(tmp_path: Path):
    assert safe_model_name("mtplx/example") == "mtplx--example"
    assert cached_model_path("mtplx/example", cache_dir=tmp_path) == tmp_path / "mtplx--example"


def test_resolve_model_path_uses_cache_for_hf_refs(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()

    assert resolve_model_path("mtplx/example", cache_dir=tmp_path) == cached


def test_resolve_model_path_reports_missing_cache(tmp_path: Path):
    try:
        resolve_model_path("mtplx/example", cache_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "mtplx pull mtplx/example" in str(exc)
    else:
        raise AssertionError("expected missing cache error")


def test_list_and_remove_cached_models(tmp_path: Path):
    model = tmp_path / "mtplx--example"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text("{}\n", encoding="utf-8")
    (model / "small.bin").write_bytes(b"1234")

    rows = list_cached_models(cache_dir=tmp_path)

    assert len(rows) == 1
    assert rows[0].repo_id == "mtplx/example"
    assert rows[0].has_config is True
    assert rows[0].has_runtime_contract is True
    assert rows[0].size_bytes >= 4

    removed = remove_cached_model("mtplx/example", cache_dir=tmp_path)
    assert removed["removed"] is True
    assert not model.exists()


def test_hf_cache_report_is_no_network(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    cache = tmp_path / "missing-cache"
    report = hf_cache_report(cache_dir=cache)

    assert report["cache_dir"] == str(cache)
    assert report["cache_exists"] is False
    assert report["cached_models"] == 0
    assert "token_present" in report
