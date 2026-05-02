from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import save_file

from mtplx.artifacts import expected_mtp_file, inspect_model
from mtplx.constants import EXPECTED_MTP_KEYS, EXPECTED_PREQUANTIZED_MTP_KEYS


def _write_runtime_contract(path, *, arch_id="qwen3-next-mtp", profile="stable"):
    (path / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "mtplx_version": "0.1.0-preview",
                "arch_id": arch_id,
                "mtp_depth_max": 3,
                "recommended_profile": profile,
                "exactness_baseline": {"phase0h": "smoke", "max_abs_diff": 0.0},
                "verified_on": {
                    "timestamp": "2026-05-02T00:00:00Z",
                    "hardware": "test",
                    "macos": "test",
                },
            }
        ),
        encoding="utf-8",
    )


def test_expected_mtp_file_uses_extra_tensor_metadata(tmp_path):
    config = {"mlx_lm_extra_tensors": {"mtp_file": "extra-mtp.safetensors"}}
    assert expected_mtp_file(tmp_path, config) == tmp_path / "extra-mtp.safetensors"


def test_inspect_model_reports_missing_config(tmp_path):
    result = inspect_model(tmp_path)
    assert result.config_exists is False
    assert result.passes_primary_gate is False


def test_inspect_model_reads_qwen_mtp_config_without_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        )
    )
    result = inspect_model(tmp_path)
    assert result.model_type == "qwen3_5"
    assert result.mtp_num_hidden_layers == 1
    assert result.mtp is not None
    assert result.mtp.exists is False
    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["exit_code"] == 3


def test_qwen3_5_text_subtype_can_pass_primary_gate_when_mtp_is_valid(monkeypatch, tmp_path):
    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "mtp_num_hidden_layers": 1,
                    "hidden_size": 5120,
                    "num_hidden_layers": 64,
                    "vocab_size": 248320,
                },
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        )
    )

    monkeypatch.setattr(
        artifacts,
        "inspect_mtp_tensors",
        lambda *_args, **_kwargs: MTPInspection(
            mtp_file=str(tmp_path / "mtp.safetensors"),
            exists=True,
            tensor_count=15,
            missing_expected_keys=(),
        ),
    )
    _write_runtime_contract(tmp_path)
    assert inspect_model(tmp_path).passes_primary_gate is True


def test_prequantized_mtp_sidecar_accepts_mlx_affine_scale_bias_tensors(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "mtplx_mtp_quantization": {
                    "policy": "cyankiwi",
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                    "prequantized": True,
                },
            }
        )
    )
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in EXPECTED_PREQUANTIZED_MTP_KEYS},
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)
    assert result.mtp is not None
    assert result.mtp.sidecar_format == "prequantized-mlx-affine"
    assert result.mtp.tensor_count == 29
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True
    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["recommended_profile"] == "performance-cold"


def test_qwen_mtp_without_runtime_contract_is_unverified(monkeypatch, tmp_path):
    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        artifacts,
        "inspect_mtp_tensors",
        lambda *_args, **_kwargs: MTPInspection(
            mtp_file=str(tmp_path / "mtp.safetensors"),
            exists=True,
            tensor_count=15,
            missing_expected_keys=(),
        ),
    )

    result = inspect_model(tmp_path)

    assert result.passes_primary_gate is False
    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["unsafe_force_required"] is True


def test_qwen3_next_architecture_without_mtp_sidecar_is_unverified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["exit_code"] == 3
    assert result.compatibility["runtime_compatibility"] == "needs-grafting"


def test_deepseek_mtp_is_incompatible_architecture(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "num_nextn_predict_layers": 2,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "incompatible-architecture"
    assert result.compatibility["exit_code"] == 4


def test_llama_without_mtp_is_no_mtp(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["exit_code"] == 2


def test_hf_qwen_mtp_without_runtime_contract_is_unverified(monkeypatch):
    from mtplx import artifacts

    calls = []

    def fake_files(repo_id):
        assert repo_id == "Qwen/Qwen3-Next-80B-A3B-Instruct"
        return {"config.json", "mtp.safetensors", "model.safetensors.index.json"}, None

    def fake_json(repo_id, filename):
        if filename == "config.json":
            return (
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "mtp_num_hidden_layers": 1,
                },
                "/tmp/config.json",
                None,
            )
        if filename == "mtplx_runtime.json":
            return None, None, "404 Client Error: entry not found"
        raise AssertionError(filename)

    def fake_keys(repo_id, filename):
        calls.append((repo_id, filename))
        return tuple(sorted(EXPECTED_MTP_KEYS)), None

    monkeypatch.setattr(artifacts, "_hf_list_repo_files", fake_files)
    monkeypatch.setattr(artifacts, "_hf_download_json", fake_json)
    monkeypatch.setattr(artifacts, "_remote_safetensors_keys", fake_keys)

    result = inspect_model("Qwen/Qwen3-Next-80B-A3B-Instruct")

    assert result.source == "hf"
    assert result.mtp is not None
    assert result.mtp.metadata_only is True
    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["exit_code"] == 3
    assert calls == [("Qwen/Qwen3-Next-80B-A3B-Instruct", "mtp.safetensors")]


def test_hf_verified_contract_passes_metadata_gate(monkeypatch):
    from mtplx import artifacts

    def fake_files(_repo_id):
        return {"config.json", "mtplx_runtime.json", "mtp.safetensors"}, None

    def fake_json(_repo_id, filename):
        if filename == "config.json":
            return (
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "mtp_num_hidden_layers": 1,
                    "mtplx_mtp_quantization": {
                        "prequantized": True,
                        "bits": 4,
                        "group_size": 32,
                        "mode": "affine",
                    },
                },
                "/tmp/config.json",
                None,
            )
        if filename == "mtplx_runtime.json":
            return (
                {
                    "mtplx_version": "0.1.0-preview",
                    "arch_id": "qwen3-next-mtp",
                    "mtp_depth_max": 3,
                    "recommended_profile": "stable",
                    "exactness_baseline": {"phase0h": "smoke", "max_abs_diff": 0.0},
                    "verified_on": {"timestamp": "2026-05-02T00:00:00Z"},
                },
                "/tmp/mtplx_runtime.json",
                None,
            )
        raise AssertionError(filename)

    monkeypatch.setattr(artifacts, "_hf_list_repo_files", fake_files)
    monkeypatch.setattr(artifacts, "_hf_download_json", fake_json)
    monkeypatch.setattr(
        artifacts,
        "_remote_safetensors_keys",
        lambda _repo_id, _filename: (tuple(sorted(EXPECTED_PREQUANTIZED_MTP_KEYS)), None),
    )

    result = inspect_model("https://huggingface.co/mtplx/example/tree/main")

    assert result.source == "hf"
    assert result.mtp is not None
    assert result.mtp.metadata_only is True
    assert result.mtp.passes_tensor_gate is True
    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["can_run"] is True
    assert result.runtime_contract_path == "/tmp/mtplx_runtime.json"


def test_hf_llama_without_mtp_is_no_mtp(monkeypatch):
    from mtplx import artifacts

    monkeypatch.setattr(
        artifacts,
        "_hf_list_repo_files",
        lambda _repo_id: ({"config.json", "model.safetensors.index.json"}, None),
    )

    def fake_json(_repo_id, filename):
        if filename == "config.json":
            return (
                {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
                "/tmp/config.json",
                None,
            )
        if filename == "mtplx_runtime.json":
            return None, None, "404 Client Error: not found"
        raise AssertionError(filename)

    monkeypatch.setattr(artifacts, "_hf_download_json", fake_json)

    result = inspect_model("https://huggingface.co/meta-llama/Llama-3.2-1B")

    assert result.source == "hf"
    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["exit_code"] == 2
