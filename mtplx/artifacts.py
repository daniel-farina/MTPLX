"""Model artifact inspection for Qwen3.6 MTP gates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    EXPECTED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_TENSOR_COUNT,
    EXPECTED_MTP_TENSOR_COUNT,
    MULTIMODAL_SIDECARS,
)


def load_config(model_dir: Path | str) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    return json.loads(path.read_text())


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config", config)


def expected_mtp_file(model_dir: Path | str, config: dict[str, Any] | None = None) -> Path:
    model_path = Path(model_dir)
    config = config if config is not None else load_config(model_path)
    extra = config.get("mlx_lm_extra_tensors", {})
    if isinstance(extra, dict) and extra.get("mtp_file"):
        return model_path / str(extra["mtp_file"])
    for rel in ("mtp.safetensors", "mtp/weights.safetensors", "model-mtp.safetensors"):
        candidate = model_path / rel
        if candidate.exists():
            return candidate
    return model_path / "mtp.safetensors"


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "elements": self.elements,
        }


@dataclass(frozen=True)
class MTPInspection:
    mtp_file: str
    exists: bool
    tensor_count: int = 0
    sidecar_format: str = "bf16"
    expected_tensor_count: int = EXPECTED_MTP_TENSOR_COUNT
    tensors: tuple[TensorInfo, ...] = ()
    missing_expected_keys: tuple[str, ...] = ()
    extra_keys: tuple[str, ...] = ()

    @property
    def passes_tensor_gate(self) -> bool:
        return (
            self.exists
            and self.tensor_count == self.expected_tensor_count
            and not self.missing_expected_keys
            and not self.extra_keys
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mtp_file": self.mtp_file,
            "exists": self.exists,
            "tensor_count": self.tensor_count,
            "sidecar_format": self.sidecar_format,
            "expected_tensor_count": self.expected_tensor_count,
            "passes_tensor_gate": self.passes_tensor_gate,
            "missing_expected_keys": list(self.missing_expected_keys),
            "extra_keys": list(self.extra_keys),
            "tensors": [t.to_dict() for t in self.tensors],
        }


@dataclass(frozen=True)
class ModelInspection:
    model_dir: str
    config_exists: bool
    architecture: str | None
    model_type: str | None
    mtp_num_hidden_layers: int
    hidden_size: int | None
    num_hidden_layers: int | None
    vocab_size: int | None
    quantization: dict[str, Any] = field(default_factory=dict)
    sidecars: dict[str, bool] = field(default_factory=dict)
    model_files: tuple[str, ...] = ()
    mtp: MTPInspection | None = None
    compatibility: dict[str, Any] = field(default_factory=dict)

    @property
    def passes_primary_gate(self) -> bool:
        if self.compatibility:
            return self.compatibility.get("tier") == "verified"
        return (
            self.config_exists
            and (self.model_type or "").startswith("qwen3_5")
            and self.mtp_num_hidden_layers == 1
            and self.mtp is not None
            and self.mtp.passes_tensor_gate
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "config_exists": self.config_exists,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "quantization": self.quantization,
            "sidecars": self.sidecars,
            "model_files": list(self.model_files),
            "passes_primary_gate": self.passes_primary_gate,
            "mtp": self.mtp.to_dict() if self.mtp else None,
            "compatibility": self.compatibility,
            "mtp_supported": self.compatibility.get("mtp_supported"),
            "mtp_arch": self.compatibility.get("arch_id"),
            "recommended_backend": self.compatibility.get("recommended_backend"),
            "recommended_profile": self.compatibility.get("recommended_profile"),
            "runtime_compatibility": self.compatibility.get("runtime_compatibility"),
            "unverified_model": self.compatibility.get("unverified_model", False),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def inspect_mtp_tensors(model_dir: Path | str, config: dict[str, Any] | None = None) -> MTPInspection:
    mtp_path = expected_mtp_file(model_dir, config)
    mtp_quant = (config or {}).get("mtplx_mtp_quantization", {})
    prequantized = isinstance(mtp_quant, dict) and bool(mtp_quant.get("prequantized"))
    expected_keys = set(EXPECTED_PREQUANTIZED_MTP_KEYS if prequantized else EXPECTED_MTP_KEYS)
    expected_count = (
        EXPECTED_PREQUANTIZED_MTP_TENSOR_COUNT if prequantized else EXPECTED_MTP_TENSOR_COUNT
    )
    sidecar_format = "prequantized-mlx-affine" if prequantized else "bf16"
    if not mtp_path.exists():
        return MTPInspection(
            mtp_file=str(mtp_path),
            exists=False,
            sidecar_format=sidecar_format,
            expected_tensor_count=expected_count,
        )

    from safetensors import safe_open

    tensors: list[TensorInfo] = []
    with safe_open(str(mtp_path), framework="np") as handle:
        keys = sorted(handle.keys())
        for key in keys:
            sl = handle.get_slice(key)
            tensors.append(
                TensorInfo(
                    key=key,
                    dtype=str(sl.get_dtype()),
                    shape=tuple(int(x) for x in sl.get_shape()),
                )
            )

    key_set = {t.key for t in tensors}
    return MTPInspection(
        mtp_file=str(mtp_path),
        exists=True,
        tensor_count=len(tensors),
        sidecar_format=sidecar_format,
        expected_tensor_count=expected_count,
        tensors=tuple(tensors),
        missing_expected_keys=tuple(sorted(expected_keys - key_set)),
        extra_keys=tuple(sorted(key_set - expected_keys)),
    )


def inspect_model(model_dir: Path | str) -> ModelInspection:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    config_exists = config_path.exists()
    config: dict[str, Any] = json.loads(config_path.read_text()) if config_exists else {}
    tcfg = text_config(config)
    archs = config.get("architectures") or tcfg.get("architectures") or []
    architecture = archs[0] if archs else None
    quant = config.get("quantization_config") or config.get("quantization") or {}
    if not quant:
        quant = tcfg.get("quantization_config") or tcfg.get("quantization") or {}

    mtp = inspect_mtp_tensors(model_path, config) if config_exists else None
    inspection = ModelInspection(
        model_dir=str(model_path),
        config_exists=config_exists,
        architecture=architecture,
        model_type=tcfg.get("model_type") or config.get("model_type"),
        mtp_num_hidden_layers=int(
            tcfg.get("mtp_num_hidden_layers")
            or tcfg.get("num_nextn_predict_layers")
            or config.get("num_nextn_predict_layers")
            or 0
        ),
        hidden_size=tcfg.get("hidden_size"),
        num_hidden_layers=tcfg.get("num_hidden_layers"),
        vocab_size=tcfg.get("vocab_size"),
        quantization=quant,
        sidecars={name: (model_path / name).exists() for name in MULTIMODAL_SIDECARS},
        model_files=tuple(sorted(p.name for p in model_path.glob("model*.safetensors"))),
        mtp=mtp,
    )
    from mtplx.backends.registry import compatibility_for_inspection

    compatibility = compatibility_for_inspection(inspection).to_dict()
    return ModelInspection(
        model_dir=inspection.model_dir,
        config_exists=inspection.config_exists,
        architecture=inspection.architecture,
        model_type=inspection.model_type,
        mtp_num_hidden_layers=inspection.mtp_num_hidden_layers,
        hidden_size=inspection.hidden_size,
        num_hidden_layers=inspection.num_hidden_layers,
        vocab_size=inspection.vocab_size,
        quantization=inspection.quantization,
        sidecars=inspection.sidecars,
        model_files=inspection.model_files,
        mtp=inspection.mtp,
        compatibility=compatibility,
    )


def require_primary_mtp_artifact(model_dir: Path | str) -> ModelInspection:
    inspection = inspect_model(model_dir)
    if not inspection.passes_primary_gate:
        raise RuntimeError(inspection.to_json())
    return inspection
