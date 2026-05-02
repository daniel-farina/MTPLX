"""Architecture compatibility registry and runtime-contract checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mtplx.profiles import DEFAULT_PROFILE_NAME, PROFILE_CHOICES


RUNTIME_CONTRACT_FILE = "mtplx_runtime.json"
SUPPORTED_ARCH_IDS = {"qwen3-next-mtp"}

TIER_VERIFIED = "verified"
TIER_ARCH_COMPATIBLE_UNVERIFIED = "architecture-compatible-but-unverified"
TIER_INCOMPATIBLE_ARCHITECTURE = "incompatible-architecture"
TIER_NO_MTP = "no-MTP"

EXIT_VERIFIED = 0
EXIT_NO_MTP = 2
EXIT_UNVERIFIED = 3
EXIT_INCOMPATIBLE_ARCHITECTURE = 4


class ModelCompatibilityError(RuntimeError):
    exit_code = 1


class UnverifiedArchitectureError(ModelCompatibilityError):
    exit_code = EXIT_UNVERIFIED


class IncompatibleArchitectureError(ModelCompatibilityError):
    exit_code = EXIT_INCOMPATIBLE_ARCHITECTURE


class NoMTPError(ModelCompatibilityError):
    exit_code = EXIT_NO_MTP


@dataclass(frozen=True)
class RuntimeContract:
    mtplx_version: str
    arch_id: str
    mtp_depth_max: int
    recommended_profile: str
    exactness_baseline: dict[str, Any]
    verified_on: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeContract":
        missing = [
            key
            for key in (
                "mtplx_version",
                "arch_id",
                "mtp_depth_max",
                "recommended_profile",
                "exactness_baseline",
                "verified_on",
            )
            if key not in data
        ]
        if missing:
            raise ValueError(f"runtime contract missing required keys: {', '.join(missing)}")
        profile = str(data["recommended_profile"])
        if profile not in PROFILE_CHOICES:
            raise ValueError(f"runtime contract has invalid recommended_profile: {profile}")
        depth = int(data["mtp_depth_max"])
        if depth <= 0:
            raise ValueError("runtime contract mtp_depth_max must be positive")
        return cls(
            mtplx_version=str(data["mtplx_version"]),
            arch_id=str(data["arch_id"]),
            mtp_depth_max=depth,
            recommended_profile=profile,
            exactness_baseline=dict(data["exactness_baseline"]),
            verified_on=dict(data["verified_on"]),
            raw=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mtplx_version": self.mtplx_version,
            "arch_id": self.arch_id,
            "mtp_depth_max": self.mtp_depth_max,
            "recommended_profile": self.recommended_profile,
            "exactness_baseline": self.exactness_baseline,
            "verified_on": self.verified_on,
        }


@dataclass(frozen=True)
class CompatibilityVerdict:
    tier: str
    arch_id: str | None
    supported: bool
    can_run: bool
    exit_code: int
    message: str
    recommended_backend: str | None = None
    recommended_profile: str | None = None
    runtime_contract: RuntimeContract | None = None
    runtime_contract_path: str | None = None
    runtime_contract_error: str | None = None
    unsafe_force_required: bool = False
    unverified_model: bool = False
    mtp_supported: str = "no"
    runtime_compatibility: str = "unsupported"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "arch_id": self.arch_id,
            "supported": self.supported,
            "can_run": self.can_run,
            "exit_code": self.exit_code,
            "message": self.message,
            "recommended_backend": self.recommended_backend,
            "recommended_profile": self.recommended_profile,
            "runtime_contract": (
                self.runtime_contract.to_dict() if self.runtime_contract else None
            ),
            "runtime_contract_path": self.runtime_contract_path,
            "runtime_contract_error": self.runtime_contract_error,
            "unsafe_force_required": self.unsafe_force_required,
            "unverified_model": self.unverified_model,
            "mtp_supported": self.mtp_supported,
            "runtime_compatibility": self.runtime_compatibility,
        }


def _contract_path(model_dir: Path) -> Path:
    return model_dir / RUNTIME_CONTRACT_FILE


def load_runtime_contract(model_dir: Path | str) -> tuple[RuntimeContract | None, str | None]:
    path = _contract_path(Path(model_dir))
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RuntimeContract.from_dict(data), None
    except Exception as exc:
        return None, str(exc)


def _text(value: Any) -> str:
    return str(value or "").lower().replace("-", "_")


def _detect_arch_id(inspection: Any) -> str | None:
    architecture = _text(getattr(inspection, "architecture", None))
    model_type = _text(getattr(inspection, "model_type", None))
    combined = f"{architecture} {model_type}"
    if "qwen3_next" in combined or "qwen3_5" in combined or "qwen3_6" in combined:
        return "qwen3-next-mtp"
    if "deepseek" in combined:
        return "deepseek-v3-mtp"
    if "llama" in combined:
        return "llama-mtp"
    if "mtp" in combined or "nextn" in combined:
        return "generic-mtp"
    return None


def _has_mtp_markers(inspection: Any) -> bool:
    mtp = getattr(inspection, "mtp", None)
    return bool(
        int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0) > 0
        or (mtp is not None and bool(getattr(mtp, "exists", False)))
    )


def compatibility_for_inspection(inspection: Any) -> CompatibilityVerdict:
    model_dir = Path(getattr(inspection, "model_dir", "."))
    contract, contract_error = load_runtime_contract(model_dir)
    detected_arch_id = _detect_arch_id(inspection)
    has_mtp = _has_mtp_markers(inspection)
    tensor_gate = bool(getattr(getattr(inspection, "mtp", None), "passes_tensor_gate", False))
    contract_path = str(_contract_path(model_dir)) if _contract_path(model_dir).exists() else None

    if contract is not None:
        arch_id = contract.arch_id
        if arch_id in SUPPORTED_ARCH_IDS and has_mtp and tensor_gate:
            return CompatibilityVerdict(
                tier=TIER_VERIFIED,
                arch_id=arch_id,
                supported=True,
                can_run=True,
                exit_code=EXIT_VERIFIED,
                message="Verified MTPLX runtime contract found.",
                recommended_backend="qwen3_next",
                recommended_profile=contract.recommended_profile,
                runtime_contract=contract,
                runtime_contract_path=contract_path,
                mtp_supported="yes",
                runtime_compatibility="native",
            )
        if arch_id not in SUPPORTED_ARCH_IDS:
            return CompatibilityVerdict(
                tier=TIER_INCOMPATIBLE_ARCHITECTURE,
                arch_id=arch_id,
                supported=False,
                can_run=False,
                exit_code=EXIT_INCOMPATIBLE_ARCHITECTURE,
                message=(
                    f"{arch_id} runtime contract detected; not supported in "
                    "v0.1.0-preview. Planned for a later backend."
                ),
                runtime_contract=contract,
                runtime_contract_path=contract_path,
                mtp_supported="partial" if has_mtp else "no",
                runtime_compatibility="unsupported",
            )
        return CompatibilityVerdict(
            tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
            arch_id=arch_id,
            supported=False,
            can_run=False,
            exit_code=EXIT_UNVERIFIED,
            message=(
                "Runtime contract exists but local MTP tensor inspection did not "
                "pass; refusing to run without repair."
            ),
            recommended_backend="qwen3_next",
            recommended_profile=contract.recommended_profile,
            runtime_contract=contract,
            runtime_contract_path=contract_path,
            runtime_contract_error=contract_error,
            unsafe_force_required=True,
            unverified_model=True,
            mtp_supported="partial",
            runtime_compatibility="needs-grafting",
        )

    if contract_error:
        return CompatibilityVerdict(
            tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
            arch_id=detected_arch_id,
            supported=False,
            can_run=False,
            exit_code=EXIT_UNVERIFIED,
            message=f"Invalid {RUNTIME_CONTRACT_FILE}: {contract_error}",
            recommended_backend="qwen3_next" if detected_arch_id == "qwen3-next-mtp" else None,
            runtime_contract_path=contract_path,
            runtime_contract_error=contract_error,
            unsafe_force_required=True,
            unverified_model=True,
            mtp_supported="partial" if has_mtp else "no",
            runtime_compatibility="needs-grafting" if has_mtp else "unsupported",
        )

    if not has_mtp:
        return CompatibilityVerdict(
            tier=TIER_NO_MTP,
            arch_id=detected_arch_id,
            supported=False,
            can_run=False,
            exit_code=EXIT_NO_MTP,
            message=(
                "Model has no MTP head. MTPLX requires an MTP-equipped model."
            ),
            mtp_supported="no",
            runtime_compatibility="unsupported",
        )

    if detected_arch_id == "qwen3-next-mtp":
        return CompatibilityVerdict(
            tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
            arch_id=detected_arch_id,
            supported=False,
            can_run=False,
            exit_code=EXIT_UNVERIFIED,
            message=(
                "Qwen3-Next MTP markers detected, but no mtplx_runtime.json "
                "verified contract is present. Use --unsafe-force-unverified "
                "--yes to proceed without support guarantees."
            ),
            recommended_backend="qwen3_next",
            recommended_profile=DEFAULT_PROFILE_NAME,
            unsafe_force_required=True,
            unverified_model=True,
            mtp_supported="partial",
            runtime_compatibility="needs-grafting",
        )

    return CompatibilityVerdict(
        tier=TIER_INCOMPATIBLE_ARCHITECTURE,
        arch_id=detected_arch_id or "generic-mtp",
        supported=False,
        can_run=False,
        exit_code=EXIT_INCOMPATIBLE_ARCHITECTURE,
        message=(
            f"{detected_arch_id or 'generic MTP'} detected; not supported in "
            "v0.1.0-preview. Qwen3-Next MTP is the only running backend."
        ),
        mtp_supported="partial",
        runtime_compatibility="unsupported",
    )


def require_verified_or_raise(
    inspection: Any,
    *,
    unsafe_force_unverified: bool = False,
    yes: bool = False,
) -> CompatibilityVerdict:
    verdict = compatibility_for_inspection(inspection)
    if verdict.can_run:
        return verdict
    if (
        unsafe_force_unverified
        and yes
        and verdict.tier == TIER_ARCH_COMPATIBLE_UNVERIFIED
    ):
        return verdict
    if verdict.tier == TIER_NO_MTP:
        raise NoMTPError(verdict.message)
    if verdict.tier == TIER_INCOMPATIBLE_ARCHITECTURE:
        raise IncompatibleArchitectureError(verdict.message)
    raise UnverifiedArchitectureError(verdict.message)
