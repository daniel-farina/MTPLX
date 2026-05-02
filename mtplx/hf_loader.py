"""Hugging Face model resolution and local cache helpers."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.artifacts import _hf_repo_id_from_ref


DEFAULT_MODEL_CACHE = Path("~/.mtplx/models").expanduser()


def model_cache_dir(value: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("MTPLX_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_MODEL_CACHE


def safe_model_name(repo_id: str) -> str:
    return repo_id.strip("/").replace("/", "--")


def repo_id_from_model_ref(value: str) -> str | None:
    return _hf_repo_id_from_ref(value)


def cached_model_path(repo_id: str, *, cache_dir: str | Path | None = None) -> Path:
    return model_cache_dir(cache_dir) / safe_model_name(repo_id)


def resolve_model_path(model_ref: str, *, cache_dir: str | Path | None = None) -> Path:
    local = Path(model_ref).expanduser()
    if local.exists():
        return local
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        return local
    cached = cached_model_path(repo_id, cache_dir=cache_dir)
    if cached.exists():
        return cached
    raise FileNotFoundError(
        f"Model {repo_id} is not cached. Run: mtplx pull {repo_id}"
    )


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


@dataclass(frozen=True)
class CachedModel:
    repo_id: str
    path: Path
    size_bytes: int
    has_runtime_contract: bool
    has_config: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / 1_000_000_000, 3),
            "has_runtime_contract": self.has_runtime_contract,
            "has_config": self.has_config,
        }


def list_cached_models(*, cache_dir: str | Path | None = None) -> list[CachedModel]:
    root = model_cache_dir(cache_dir)
    if not root.exists():
        return []
    rows: list[CachedModel] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        repo_id = child.name.replace("--", "/")
        rows.append(
            CachedModel(
                repo_id=repo_id,
                path=child,
                size_bytes=directory_size_bytes(child),
                has_runtime_contract=(child / "mtplx_runtime.json").exists(),
                has_config=(child / "config.json").exists(),
            )
        )
    return rows


def pull_model(
    model_ref: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref)
    if repo_id is None:
        raise ValueError(f"pull requires a Hugging Face repo id or URL, got: {model_ref}")
    root = model_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = cached_model_path(repo_id, cache_dir=root)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is required for mtplx pull: {exc}") from exc

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=str(destination),
    )
    resolved = Path(path)
    return {
        "repo_id": repo_id,
        "path": str(resolved),
        "cache_dir": str(root),
        "revision": revision,
        "size_bytes": directory_size_bytes(resolved),
        "has_runtime_contract": (resolved / "mtplx_runtime.json").exists(),
        "has_config": (resolved / "config.json").exists(),
    }


def remove_cached_model(model_ref: str, *, cache_dir: str | Path | None = None) -> dict[str, Any]:
    repo_id = repo_id_from_model_ref(model_ref) or model_ref.replace("--", "/")
    path = cached_model_path(repo_id, cache_dir=cache_dir)
    existed = path.exists()
    size = directory_size_bytes(path) if existed else 0
    if existed:
        shutil.rmtree(path)
    return {
        "repo_id": repo_id,
        "path": str(path),
        "removed": existed,
        "size_bytes_removed": size,
    }


def hf_cache_report(*, cache_dir: str | Path | None = None) -> dict[str, Any]:
    root = model_cache_dir(cache_dir)
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    token_source = "environment" if token_present else None
    if not token_present:
        try:
            from huggingface_hub import get_token

            token_present = bool(get_token())
            token_source = "huggingface_hub" if token_present else None
        except Exception:
            token_present = False
    return {
        "cache_dir": str(root),
        "cache_exists": root.exists(),
        "cache_writable": os.access(root if root.exists() else root.parent, os.W_OK),
        "cached_models": len(list_cached_models(cache_dir=root)),
        "token_present": token_present,
        "token_source": token_source,
    }
