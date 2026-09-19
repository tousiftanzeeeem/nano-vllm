"""Discover models the visualizer can run, and say plainly why one cannot."""

import json
import os
from dataclasses import dataclass, asdict
from importlib.util import find_spec
from pathlib import Path


MOCK_ID = "__mock__"

SEARCH_DIRS = [
    Path.home() / "huggingface",
    Path.home() / "models",
    Path(os.environ.get("NANOVLLM_MODELS", "")) if os.environ.get("NANOVLLM_MODELS") else None,
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
]


@dataclass
class ModelEntry:
    id: str
    label: str
    path: str | None
    kind: str           # "mock" | "local"
    available: bool
    reason: str         # why it cannot run, "" when it can
    params: str = ""


def _missing_runtime() -> str:
    """Return a human explanation if the real engine cannot run here."""
    missing = [m for m in ("torch", "transformers", "triton", "flash_attn") if not find_spec(m)]
    if missing:
        return f"missing {', '.join(missing)}"
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA device"
    except Exception as exc:
        return f"torch unusable ({exc})"
    return ""


def _describe(path: Path) -> tuple[str, str]:
    """Return (architecture label, parameter hint) from config.json."""
    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    arch = (cfg.get("architectures") or [""])[0]
    h = cfg.get("hidden_size")
    n = cfg.get("num_hidden_layers")
    hint = f"{n}L·{h}d" if h and n else ""
    return arch, hint


def _snapshot_dir(repo: Path) -> Path | None:
    """Resolve a HF hub cache repo dir to its newest snapshot."""
    snaps = repo / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [d for d in snaps.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def discover() -> list[ModelEntry]:
    runtime_problem = _missing_runtime()
    entries = [
        ModelEntry(
            id=MOCK_ID,
            label="Mock engine — no GPU, no weights",
            path=None,
            kind="mock",
            available=True,
            reason="",
            params="real scheduler & block manager, fake forward pass",
        )
    ]

    seen: set[str] = set()
    for base in SEARCH_DIRS:
        if base is None or not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            path = child
            name = child.name
            if name.startswith("models--"):
                resolved = _snapshot_dir(child)
                if resolved is None:
                    continue
                path, name = resolved, name.removeprefix("models--").replace("--", "/")
            if not (path / "config.json").exists():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            arch, hint = _describe(path)
            supported = arch == "Qwen3ForCausalLM"
            reason = runtime_problem or ("" if supported else f"unsupported architecture {arch or '?'}")
            entries.append(
                ModelEntry(
                    id=key,
                    label=name,
                    path=key,
                    kind="local",
                    available=not reason,
                    reason=reason,
                    params=" · ".join(p for p in (arch, hint) if p),
                )
            )
    return entries


def to_json() -> dict:
    entries = discover()
    return {
        "models": [asdict(e) for e in entries],
        "runtimeProblem": _missing_runtime(),
        "searched": [str(p) for p in SEARCH_DIRS if p is not None],
    }


def resolve(model_id: str) -> ModelEntry:
    for e in discover():
        if e.id == model_id:
            return e
    # allow an arbitrary path typed into the UI
    path = Path(model_id).expanduser()
    if (path / "config.json").exists():
        arch, hint = _describe(path)
        reason = _missing_runtime() or ("" if arch == "Qwen3ForCausalLM" else f"unsupported architecture {arch or '?'}")
        return ModelEntry(str(path), path.name, str(path), "local", not reason, reason, hint)
    raise ValueError(f"no model at {model_id!r}")
