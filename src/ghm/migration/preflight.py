"""Validate the shared remote multimodal runtime without downloading models."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


MINIMUM_VERSIONS = {
    "transformers": "4.57.0",
    "accelerate": "1.13.0",
}


def collect_preflight(
    *,
    data_root: Path,
    model_paths: list[Path] | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Return only environment and aggregate readiness information."""

    if model_paths is None:
        model_paths = [model_path] if model_path is not None else []
    if model_path is not None and model_paths != [model_path]:
        raise ValueError("use model_path or model_paths, not both")
    packages: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for import_name, display_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("PIL", "pillow"),
        ("pyarrow", "pyarrow"),
        ("modelscope", "modelscope"),
    ):
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            packages[display_name] = {"installed": True, "version": str(version)}
        except ModuleNotFoundError:
            packages[display_name] = {"installed": False, "version": None}
            failures.append(f"missing_package:{display_name}")

    for name, expected in MINIMUM_VERSIONS.items():
        actual = packages.get(name, {}).get("version")
        normalized = str(actual).split("+", 1)[0] if actual is not None else None
        if normalized is not None and _version_tuple(normalized) < _version_tuple(expected):
            failures.append(f"version_too_old:{name}")

    transformers = sys.modules.get("transformers")
    loader_support = {
        "image_text": bool(
            transformers is not None
            and hasattr(transformers, "AutoModelForImageTextToText")
        ),
        "multimodal": bool(
            transformers is not None
            and hasattr(transformers, "AutoModelForMultimodalLM")
        ),
    }
    for loader, supported in loader_support.items():
        if not supported:
            failures.append(f"missing_transformers_loader:{loader}")

    python_supported = sys.version_info[:2] in {(3, 10), (3, 11)}
    if not python_supported:
        failures.append("unsupported_python_version")

    cuda_available = False
    gpu_count = 0
    torch = sys.modules.get("torch")
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    if not cuda_available:
        failures.append("cuda_unavailable")

    models: list[dict[str, Any]] = []
    for index, model_path in enumerate(model_paths):
        exists = model_path.is_dir()
        config_exists = (model_path / "config.json").is_file()
        if not exists:
            failures.append(f"model_directory_missing:{index}")
            failures.append("model_directory_missing")
        elif not config_exists:
            failures.append(f"model_config_missing:{index}")
            failures.append("model_config_missing")
        models.append(
            {
                "index": index,
                "directory_exists": exists,
                "config_exists": config_exists,
            }
        )

    data_exists = data_root.is_dir()
    if not data_exists:
        failures.append("data_root_missing")
        free_bytes = None
    else:
        free_bytes = shutil.disk_usage(data_root).free

    return {
        "preflight_schema_version": 1,
        "ready": not failures,
        "failures": sorted(failures),
        "python_version": platform.python_version(),
        "python_supported": python_supported,
        "platform_system": platform.system(),
        "packages": packages,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "minimum_versions": MINIMUM_VERSIONS,
        "transformers_loader_support": loader_support,
        "models": models,
        "data_root_exists": data_exists,
        "data_root_free_bytes": free_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = collect_preflight(data_root=args.data_root, model_paths=args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    print(
        "Multimodal preflight: "
        f"ready={result['ready']}, cuda={result['cuda_available']}, "
        f"gpu_count={result['gpu_count']}, failures={len(result['failures'])}"
    )
    return 0 if result["ready"] else 1


def _version_tuple(value: str) -> tuple[int, ...]:
    """Parse the numeric release prefix without importing packaging."""

    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


if __name__ == "__main__":
    raise SystemExit(main())
