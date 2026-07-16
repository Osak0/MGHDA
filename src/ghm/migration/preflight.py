"""Validate a remote MedGemma runtime without downloading models."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


EXPECTED_VERSIONS = {
    "torch": "2.5.1",
    "transformers": "4.50.3",
    "accelerate": "1.13.0",
}


def collect_preflight(*, data_root: Path, model_path: Path) -> dict[str, Any]:
    """Return only environment and aggregate readiness information."""

    packages: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for import_name, display_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("PIL", "pillow"),
        ("pyarrow", "pyarrow"),
    ):
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            packages[display_name] = {"installed": True, "version": str(version)}
        except ModuleNotFoundError:
            packages[display_name] = {"installed": False, "version": None}
            failures.append(f"missing_package:{display_name}")

    for name, expected in EXPECTED_VERSIONS.items():
        actual = packages.get(name, {}).get("version")
        normalized = str(actual).split("+", 1)[0] if actual is not None else None
        if normalized is not None and normalized != expected:
            failures.append(f"version_mismatch:{name}")

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

    model_exists = model_path.is_dir()
    model_config_exists = (model_path / "config.json").is_file()
    if not model_exists:
        failures.append("model_directory_missing")
    elif not model_config_exists:
        failures.append("model_config_missing")

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
        "model_directory_exists": model_exists,
        "model_config_exists": model_config_exists,
        "data_root_exists": data_exists,
        "data_root_free_bytes": free_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = collect_preflight(data_root=args.data_root, model_path=args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    print(
        "MedGemma preflight: "
        f"ready={result['ready']}, cuda={result['cuda_available']}, "
        f"gpu_count={result['gpu_count']}, failures={len(result['failures'])}"
    )
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
