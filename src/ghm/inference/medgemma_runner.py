"""Run MedGemma image-text inference over model-input prompt JSONL.

This runner is intended for the remote GPU machine. It keeps the same raw
response schema as the mock runner so existing parsing and scoring code can be
reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl


SPLITS = ["g1_h1", "g1_h2", "g2_h1", "g2_h2"]


def resolve_image_path(image_path: Any, data_root: Path | None) -> Path | None:
    """Resolve a prompt image_path against the remote data root."""

    if image_path is None:
        return None
    path = Path(str(image_path))
    if path.is_absolute():
        return path
    if data_root is not None:
        return data_root / path
    return path


def metadata_by_item_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index eval metadata rows by item_id."""

    return {str(row["item_id"]): row for row in rows if row.get("item_id") is not None}


def build_error_row(
    record: dict[str, Any],
    metadata: dict[str, Any],
    *,
    model_name: str,
    model_version: str | None,
    generation_config: dict[str, Any],
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Build a schema-compatible row for missing images or inference failures."""

    return {
        "item_id": record.get("item_id"),
        "model_name": model_name,
        "model_version": model_version,
        "image_path": record.get("image_path"),
        "prompt_template_id": record.get("prompt_template_id"),
        "prompt": record.get("prompt"),
        "raw_response": "",
        "parsed_answer": None,
        "parse_status": "unparsed",
        "generation_config": generation_config,
        "runtime": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": generation_config.get("device"),
            "status": "error",
            "error_type": error_type,
            "error_message": error_message,
        },
        "answer_label": metadata.get("answer_label"),
        "granularity": metadata.get("granularity"),
        "question_type": metadata.get("question_type"),
        "hallucination_probe": metadata.get("hallucination_probe"),
    }


def dry_run_rows(
    prompt_records: list[dict[str, Any]],
    eval_metadata_rows: list[dict[str, Any]],
    *,
    data_root: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate prompt, metadata, and image paths without loading the model."""

    metadata_index = metadata_by_item_id(eval_metadata_rows)
    rows: list[dict[str, Any]] = []
    missing_metadata = 0
    missing_image_path = 0
    existing_images = 0
    missing_images = 0
    for record in prompt_records:
        item_id = str(record.get("item_id"))
        metadata = metadata_index.get(item_id, {})
        if not metadata:
            missing_metadata += 1
        resolved = resolve_image_path(record.get("image_path"), data_root)
        exists = bool(resolved and resolved.exists())
        if resolved is None:
            missing_image_path += 1
        elif exists:
            existing_images += 1
        else:
            missing_images += 1
        rows.append(
            {
                "item_id": record.get("item_id"),
                "image_path": record.get("image_path"),
                "resolved_image_path": str(resolved) if resolved else None,
                "image_exists": exists,
                "has_eval_metadata": bool(metadata),
            }
        )
    return rows, {
        "prompt_records": len(prompt_records),
        "missing_metadata": missing_metadata,
        "missing_image_path": missing_image_path,
        "existing_images": existing_images,
        "missing_images": missing_images,
    }


def run_medgemma(
    prompt_records: list[dict[str, Any]],
    eval_metadata_rows: list[dict[str, Any]],
    *,
    model_path: Path,
    data_root: Path | None,
    model_name: str,
    device: str,
    dtype: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Run MedGemma generation and return raw response rows."""

    if batch_size != 1:
        raise ValueError("MedGemma runner currently supports --batch-size 1 only")

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MedGemma inference requires torch, pillow, and transformers installed "
            "on the remote environment."
        ) from exc

    torch_dtype = _torch_dtype(torch, dtype)
    processor = AutoProcessor.from_pretrained(str(model_path))
    load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kwargs)
    if device in {"cuda", "cpu"}:
        model.to(device)
    model.eval()

    metadata_index = metadata_by_item_id(eval_metadata_rows)
    selected_records = prompt_records[:limit] if limit is not None else prompt_records
    generation_config = {
        "model_path": str(model_path),
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    outputs: list[dict[str, Any]] = []

    for index, record in enumerate(selected_records, start=1):
        started = time.time()
        metadata = metadata_index.get(str(record.get("item_id")), {})
        image_path = resolve_image_path(record.get("image_path"), data_root)
        if image_path is None:
            outputs.append(
                build_error_row(
                    record,
                    metadata,
                    model_name=model_name,
                    model_version=str(model_path),
                    generation_config=generation_config,
                    error_type="missing_image_path",
                    error_message="image_path is missing",
                )
            )
            continue
        if not image_path.exists():
            outputs.append(
                build_error_row(
                    record,
                    metadata,
                    model_name=model_name,
                    model_version=str(model_path),
                    generation_config=generation_config,
                    error_type="image_not_found",
                    error_message="resolved image path does not exist",
                )
            )
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            raw_response = generate_one(
                model=model,
                processor=processor,
                image=image,
                prompt=str(record.get("prompt", "")),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                torch_dtype=torch_dtype,
            )
            outputs.append(
                {
                    "item_id": record.get("item_id"),
                    "model_name": model_name,
                    "model_version": str(model_path),
                    "image_path": record.get("image_path"),
                    "prompt_template_id": record.get("prompt_template_id"),
                    "prompt": record.get("prompt"),
                    "raw_response": raw_response,
                    "parsed_answer": None,
                    "parse_status": "unparsed",
                    "generation_config": generation_config,
                    "runtime": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "device": device,
                        "status": "success",
                        "latency_seconds": round(time.time() - started, 6),
                        "row_index": index,
                    },
                    "answer_label": metadata.get("answer_label"),
                    "granularity": metadata.get("granularity"),
                    "question_type": metadata.get("question_type"),
                    "hallucination_probe": metadata.get("hallucination_probe"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch robust on remote runs.
            outputs.append(
                build_error_row(
                    record,
                    metadata,
                    model_name=model_name,
                    model_version=str(model_path),
                    generation_config=generation_config,
                    error_type=exc.__class__.__name__,
                    error_message=str(exc)[:500],
                )
            )

    return outputs


def generate_one(
    *,
    model: Any,
    processor: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    torch_dtype: Any,
) -> str:
    """Generate one MedGemma response."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    moved_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            moved_inputs[key] = value.to(model_device, dtype=torch_dtype)
        elif hasattr(value, "to"):
            moved_inputs[key] = value.to(model_device)
        else:
            moved_inputs[key] = value
    inputs = moved_inputs
    do_sample = temperature > 0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    generated = model.generate(**inputs, **generate_kwargs)
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = generated[:, input_length:]
    return processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run MedGemma on model-input JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--model-name", default="medgemma")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prompt and image paths without loading the model.",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive when provided.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative.")

    prompt_records = read_jsonl(args.input)
    eval_metadata_rows = read_jsonl(args.eval_metadata)
    if args.limit is not None:
        prompt_records = prompt_records[: args.limit]

    if args.dry_run:
        rows, summary = dry_run_rows(
            prompt_records,
            eval_metadata_rows,
            data_root=args.data_root,
        )
        write_jsonl(rows, args.output)
        print(
            "MedGemma dry run: "
            f"prompt_records={summary['prompt_records']}, "
            f"existing_images={summary['existing_images']}, "
            f"missing_images={summary['missing_images']}, "
            f"missing_metadata={summary['missing_metadata']}"
        )
        return 0

    try:
        outputs = run_medgemma(
            prompt_records,
            eval_metadata_rows,
            model_path=args.model_path,
            data_root=args.data_root,
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            limit=None,
        )
    except (RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    write_jsonl(outputs, args.output)
    success = sum(1 for row in outputs if row.get("runtime", {}).get("status") == "success")
    failed = len(outputs) - success
    print(
        "Ran MedGemma inference: "
        f"inputs={len(prompt_records)}, outputs={len(outputs)}, "
        f"success={success}, failed={failed}"
    )
    return 0


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


if __name__ == "__main__":
    raise SystemExit(main())
