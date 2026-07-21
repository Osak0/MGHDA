"""Run a local Hugging Face image-text model over model-input prompt JSONL.

This runner is intended for the remote GPU machine. It keeps the same raw
response schema as the mock runner so existing parsing and scoring code can be
reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ghm.granularity.common import read_jsonl, write_jsonl


SPLITS = ["study2_g1", "study2_g2"]


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
        "model_id": model_name,
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
        "claim_polarity": metadata.get("claim_polarity"),
        "evidence_state": metadata.get("evidence_state"),
    }


def dry_run_rows(
    prompt_records: list[dict[str, Any]],
    eval_metadata_rows: list[dict[str, Any]],
    *,
    data_root: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate prompt, metadata, and image paths without loading the model."""

    metadata_index = metadata_by_item_id(eval_metadata_rows)
    prompt_ids = [
        str(record["item_id"])
        for record in prompt_records
        if record.get("item_id") is not None
    ]
    metadata_ids = [
        str(record["item_id"])
        for record in eval_metadata_rows
        if record.get("item_id") is not None
    ]
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
        "eval_metadata_records": len(eval_metadata_rows),
        "duplicate_prompt_item_ids": len(prompt_ids) - len(set(prompt_ids)),
        "duplicate_metadata_item_ids": len(metadata_ids) - len(set(metadata_ids)),
        "unexpected_metadata": len(set(metadata_ids) - set(prompt_ids)),
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
    seed: int = 42,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run deterministic local multimodal generation and return raw rows."""

    if batch_size != 1:
        raise ValueError("multimodal runner currently supports --batch-size 1 only")

    try:
        import torch
        from PIL import Image
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Multimodal inference requires torch, pillow, and transformers installed "
            "on the remote environment."
        ) from exc

    torch_dtype = _torch_dtype(torch, dtype)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": True,
    }
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    model_class = (
        Qwen3VLForConditionalGeneration
        if model_name == "Qwen/Qwen3-VL-8B-Instruct"
        else AutoModelForImageTextToText
    )
    model = model_class.from_pretrained(str(model_path), **load_kwargs)
    if device in {"cuda", "cpu"}:
        model.to(device)
    model.eval()
    _seed_torch(torch, seed)

    metadata_index = metadata_by_item_id(eval_metadata_rows)
    selected_records = prompt_records[:limit] if limit is not None else prompt_records
    generation_config = {
        "model_path": str(model_path),
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    checkpoint_rows = _latest_rows_by_item_id(checkpoint_path) if resume else {}
    if checkpoint_path is not None and not resume:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("", encoding="utf-8")
    outputs: list[dict[str, Any]] = []

    for index, record in enumerate(selected_records, start=1):
        item_id = str(record.get("item_id"))
        previous = checkpoint_rows.get(item_id)
        if previous is not None and previous.get("runtime", {}).get("status") == "success":
            _validate_resumed_row(
                previous,
                record,
                model_name=model_name,
                generation_config=generation_config,
            )
            outputs.append(previous)
            continue
        started = time.time()
        metadata = metadata_index.get(str(record.get("item_id")), {})
        image_path = resolve_image_path(record.get("image_path"), data_root)
        if image_path is None:
            row = build_error_row(
                    record,
                    metadata,
                    model_name=model_name,
                    model_version=str(model_path),
                    generation_config=generation_config,
                    error_type="missing_image_path",
                    error_message="image_path is missing",
                )
            outputs.append(row)
            _append_checkpoint(checkpoint_path, row)
            continue
        if not image_path.exists():
            row = build_error_row(
                    record,
                    metadata,
                    model_name=model_name,
                    model_version=str(model_path),
                    generation_config=generation_config,
                    error_type="image_not_found",
                    error_message="resolved image path does not exist",
                )
            outputs.append(row)
            _append_checkpoint(checkpoint_path, row)
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
            row = {
                    "item_id": record.get("item_id"),
                    "model_id": model_name,
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
                    "claim_polarity": metadata.get("claim_polarity"),
                    "evidence_state": metadata.get("evidence_state"),
                }
            outputs.append(row)
            _append_checkpoint(checkpoint_path, row)
        except Exception as exc:  # noqa: BLE001 - keep batch robust on remote runs.
            row = build_error_row(
                    record,
                    metadata,
                    model_name=model_name,
                    model_version=str(model_path),
                    generation_config=generation_config,
                    error_type=exc.__class__.__name__,
                    error_message=str(exc)[:500],
                )
            outputs.append(row)
            _append_checkpoint(checkpoint_path, row)

    if checkpoint_path is not None:
        write_jsonl(outputs, checkpoint_path)
    return outputs


def _append_checkpoint(path: Path | None, row: dict[str, Any]) -> None:
    """Durably append one completed attempt to the private checkpoint journal."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def _latest_rows_by_item_id(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load the latest checkpoint attempt for every item."""

    if path is None or not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("item_id") is not None:
            rows[str(row["item_id"])] = row
    return rows


def _validate_resumed_row(
    previous: dict[str, Any],
    current_input: dict[str, Any],
    *,
    model_name: str,
    generation_config: dict[str, Any],
) -> None:
    """Reject a successful checkpoint row from another model or configuration."""

    previous_model = previous.get("model_id") or previous.get("model_name")
    if previous_model != model_name:
        raise ValueError("checkpoint model identity does not match current run")
    if previous.get("prompt_template_id") != current_input.get("prompt_template_id"):
        raise ValueError("checkpoint prompt template does not match current input")
    if previous.get("generation_config") != generation_config:
        raise ValueError("checkpoint generation configuration does not match current run")


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    parser = argparse.ArgumentParser(description="Run a local multimodal model on JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--eval-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--model-name",
        required=True,
        help="Canonical model ID recorded in every output row.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Private per-item checkpoint journal used for interruption recovery.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful rows from --checkpoint and retry failed rows.",
    )
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
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint.")

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
            "Multimodal dry run: "
            f"prompt_records={summary['prompt_records']}, "
            f"existing_images={summary['existing_images']}, "
            f"missing_images={summary['missing_images']}, "
            f"missing_metadata={summary['missing_metadata']}"
        )
        failure_fields = (
            "duplicate_prompt_item_ids",
            "duplicate_metadata_item_ids",
            "unexpected_metadata",
            "missing_metadata",
            "missing_image_path",
            "missing_images",
        )
        return 1 if any(summary[field] for field in failure_fields) else 0

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
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
        )
    except (RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    write_jsonl(outputs, args.output)
    success = sum(1 for row in outputs if row.get("runtime", {}).get("status") == "success")
    failed = len(outputs) - success
    print(
        "Ran multimodal inference: "
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
