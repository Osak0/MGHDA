# Study 2/3 answer-preference analysis

This workflow diagnoses answer-format failures and model response preferences in
addition to the usual accuracy, F1, Hamming accuracy, and Jaccard metrics.

It reads private row-level results locally, but writes aggregate counts only.
The generated files do not contain prompts, response text, item/anchor/set IDs,
option text, image paths, or evidence records.

## Run on Windows

```powershell
$resultRoot = "D:\path\to\experiment-results"

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/analysis/analyze_experiment_preferences.ps1 `
  -Study2Dir "$resultRoot\study2_second" `
  -Study3Dir "$resultRoot\study3" `
  -OutputDir "outputs/analysis/experiment_preferences"
```

If `python` is not on `PATH`, pass the interpreter explicitly:

```powershell
-Python "D:\path\to\python.exe"
```

## Aggregate outputs

- `file_inventory.csv`: filename, size, empty-file status, and file type.
- `study2_answer_distribution.csv`: all six evidence/polarity cells, with
  `Supported`, `Contradicted`, `Not enough evidence`, `No`, null/unparsed, and
  other out-of-schema counts and rates.
- `study3_summary_metrics.csv`: cross-model metrics recovered from each
  nonempty combined summary, even when a scored JSONL was not downloaded.
- `study3_group_metrics.csv`: valid-only and end-to-end metrics plus predicted
  and gold selection rates by granularity, framing, relation, variant, and K.
- `study3_parse_distribution.csv`: parser-status counts and rates.
- `study3_set_size_distribution.csv`: predicted versus gold set-size
  distributions.
- `study3_position_bias.csv`: selection rate, gold rate, selection gap, and
  accuracy by option position.
- `study3_pair_consistency.csv`: present/absent complement consistency.
- `study3_invalid_shape_distribution.csv`: aggregate invalid-response shapes
  and length buckets, without response snippets.
- `study3_runtime_diagnostics.csv`: runtime status, latency quantiles, and safe
  decoding parameters from raw result files.
- `sanitized_experiment_analysis.json`: file completeness and artifact row
  counts.

Only these aggregate outputs should be shared for collaborative analysis. Do
not upload the original raw/scored JSONL files or any patient-linked material.

## Interpretation order

1. Check file completeness and invalid rate before comparing model quality.
2. Compare Study 2 answer distributions across all six matched conditions.
3. For Study 3, compare predicted selection rate with the gold selection rate.
4. Separate valid-only metrics from end-to-end metrics when parsing fails.
5. Inspect `NONE`, position, present/absent, and K effects before attributing
   errors to visual understanding.
