[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot '..\lib\Study3Data.ps1')
$context = Get-Study3Context
$items = Join-Path $context.DataRoot 'processed\study3\v2\items'
$prompts = Join-Path $context.DataRoot 'processed\study3\v2\prompts'
New-Item -ItemType Directory -Force -Path $prompts | Out-Null
foreach ($granularity in @('g1', 'g2')) {
    $input = Join-Path $items "study3_${granularity}_multiselect_items.jsonl"
    Assert-Study3Path $input
    Invoke-Study3Python $context.Python @(
        '-m', 'ghm.study3.prompts',
        '--input', $input,
        '--model-inputs-output', (Join-Path $prompts "study3_${granularity}_multiselect_model_inputs.jsonl"),
        '--eval-metadata-output', (Join-Path $prompts "study3_${granularity}_multiselect_eval_metadata.jsonl")
    )
}
$smoke = Join-Path $context.DataRoot 'processed\study3\v2\smoke'
New-Item -ItemType Directory -Force -Path $smoke | Out-Null
Invoke-Study3Python $context.Python @(
    '-m', 'ghm.study3.smoke',
    '--model-inputs',
    (Join-Path $prompts 'study3_g1_multiselect_model_inputs.jsonl'),
    (Join-Path $prompts 'study3_g2_multiselect_model_inputs.jsonl'),
    '--eval-metadata',
    (Join-Path $prompts 'study3_g1_multiselect_eval_metadata.jsonl'),
    (Join-Path $prompts 'study3_g2_multiselect_eval_metadata.jsonl'),
    '--model-inputs-output', (Join-Path $smoke 'model_inputs.jsonl'),
    '--eval-metadata-output', (Join-Path $smoke 'eval_metadata.jsonl'),
    '--seed', '42'
)
