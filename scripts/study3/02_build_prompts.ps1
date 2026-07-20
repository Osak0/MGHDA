[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot '..\lib\Study3Data.ps1')
$context = Get-Study3Context
$items = Join-Path $context.DataRoot 'processed\study3\items'
$prompts = Join-Path $context.DataRoot 'processed\study3\prompts'
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
