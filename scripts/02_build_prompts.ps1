[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'lib\Study2Data.ps1')
$context = Get-Study2Context
$items = Join-Path $context.DataRoot 'processed\items'
$prompts = Join-Path $context.DataRoot 'processed\prompts'
New-Item -ItemType Directory -Force -Path $prompts | Out-Null

foreach ($split in @('study2_g1', 'study2_g2')) {
    $input = Join-Path $items "${split}_claim_verification_items_linked.jsonl"
    Assert-Study2Path $input
    Invoke-Study2Python $context.Python @(
        '-m', 'ghm.prompts.build_prompts',
        '--input', $input,
        '--model-inputs-output', (Join-Path $prompts "${split}_claim_verification_model_inputs.jsonl"),
        '--eval-metadata-output', (Join-Path $prompts "${split}_claim_verification_eval_metadata.jsonl")
    )
}
