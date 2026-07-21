[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'lib\Study2Data.ps1')
$context = Get-Study2Context
$items = Join-Path $context.DataRoot 'processed\items'
$promptsV1 = Join-Path $context.DataRoot 'processed\study2\v1'
$promptsV2 = Join-Path $context.DataRoot 'processed\study2\v2'
$ablation = Join-Path $context.DataRoot 'processed\study2\ablation'
New-Item -ItemType Directory -Force -Path $promptsV1, $promptsV2, $ablation | Out-Null

foreach ($split in @('study2_g1', 'study2_g2')) {
    $input = Join-Path $items "${split}_claim_verification_items_linked.jsonl"
    Assert-Study2Path $input
    foreach ($version in @('v1', 'v2')) {
        $outputRoot = if ($version -eq 'v1') { $promptsV1 } else { $promptsV2 }
        Invoke-Study2Python $context.Python @(
            '-m', 'ghm.prompts.build_prompts',
            '--input', $input,
            '--model-inputs-output', (Join-Path $outputRoot "${split}_model_inputs.jsonl"),
            '--eval-metadata-output', (Join-Path $outputRoot "${split}_eval_metadata.jsonl"),
            '--template-version', $version
        )
    }
}
Invoke-Study2Python $context.Python @(
    '-m', 'ghm.evaluation.study2_ablation', 'build',
    '--inputs',
    (Join-Path $items 'study2_g1_claim_verification_items_linked.jsonl'),
    (Join-Path $items 'study2_g2_claim_verification_items_linked.jsonl'),
    '--output-root', $ablation,
    '--per-stratum', '10',
    '--seed', '42'
)
