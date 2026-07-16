[CmdletBinding()]
param(
    [int]$MissingFindingSampleSize = 2,
    [int]$MissingFindingSeed = 42,
    [int]$MaxItemsTotal = 960,
    [int]$SamplingSeed = 42
)

. (Join-Path $PSScriptRoot 'lib\Study2Data.ps1')
$context = Get-Study2Context
$interim = Join-Path $context.DataRoot 'interim'
$items = Join-Path $context.DataRoot 'processed\items'
$audits = Join-Path $context.DataRoot 'outputs\audits'
$attributes = Join-Path $interim 'ci_attribute_assertions.parquet'
$objects = Join-Path $interim 'ci_objects.parquet'
$imageIndex = Join-Path $interim 'image_index.parquet'
$metadata = Join-Path $context.DataRoot 'raw\MIMIC-CXR\mimic-cxr-2.0.0-metadata.csv'
$split = Join-Path $context.DataRoot 'raw\MIMIC-CXR\mimic-cxr-2.0.0-split.csv'
$filesRoot = Join-Path $context.DataRoot 'files'
foreach ($required in @($attributes, $objects, $imageIndex, $metadata, $split, $filesRoot)) {
    Assert-Study2Path $required
}
New-Item -ItemType Directory -Force -Path $items, $audits | Out-Null

$summary = Join-Path $audits 'study2_candidate_summary.json'
$g1 = Join-Path $items 'study2_g1_claim_verification_items.jsonl'
$g2 = Join-Path $items 'study2_g2_claim_verification_items.jsonl'
Invoke-Study2Python $context.Python @(
    '-m', 'ghm.granularity.build_g1_items',
    '--input', $attributes, '--image-index', $imageIndex,
    '--output', $g1, '--summary', $summary,
    '--missing-finding-sample-size', "$MissingFindingSampleSize",
    '--missing-finding-seed', "$MissingFindingSeed"
)
Invoke-Study2Python $context.Python @(
    '-m', 'ghm.granularity.build_g2_items',
    '--attributes', $attributes, '--objects', $objects, '--image-index', $imageIndex,
    '--output', $g2, '--summary', $summary,
    '--missing-finding-sample-size', "$MissingFindingSampleSize",
    '--missing-finding-seed', "$MissingFindingSeed"
)
& (Join-Path $PSScriptRoot '01b_link_and_sample_items.ps1') `
    -MaxItemsTotal $MaxItemsTotal `
    -SamplingSeed $SamplingSeed
if ($LASTEXITCODE -ne 0) {
    throw "Link and sample step failed with exit code $LASTEXITCODE"
}
