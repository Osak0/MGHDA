[CmdletBinding()]
param(
    [int]$MissingFindingSampleSize = 2,
    [int]$MissingFindingSeed = 42
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
$g1Linked = Join-Path $items 'study2_g1_claim_verification_items_linked.jsonl'
$g2Linked = Join-Path $items 'study2_g2_claim_verification_items_linked.jsonl'

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
Invoke-Study2Python $context.Python @(
    '-m', 'ghm.data.link_and_download_mimic_jpg', '--study2-only',
    '--study2-g1-items', $g1, '--study2-g2-items', $g2,
    '--study2-g1-output', $g1Linked, '--study2-g2-output', $g2Linked,
    '--metadata', $metadata, '--split', $split,
    '--files-root', $filesRoot, '--image-path-root', 'files',
    '--needed-index', (Join-Path $interim 'study2_needed_mimic_jpg_index.parquet'),
    '--manifest', (Join-Path $interim 'study2_needed_mimic_jpg_download_manifest.csv'),
    '--url-list', (Join-Path $interim 'study2_needed_mimic_jpg_urls.txt'),
    '--summary', (Join-Path $audits 'study2_mimic_jpg_link_summary.json'),
    '--link-only-existing'
)
