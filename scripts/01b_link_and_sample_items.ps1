[CmdletBinding()]
param(
    [int]$MaxItemsTotal = 960,
    [int]$SamplingSeed = 42
)

. (Join-Path $PSScriptRoot 'lib\Study2Data.ps1')
if ($MaxItemsTotal -lt 4 -or $MaxItemsTotal % 2 -ne 0) {
    throw 'MaxItemsTotal must be an even integer of at least 4.'
}

$context = Get-Study2Context
$interim = Join-Path $context.DataRoot 'interim'
$items = Join-Path $context.DataRoot 'processed\items'
$audits = Join-Path $context.DataRoot 'outputs\audits'
$metadata = Join-Path $context.DataRoot 'raw\MIMIC-CXR\mimic-cxr-2.0.0-metadata.csv'
$split = Join-Path $context.DataRoot 'raw\MIMIC-CXR\mimic-cxr-2.0.0-split.csv'
$filesRoot = Join-Path $context.DataRoot 'files'
$g1 = Join-Path $items 'study2_g1_claim_verification_items.jsonl'
$g2 = Join-Path $items 'study2_g2_claim_verification_items.jsonl'
foreach ($required in @($metadata, $split, $filesRoot, $g1, $g2)) {
    Assert-Study2Path $required
}
$maxItemsPerGroup = [int]($MaxItemsTotal / 2)

Invoke-Study2Python $context.Python @(
    '-m', 'ghm.data.link_and_download_mimic_jpg', '--study2-only',
    '--study2-g1-items', $g1, '--study2-g2-items', $g2,
    '--study2-g1-output', (Join-Path $items 'study2_g1_claim_verification_items_linked.jsonl'),
    '--study2-g2-output', (Join-Path $items 'study2_g2_claim_verification_items_linked.jsonl'),
    '--metadata', $metadata, '--split', $split,
    '--files-root', $filesRoot, '--image-path-root', 'files',
    '--needed-index', (Join-Path $interim 'study2_needed_mimic_jpg_index.parquet'),
    '--manifest', (Join-Path $interim 'study2_needed_mimic_jpg_download_manifest.csv'),
    '--url-list', (Join-Path $interim 'study2_needed_mimic_jpg_urls.txt'),
    '--summary', (Join-Path $audits 'study2_mimic_jpg_link_summary.json'),
    '--max-linked-items-per-group', "$maxItemsPerGroup",
    '--sampling-seed', "$SamplingSeed",
    '--link-only-existing'
)
