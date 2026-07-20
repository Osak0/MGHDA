[CmdletBinding()]
param(
    [int]$NaturalAnchors = 500,
    [int]$ControlledAnchors = 100,
    [string]$ControlledK = '2,3,4,5',
    [int]$SamplingSeed = 42
)

. (Join-Path $PSScriptRoot '..\lib\Study3Data.ps1')
$context = Get-Study3Context
$attributes = Join-Path $context.DataRoot 'interim\ci_attribute_assertions.parquet'
$objects = Join-Path $context.DataRoot 'interim\ci_objects.parquet'
$metadata = Join-Path $context.DataRoot 'raw\MIMIC-CXR\mimic-cxr-2.0.0-metadata.csv'
$split = Join-Path $context.DataRoot 'raw\MIMIC-CXR\mimic-cxr-2.0.0-split.csv'
$filesRoot = Join-Path $context.DataRoot 'files'
foreach ($required in @($attributes, $objects, $metadata, $split, $filesRoot)) {
    Assert-Study3Path $required
}
$items = Join-Path $context.DataRoot 'processed\study3\items'
$audits = Join-Path $context.DataRoot 'outputs\study3\audits'
New-Item -ItemType Directory -Force -Path $items, $audits | Out-Null
Invoke-Study3Python $context.Python @(
    '-m', 'ghm.study3.build_items',
    '--attributes', $attributes, '--objects', $objects,
    '--mimic-metadata', $metadata, '--mimic-split', $split,
    '--files-root', $filesRoot, '--image-path-root', 'files',
    '--g1-output', (Join-Path $items 'study3_g1_multiselect_items.jsonl'),
    '--g2-output', (Join-Path $items 'study3_g2_multiselect_items.jsonl'),
    '--summary', (Join-Path $audits 'study3_item_summary.json'),
    '--natural-anchors', "$NaturalAnchors",
    '--controlled-anchors', "$ControlledAnchors",
    '--controlled-k', $ControlledK,
    '--seed', "$SamplingSeed"
)
