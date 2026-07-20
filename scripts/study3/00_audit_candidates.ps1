[CmdletBinding()]
param()

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
$audits = Join-Path $context.DataRoot 'outputs\study3\audits'
New-Item -ItemType Directory -Force -Path $audits | Out-Null
Invoke-Study3Python $context.Python @(
    '-m', 'ghm.study3.candidates',
    '--attributes', $attributes, '--objects', $objects,
    '--mimic-metadata', $metadata, '--mimic-split', $split,
    '--files-root', $filesRoot, '--image-path-root', 'files',
    '--output', (Join-Path $audits 'study3_candidate_audit.json')
)
