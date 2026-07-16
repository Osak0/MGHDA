[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'lib\Study2Data.ps1')
$context = Get-Study2Context
$objects = Join-Path $context.DataRoot 'interim\ci_objects.parquet'
$attributes = Join-Path $context.DataRoot 'interim\ci_attribute_assertions.parquet'
$auditDir = Join-Path $context.DataRoot 'outputs\audits'
Assert-Study2Path $objects
Assert-Study2Path $attributes
New-Item -ItemType Directory -Force -Path $auditDir | Out-Null

Invoke-Study2Python $context.Python @(
    '-m', 'ghm.data.audit_chest_imagenome',
    '--objects', $objects,
    '--assertions', $attributes,
    '--output-dir', $auditDir
)
