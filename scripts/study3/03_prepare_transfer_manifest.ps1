[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot '..\lib\Study3Data.ps1')
$context = Get-Study3Context
$prompts = Join-Path $context.DataRoot 'processed\study3\v2\prompts'
$transfer = Join-Path $context.DataRoot 'outputs\transfer\study3_v2'
$modelInputs = @(
    (Join-Path $prompts 'study3_g1_multiselect_model_inputs.jsonl'),
    (Join-Path $prompts 'study3_g2_multiselect_model_inputs.jsonl')
)
$evalMetadata = @(
    (Join-Path $prompts 'study3_g1_multiselect_eval_metadata.jsonl'),
    (Join-Path $prompts 'study3_g2_multiselect_eval_metadata.jsonl')
)
foreach ($required in $modelInputs + $evalMetadata) { Assert-Study3Path $required }
if ((Test-Path -LiteralPath $transfer) -and (Get-ChildItem -LiteralPath $transfer -Force)) {
    throw "Study 3 transfer directory must be empty: $transfer"
}
New-Item -ItemType Directory -Force -Path $transfer | Out-Null
$gitCommit = (& git -C $context.RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to read the repository commit.' }
Invoke-Study3Python $context.Python @(
    '-m', 'ghm.study3.transfer', 'manifest',
    '--data-root', $context.DataRoot,
    '--output-dir', $transfer,
    '--git-commit', $gitCommit,
    '--model-inputs', $modelInputs[0], $modelInputs[1],
    '--eval-metadata', $evalMetadata[0], $evalMetadata[1]
)
Write-Host 'No images were copied or moved.'
Write-Host 'Transfer list: outputs\transfer\study3_v2\study3_files_from.txt'
