[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'lib\Study2Data.ps1')
$context = Get-Study2Context
$prompts = Join-Path $context.DataRoot 'processed\prompts'
$transfer = Join-Path $context.DataRoot 'outputs\transfer'
$modelInputs = @(
    (Join-Path $prompts 'study2_g1_claim_verification_model_inputs.jsonl'),
    (Join-Path $prompts 'study2_g2_claim_verification_model_inputs.jsonl')
)
$evalMetadata = @(
    (Join-Path $prompts 'study2_g1_claim_verification_eval_metadata.jsonl'),
    (Join-Path $prompts 'study2_g2_claim_verification_eval_metadata.jsonl')
)
foreach ($required in $modelInputs + $evalMetadata) { Assert-Study2Path $required }

$gitCommit = (& git -C $context.RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to read the repository commit.' }
Invoke-Study2Python $context.Python @(
    '-m', 'ghm.migration.bundle', 'manifest',
    '--data-root', $context.DataRoot,
    '--output-dir', $transfer,
    '--git-commit', $gitCommit,
    '--model-inputs', $modelInputs[0], $modelInputs[1],
    '--eval-metadata', $evalMetadata[0], $evalMetadata[1]
)

Write-Host 'No images were copied or moved.'
Write-Host 'Transfer file list: outputs\transfer\study2_files_from.txt'
Write-Host 'SHA256 manifest: outputs\transfer\study2_files.sha256'
