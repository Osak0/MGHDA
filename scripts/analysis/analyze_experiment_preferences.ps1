[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Study2Dir,

    [Parameter(Mandatory = $true)]
    [string]$Study3Dir,

    [string]$OutputDir = "outputs/analysis/experiment_preferences",

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $repoRoot $OutputDir
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    & $Python -m ghm.evaluation.experiment_preferences `
        --study2-dir $Study2Dir `
        --study3-dir $Study3Dir `
        --output-dir $resolvedOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment preference analysis failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

Write-Host "Sanitized aggregate outputs: $resolvedOutput"
