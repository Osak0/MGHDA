Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-Study3Context {
    $repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    $dataRoot = if ($env:MGHDA_DATA_ROOT) {
        [System.IO.Path]::GetFullPath($env:MGHDA_DATA_ROOT)
    } else {
        'C:\Users\24540\data'
    }
    if (-not (Test-Path -LiteralPath $dataRoot -PathType Container)) {
        throw "MGHDA data root does not exist: $dataRoot"
    }

    $python = $null
    if ($env:MGHDA_PYTHON) {
        $python = $env:MGHDA_PYTHON
    } else {
        $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
            $python = $venvPython
        } else {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if ($pythonCommand) { $python = $pythonCommand.Source }
        }
    }
    if (-not $python) {
        throw 'Python was not found. Create .venv or set MGHDA_PYTHON to Python 3.10/3.11.'
    }
    $sourceRoot = Join-Path $repoRoot 'src'
    $env:PYTHONPATH = if ($env:PYTHONPATH) {
        "$sourceRoot;$($env:PYTHONPATH)"
    } else {
        $sourceRoot
    }
    return [pscustomobject]@{
        RepoRoot = $repoRoot
        DataRoot = $dataRoot
        Python = $python
    }
}

function Assert-Study3Path {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required Study 3 input is missing: $Path"
    }
}

function Invoke-Study3Python {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Study 3 Python command failed with exit code $LASTEXITCODE"
    }
}
