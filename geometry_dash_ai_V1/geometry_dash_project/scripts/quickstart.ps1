<#
Quickstart helper for the Geometry Dash project (PowerShell)

This script creates a .venv (if missing) and installs dependencies. It does
not automatically activate the venv in the calling shell — after running the
script, run `& .\.venv\Scripts\Activate.ps1` to activate.

Usage (PowerShell):
  .\quickstart.ps1

#>

param(
    [switch]$CpuOnly
)

Write-Host "Creating virtual environment (.venv) if missing..."
if (-not (Test-Path -Path .\.venv)) {
    python -m venv .venv
}

Write-Host "To activate the virtual environment run: & .\.venv\Scripts\Activate.ps1"

if ($CpuOnly) {
    Write-Host "Installing CPU-only requirements..."
    python -m pip install -r requirements-cpu.txt
} else {
    Write-Host "Installing GPU / default requirements..."
    python -m pip install -r requirements.txt
}

Write-Host "Done. Activate the venv and run the game or training scripts." 
