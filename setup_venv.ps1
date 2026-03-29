$ErrorActionPreference = "Stop"

if (Test-Path .venv) {
  Remove-Item -Recurse -Force .venv
}

python -m venv .venv

$VenvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"

& $VenvPython -m ensurepip --upgrade
& $VenvPython -m pip install --upgrade --force-reinstall pip setuptools wheel
& $VenvPython -m pip install -e .

Write-Host "OK: venv created and scaffold installed (pip repaired and upgraded)."
