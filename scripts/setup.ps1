param(
    [Parameter(Mandatory=$true)][string]$Python312,
    [string]$PipPython = 'python'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
& $Python312 -c "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 required'"
if ($LASTEXITCODE -ne 0) { throw 'Invalid Python runtime' }
if (-not (Test-Path .venv\Scripts\python.exe)) {
    & $Python312 -m venv --without-pip .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& $PipPython -m pip --python .venv\Scripts\python.exe install pip -r requirements-sim.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .venv\Scripts\python.exe -c "import carla; print('CARLA API import OK')"
if ($LASTEXITCODE -ne 0) { throw 'CARLA import failed' }
