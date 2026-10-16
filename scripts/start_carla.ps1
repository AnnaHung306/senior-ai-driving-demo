param([string]$CarlaRoot = '', [int]$Port = 2000)
$ErrorActionPreference = 'Stop'
if (-not $CarlaRoot) {
    $report = Join-Path (Split-Path -Parent $PSScriptRoot) 'artifacts\reports\installation.json'
    if (Test-Path -LiteralPath $report) {
        $installation = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
        if ($installation.status -ne 'passed') { throw 'Installation not validated' }
        $CarlaRoot = $installation.carla_root
    } else {
        $CarlaRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'artifacts\carla'
    }
}
$executable = Join-Path $CarlaRoot 'CarlaUE4.exe'
if (-not (Test-Path -LiteralPath $executable)) { throw "CARLA not installed: $executable" }
$server = Start-Process -FilePath $executable -WorkingDirectory $CarlaRoot -WindowStyle Hidden -PassThru -ArgumentList @('-RenderOffScreen', '-quality-level=Low', "-carla-rpc-port=$Port", '-nosound', '-unattended')
@{launcher_pid=$server.Id; executable=$executable; port=$Port} | ConvertTo-Json
