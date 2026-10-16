param([string]$CarlaRoot = '', [int]$Port = 2000,
      [ValidateSet('dx11','dx12')][string]$GraphicsApi = 'dx12',
      [ValidateSet('Low','Epic')][string]$Quality = 'Low')
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
$server = Start-Process -FilePath $executable -WorkingDirectory $CarlaRoot -WindowStyle Hidden -PassThru -ArgumentList @("-$GraphicsApi", '-RenderOffScreen', "-quality-level=$Quality", "-carla-rpc-port=$Port", '-nosound', '-unattended')
$shipping = Join-Path $CarlaRoot 'CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe'
$launch = @{launcher_pid=$server.Id; executable=$executable; port=$Port; graphics_api=$GraphicsApi; quality_level=$Quality; server_binary_sha256=(Get-FileHash -LiteralPath $shipping -Algorithm SHA256).Hash.ToLowerInvariant()}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path (Split-Path -Parent $PSScriptRoot) 'artifacts\reports\server-launch.json')
$launch | ConvertTo-Json
