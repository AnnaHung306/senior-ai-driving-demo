$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$record = Join-Path $projectRoot 'artifacts\reports\server-launch.json'
$launch = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
$expected = [IO.Path]::GetFullPath((Join-Path $projectRoot 'artifacts\carla\CarlaUE4.exe'))
if ([IO.Path]::GetFullPath($launch.executable) -ne $expected) {
    throw 'Refusing to stop a server outside the project CARLA installation'
}
$ownedProcess = Get-Process -Id $launch.launcher_pid -ErrorAction SilentlyContinue
if ($null -eq $ownedProcess) {
    Write-Output 'Recorded launcher is no longer running; no process was stopped.'
    return
}
if ([IO.Path]::GetFullPath($ownedProcess.Path) -ne $expected) {
    throw 'PID was reused or executable path differs; no process was stopped'
}
taskkill /PID $ownedProcess.Id /T /F
if ($LASTEXITCODE -ne 0) { throw 'Owned CARLA process tree could not be stopped' }
Write-Output 'Project-owned CARLA process tree stopped.'
