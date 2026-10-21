param([int]$Port=8768,[switch]$StartSimulator,[switch]$ReplayOnly)
$ErrorActionPreference='Stop'
if ($StartSimulator -and $ReplayOnly) { throw 'Choose live simulator or replay-only mode.' }
if ($StartSimulator) {
    $probeClient=[Net.Sockets.TcpClient]::new()
    try { $connectTask=$probeClient.ConnectAsync('127.0.0.1',2000);$alreadyRunning=$connectTask.Wait(700) -and $probeClient.Connected }
    catch { $alreadyRunning=$false }
    finally { $probeClient.Dispose() }
    if (-not $alreadyRunning) { & (Join-Path $PSScriptRoot 'scripts/start_carla.ps1') }
}
$python=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not $ReplayOnly) {
    & $python -X utf8 (Join-Path $PSScriptRoot 'scripts/navigation_preflight.py')
    if ($LASTEXITCODE -ne 0) { throw 'Navigation preflight failed; use -StartSimulator or -ReplayOnly. See PHASE89.md.' }
}
Write-Host "Navigation demo: http://127.0.0.1:$Port/ (Ctrl+C stops service and requests vehicle stop)"
$serviceArgs=@('-X','utf8',(Join-Path $PSScriptRoot 'scripts/serve_navigation.py'),'--port',"$Port")
if ($ReplayOnly) { $serviceArgs += '--replay-only' }
& $python @serviceArgs
exit $LASTEXITCODE
