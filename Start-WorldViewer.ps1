param([int]$Port = 8766, [string]$Bundle = '')
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$python = Join-Path $taskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project .venv is missing. See PHASE7.md.' }
Write-Host "World viewer: http://127.0.0.1:$Port/  (Ctrl+C stops the read-only service)"
$viewerArgs = @('-X', 'utf8', (Join-Path $taskRoot 'scripts/serve_world_viewer.py'), '--port', "$Port", '--feed', (Join-Path $taskRoot 'artifacts/runs/phase7-feed'))
if ($Bundle) { $viewerArgs += @('--bundle', $Bundle) }
& $python @viewerArgs
exit $LASTEXITCODE
