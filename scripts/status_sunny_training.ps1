param(
    [string[]]$RunDirectories = @(
        'artifacts/runs/sunny-segmentation-v4-001',
        'artifacts/runs/sunny-light-state-v1-002',
        'artifacts/runs/sunny-detection-v3-001'
    )
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot

function Read-SharedText([string]$Path) {
    if (-not [System.IO.File]::Exists($Path)) { return $null }
    # Allow the training process to atomically replace JSON while we read it.
    $taskSharing = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
    $taskStream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $taskSharing)
    $taskReader = [System.IO.StreamReader]::new($taskStream, [System.Text.Encoding]::UTF8, $true)
    try { return $taskReader.ReadToEnd() }
    finally { $taskReader.Dispose() }
}

$taskRuns = foreach ($taskDirectory in $RunDirectories) {
    $taskPath = if ([System.IO.Path]::IsPathRooted($taskDirectory)) {
        [System.IO.Path]::GetFullPath($taskDirectory)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $taskRoot $taskDirectory))
    }
    $taskStateText = Read-SharedText (Join-Path $taskPath 'state.json')
    $taskProgressText = Read-SharedText (Join-Path $taskPath 'progress.json')
    $taskFailureText = Read-SharedText (Join-Path $taskPath 'failure.json')
    $taskEpochText = Read-SharedText (Join-Path $taskPath 'epochs.jsonl')
    $taskState = if ($taskStateText) { $taskStateText | ConvertFrom-Json } else { $null }
    $taskProgress = if ($taskProgressText) { $taskProgressText | ConvertFrom-Json } else { $null }
    $taskFailure = if ($taskFailureText) { $taskFailureText | ConvertFrom-Json } else { $null }
    $taskLastEpoch = $null
    if ($taskEpochText) {
        # An append may be in flight; only consume newline-terminated records.
        $taskLines = $taskEpochText -split "`n"
        if ($taskLines.Length -gt 1) {
            $taskLastCompleteLine = $taskLines[$taskLines.Length - 2].Trim()
            if ($taskLastCompleteLine) { $taskLastEpoch = $taskLastCompleteLine | ConvertFrom-Json }
        }
    }
    $taskStatus = if ($taskFailure) { 'failure_recorded' }
                  elseif ($taskState) { $taskState.status }
                  elseif ([System.IO.Directory]::Exists($taskPath)) { 'directory_exists_no_state' }
                  else { 'not_created' }
    [ordered]@{
        run = $taskPath
        status_recorded = $taskStatus
        state = $taskState
        latest_progress = $taskProgress
        last_complete_epoch = $taskLastEpoch
        failure_message = if ($taskFailure) { $taskFailure.message } else { $null }
        lockbox_marker_exists = [System.IO.File]::Exists((Join-Path $taskPath 'lockbox_started.json'))
    }
}
[ordered]@{
    sampled_at = (Get-Date -Format o)
    scope = 'Read-only saved records; process liveness and GPU activity are not inferred. Each file is an independent observation.'
    runs = @($taskRuns)
} | ConvertTo-Json -Depth 12
