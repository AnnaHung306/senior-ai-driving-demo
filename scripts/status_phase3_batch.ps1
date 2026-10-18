param([string]$BatchDirectory = 'artifacts/runs/phase3-recovery-batch-001')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskBatchPath = if ([IO.Path]::IsPathRooted($BatchDirectory)) {
    [IO.Path]::GetFullPath($BatchDirectory)
} else { [IO.Path]::GetFullPath((Join-Path $taskRoot $BatchDirectory)) }

function Read-BatchSharedJson([string]$Path) {
    if (-not [IO.File]::Exists($Path)) { return $null }
    $taskSharing = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $taskStream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $taskSharing)
    $taskReader = [IO.StreamReader]::new($taskStream, [Text.Encoding]::UTF8, $true)
    try { $taskText = $taskReader.ReadToEnd() }
    finally { $taskReader.Dispose() }
    return ($taskText | ConvertFrom-Json)
}

$taskContract = Read-BatchSharedJson (Join-Path $taskBatchPath 'contract.json')
$taskStages = @(Read-BatchSharedJson (Join-Path $taskBatchPath 'stages.json'))
$taskLatestStage = if ($taskStages.Count -gt 0) { $taskStages[-1] } else { $null }
$taskModelRows = if ($null -ne $taskContract) {
    foreach ($taskProperty in $taskContract.runs.PSObject.Properties) {
        $taskModelPath = [IO.Path]::GetFullPath([string]$taskProperty.Value)
        $taskState = Read-BatchSharedJson (Join-Path $taskModelPath 'state.json')
        [ordered]@{
            model = $taskProperty.Name
            run = $taskModelPath
            status_recorded = if ($taskState) { $taskState.status } elseif ([IO.Directory]::Exists($taskModelPath)) { 'directory_exists_no_state' } else { 'not_created' }
            state = $taskState
            progress = Read-BatchSharedJson (Join-Path $taskModelPath 'progress.json')
            failure_history = Read-BatchSharedJson (Join-Path $taskModelPath 'failure.json')
            lockbox_marker_exists = [IO.File]::Exists((Join-Path $taskModelPath 'lockbox_started.json'))
        }
    }
}
[ordered]@{
    sampled_at_utc = [DateTime]::UtcNow.ToString('o')
    scope = 'Read-only saved observations; process liveness is not inferred. Delete sharing permits atomic writer replacement.'
    batch = $taskBatchPath
    latest_stage = $taskLatestStage
    finished_stages = @($taskStages | Where-Object { $_.finished } | Select-Object name, return_code, wall_s)
    models = @($taskModelRows)
    validation_gates = Read-BatchSharedJson (Join-Path $taskBatchPath 'validation_gates.json')
    batch_state = Read-BatchSharedJson (Join-Path $taskBatchPath 'state.json')
    batch_failure_history = Read-BatchSharedJson (Join-Path $taskBatchPath 'failure.json')
    awake_receipt = Read-BatchSharedJson (Join-Path $taskBatchPath 'awake.json')
} | ConvertTo-Json -Depth 14
