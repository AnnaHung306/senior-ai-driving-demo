$ErrorActionPreference = 'Stop'
# Run before launching CARLA, not after the simulator has claimed GPU memory.
$taskGpuScripts = '(?:^|[\\/\s])(?:train_sunny|train_light_state|train_segmentation|predict_perception|benchmark_perception|benchmark_sunny|validate_light_joint|evaluate_light_state|evaluate_sunny|verify_sunny_capacity|run_phase3_recovery_batch)\.py(?:\s|"|$)'
$taskBusyProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(?:w)?\.exe$' -and $_.CommandLine -match $taskGpuScripts
})
if ($taskBusyProcesses.Count -gt 0) {
    $taskProcessIds = ($taskBusyProcesses | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Perception training/evaluation is still running (PID $taskProcessIds). Keep CARLA off until it exits."
}
$taskMemoryLines = @(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if ($LASTEXITCODE -ne 0 -or $taskMemoryLines.Count -eq 0) {
    throw 'Cannot confirm idle NVIDIA GPU; simulator was not started.'
}
foreach ($taskMemory in $taskMemoryLines) {
    $taskUsedMib = 0
    if (-not [int]::TryParse($taskMemory.Trim(), [ref]$taskUsedMib)) {
        throw 'Unrecognized GPU memory query; simulator was not started.'
    }
    if ($taskUsedMib -gt 2048) {
        throw "GPU already uses $taskUsedMib MiB; wait for perception work to finish before starting CARLA."
    }
}
