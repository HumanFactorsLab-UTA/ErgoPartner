# Stop the local React/FastAPI dashboard started by launch_dashboard.ps1.
#
# A process id is never trusted on its own. Windows reuses ids, so the process
# is stopped only after its executable and command line match this project.

param(
    [int]$Port = 8000,
    [int]$TimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot "logs\dashboard.pid"
$url = "http://127.0.0.1:$Port"

function Test-DashboardHealth {
    try {
        $response = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 2
        return [bool]$response.ok
    }
    catch {
        return $false
    }
}

function Test-IsDashboardProcess {
    param([int]$ProcessId)

    if ($ProcessId -le 0) { return $false }
    try {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
    }
    catch {
        return $false
    }
    if ($null -eq $process -or $process.Name -notmatch '^python(w)?\.exe$') { return $false }

    $commandLine = [string]$process.CommandLine
    $executable = [string]$process.ExecutablePath
    if ($commandLine -notmatch 'uvicorn') { return $false }
    if ($commandLine -notmatch 'api\.main:app') { return $false }
    if ($commandLine -notmatch "(--port\s+|--port=)$Port") { return $false }
    $usesProjectRuntime =
        ($executable -and $executable.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) -or
        ($commandLine.IndexOf($projectRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)
    if (-not $usesProjectRuntime) {
        # Windows venv python.exe can delegate to its base Python executable.
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.ParentProcessId)" -ErrorAction SilentlyContinue
        if ($null -eq $parent -or $parent.CreationDate -gt $process.CreationDate) { return $false }
        $parentExecutable = [string]$parent.ExecutablePath
        $parentCommand = [string]$parent.CommandLine
        $runtimePaths = @('.venv', '.venv313', '.venv-audit') | ForEach-Object {
            Join-Path $projectRoot "$_\Scripts\python.exe"
        }
        if ($parentExecutable -notin $runtimePaths -or
            $parentCommand -notmatch 'uvicorn' -or
            $parentCommand -notmatch 'api\.main:app' -or
            $parentCommand -notmatch "(--port\s+|--port=)$Port") { return $false }
    }
    return $true
}

function Get-DashboardProcessId {
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        if (Test-IsDashboardProcess -ProcessId $listener.OwningProcess) {
            return [int]$listener.OwningProcess
        }
    }
    if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
        $recorded = (Get-Content -LiteralPath $pidFile -Raw).Trim()
        $parsed = 0
        if ([int]::TryParse($recorded, [ref]$parsed) -and (Test-IsDashboardProcess -ProcessId $parsed)) {
            return $parsed
        }
        Write-Host "Recorded process id $recorded is stale; checking port $Port instead."
    }

    return 0
}

$dashboardProcessId = Get-DashboardProcessId
if ($dashboardProcessId -eq 0) {
    if (Test-DashboardHealth) {
        Write-Warning "Something is serving $url but it is not this project's dashboard. Nothing was stopped."
        exit 1
    }
    Write-Host "The dashboard is not running. Nothing to stop."
    if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
    exit 0
}

try {
    $activity = Invoke-RestMethod -Uri "$url/api/analysis/jobs?active=true" -TimeoutSec 2
    $capture = Invoke-RestMethod -Uri "$url/api/capture/status" -TimeoutSec 2
}
catch {
    throw "Analysis/camera activity could not be checked. The dashboard was left running."
}
if (@($activity.jobs).Count -gt 0 -or $capture.recording -or $capture.preview_active -or $capture.assessmentId) {
    Write-Warning "Finish the analysis, or stop and save the recording and close the cameras, before stopping the dashboard. Nothing was stopped."
    exit 1
}

Write-Host "Stopping dashboard (process id $dashboardProcessId)..."
Stop-Process -Id $dashboardProcessId -ErrorAction Stop

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (-not (Get-Process -Id $dashboardProcessId -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Milliseconds 250
}

if (Get-Process -Id $dashboardProcessId -ErrorAction SilentlyContinue) {
    Write-Warning "Process $dashboardProcessId did not exit within $TimeoutSeconds seconds."
    exit 1
}

if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }

if (Test-DashboardHealth) {
    Write-Warning "The dashboard process stopped but $url is still responding. Another server may be running."
    exit 1
}

Write-Host "Dashboard stopped. Recordings and analysis results are unaffected."
exit 0
