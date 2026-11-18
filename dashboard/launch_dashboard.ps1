param(
    [switch]$NoOpen,
    [ValidateSet('Home', 'Live', 'Video', 'Assessments', 'Overview', 'Settings')]
    [string]$Page = 'Home'
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$port = 8000
$url = "http://127.0.0.1:$port"
$pagePaths = @{
    Home = '/'
    Live = '/new?source=live'
    Video = '/new?source=video'
    Assessments = '/assessments'
    Overview = '/overview'
    Settings = '/settings'
}
$destination = $url + $pagePaths[$Page]
$logs = Join-Path $projectRoot "logs"
$pidFile = Join-Path $logs "dashboard.pid"
$clientIndex = Join-Path $projectRoot "web\dist\index.html"

function Get-DashboardPython {
    $candidates = @(
        (Join-Path $projectRoot ".venv\Scripts\python.exe"),
        (Join-Path $projectRoot ".venv313\Scripts\python.exe"),
        (Join-Path $projectRoot ".venv-audit\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        & $candidate -c "import fastapi, uvicorn" 2>$null
        if ($LASTEXITCODE -eq 0) { return $candidate }
    }
    throw "No project Python runtime with FastAPI and Uvicorn was found."
}

function Test-DashboardHealth {
    try {
        $api = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 2
        if (-not $api.ok) { return $false }
        $page = Invoke-WebRequest -UseBasicParsing -Uri "$url/" -TimeoutSec 2
        return $page.StatusCode -eq 200 -and $page.Content -match 'id="root"'
    }
    catch {
        return $false
    }
}

function Get-DashboardActivityState {
    try {
        $state = Invoke-RestMethod -Uri "$url/api/analysis/jobs?active=true" -TimeoutSec 2
        if ($null -ne $state.jobs -and @($state.jobs).Count -gt 0) {
            return "active"
        }
        $capture = Invoke-RestMethod -Uri "$url/api/capture/status" -TimeoutSec 2
        if ($capture.preview_active -or $capture.recording -or $capture.assessmentId) {
            return "active"
        }
        return "idle"
    }
    catch {
        return "unknown"
    }
}

function Test-ClientBuildFreshness {
    $sourceRoot = Join-Path $projectRoot "web\src"
    $distRoot = Join-Path $projectRoot "web\dist"
    $latestSource = Get-ChildItem -LiteralPath $sourceRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @(".ts", ".tsx", ".css", ".html") } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    $latestBuild = Get-ChildItem -LiteralPath $distRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @(".html", ".css", ".js", ".map") } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $latestSource) { return $true }
    if ($null -eq $latestBuild) { return $false }
    return $latestSource.LastWriteTimeUtc -le $latestBuild.LastWriteTimeUtc
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
    if ($commandLine -notmatch "(--port\s+|--port=)$port") { return $false }
    $usesProjectRuntime =
        ($executable -and $executable.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) -or
        ($commandLine.IndexOf($projectRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)
    if (-not $usesProjectRuntime) {
        # Windows venv python.exe can delegate to its base Python executable.
        # Verify that child's immediate parent is this project's venv launcher.
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
            $parentCommand -notmatch "(--port\s+|--port=)$port") { return $false }
    }
    return $true
}

function Get-DashboardProcessId {
    # Prefer the process serving the port over a venv redirector saved at startup.
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
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
    }

    return 0
}

function Test-DashboardFreshness {
    param([int]$ProcessId)

    if ($ProcessId -le 0) { return $false }
    try {
        $startedUtc = (Get-Process -Id $ProcessId -ErrorAction Stop).StartTime.ToUniversalTime()
    }
    catch {
        return $false
    }

    $watchedDirectories = @(
        (Join-Path $projectRoot "api"),
        (Join-Path $projectRoot "assistant"),
        (Join-Path $projectRoot "core"),
        (Join-Path $projectRoot "dashboard"),
        (Join-Path $projectRoot "ergonomics"),
        (Join-Path $projectRoot "pipeline"),
        (Join-Path $projectRoot "reporting"),
        (Join-Path $projectRoot "temporal"),
        (Join-Path $projectRoot "utils"),
        (Join-Path $projectRoot "web\src"),
        (Join-Path $projectRoot "web\dist")
    )
    $watchedFiles = @(
        Get-ChildItem -LiteralPath $watchedDirectories -Recurse -File -ErrorAction SilentlyContinue
        Get-ChildItem -LiteralPath $projectRoot -File -ErrorAction SilentlyContinue
    ) | Where-Object {
        $_.Extension -in @(".py", ".yaml", ".yml", ".json", ".ts", ".tsx", ".html", ".css", ".js", ".map")
    }
    $latest = $watchedFiles |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $latest) { return $true }
    return $latest.LastWriteTimeUtc -le $startedUtc.AddSeconds(2)
}

function Wait-ForProcessExit {
    param([int]$ProcessId, [int]$TimeoutSeconds = 15)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Milliseconds 250
    }
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        throw "The previous dashboard process did not stop within $TimeoutSeconds seconds."
    }
}

if (-not (Test-Path -LiteralPath $clientIndex -PathType Leaf)) {
    throw "The React dashboard build is missing. Build the web client before using this shortcut."
}
if (-not (Test-ClientBuildFreshness)) {
    throw "The React dashboard source is newer than its production build. Build the web client before using this shortcut."
}

New-Item -ItemType Directory -Path $logs -Force | Out-Null

$healthy = Test-DashboardHealth
$existingProcessId = Get-DashboardProcessId
$refreshRequired = -not $healthy
$analysisState = $null
if ($healthy -and $existingProcessId -gt 0 -and -not (Test-DashboardFreshness -ProcessId $existingProcessId)) {
    $analysisState = Get-DashboardActivityState
    if ($analysisState -eq "active") {
        Write-Host "A newer dashboard build is available, but an analysis or camera session is active."
        Write-Host "Your work was left undisturbed. Start the dashboard again after it finishes."
    }
    elseif ($analysisState -eq "unknown") {
        throw "A newer dashboard build is available, but analysis/camera activity could not be checked. The running process was left undisturbed."
    }
    else {
        $refreshRequired = $true
        Write-Host "A newer dashboard build is available."
    }
}
if ($healthy -and $existingProcessId -eq 0) {
    throw "Port $port is responding, but it is not this project's dashboard. Nothing was changed."
}
if ($refreshRequired -and $existingProcessId -gt 0) {
    if ($null -eq $analysisState) {
        $analysisState = Get-DashboardActivityState
    }
    if ($analysisState -eq "active") {
        Write-Host "The dashboard did not answer its health check, but an analysis or camera session is active."
        Write-Host "Your work was left undisturbed. Start the dashboard again after it finishes."
        $refreshRequired = $false
    }
    elseif ($analysisState -eq "unknown") {
        throw "The dashboard health check failed and analysis/camera activity could not be checked. The running process was left undisturbed."
    }
}

if ($refreshRequired) {
    if ($existingProcessId -gt 0) {
        Write-Host "Refreshing the dashboard server to load the current interface..."
        Stop-Process -Id $existingProcessId -ErrorAction Stop
        Wait-ForProcessExit -ProcessId $existingProcessId
    }
    else {
        $occupied = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($occupied) {
            throw "Port $port is already used by another application. The dashboard was not started."
        }
    }

    $python = Get-DashboardPython
    $startArguments = @{
        FilePath = $python
        ArgumentList = @(
            "-m", "uvicorn", "api.main:app",
            "--host", "127.0.0.1",
            "--port", "$port",
            "--log-level", "warning"
        )
        WorkingDirectory = $projectRoot
        WindowStyle = "Hidden"
        PassThru = $true
        RedirectStandardOutput = Join-Path $logs "dashboard.stdout.log"
        RedirectStandardError = Join-Path $logs "dashboard.stderr.log"
    }
    $process = Start-Process @startArguments
    Set-Content -LiteralPath $pidFile -Value $process.Id

    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline -and -not (Test-DashboardHealth)) {
        if ($process.HasExited) {
            throw "Dashboard stopped during startup. See logs/dashboard.stderr.log."
        }
        Start-Sleep -Milliseconds 400
    }
    if (-not (Test-DashboardHealth)) {
        throw "Dashboard did not become healthy within 60 seconds. See logs/dashboard.stderr.log."
    }
    Set-Content -LiteralPath $pidFile -Value (Get-DashboardProcessId)
}
else {
    Set-Content -LiteralPath $pidFile -Value $existingProcessId
}

# Give each production build its own entry URL, avoiding an older cached index.
$buildVersion = (Get-FileHash -LiteralPath $clientIndex -Algorithm SHA256).Hash.Substring(0, 12).ToLowerInvariant()
$separator = if ($destination.Contains('?')) { '&' } else { '?' }
$destination = $destination + $separator + 'build=' + $buildVersion
if (-not $NoOpen) {
    Start-Process $destination
}
Write-Host "ErgoPartner is ready: $destination"
