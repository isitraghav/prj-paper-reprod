# Sequential GPU job queue (one job at a time, never P1 and P2 together). Runs detached from any shell:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','code\repro\queue.ps1'
# Jobs file: results/queue/jobs.txt, one job per line:  name | working dir (repo-relative) | command line | log (repo-relative)
# Lines starting with # are ignored. A line "END" stops the queue once every job above it is done. Jobs can be
# appended while the queue runs. Finished jobs are recorded in results/queue/done.txt (name, exit code, times).
$ErrorActionPreference = 'Continue'
$PRJ = (Resolve-Path "$PSScriptRoot\..\..").Path
$Q = Join-Path $PRJ 'results\queue'
New-Item -ItemType Directory -Force $Q | Out-Null
$jobs = Join-Path $Q 'jobs.txt'; $done = Join-Path $Q 'done.txt'
if (-not (Test-Path $done)) { New-Item -ItemType File $done | Out-Null }
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONDONTWRITEBYTECODE = '1'
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] queue started, pid $PID" | Add-Content (Join-Path $Q 'queue.log')
while ($true) {
    $finished = @(Get-Content $done | ForEach-Object { ($_ -split '\|')[0].Trim() })
    $next = $null; $end = $false
    foreach ($l in (Get-Content $jobs)) {
        if ($l -match '^\s*(#|$)') { continue }
        if ($l.Trim() -eq 'END') { $end = $true; break }
        $f = $l -split '\|', 4 | ForEach-Object { $_.Trim() }
        if ($finished -notcontains $f[0]) { $next = $f; break }
    }
    if ($null -eq $next) {
        if ($end) { break }
        Start-Sleep -Seconds 60; continue
    }
    $name, $wd, $cmd, $log = $next
    $t0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "[$t0] start $name" | Add-Content (Join-Path $Q 'queue.log')
    $logPath = Join-Path $PRJ $log
    New-Item -ItemType Directory -Force (Split-Path $logPath) | Out-Null
    Push-Location (Join-Path $PRJ $wd)
    cmd /c "$cmd > `"$logPath`" 2>&1"
    $rc = $LASTEXITCODE
    Pop-Location
    $t1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "$name | exit $rc | $t0 | $t1" | Add-Content $done
    "[$t1] done  $name exit $rc" | Add-Content (Join-Path $Q 'queue.log')
}
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] queue finished" | Add-Content (Join-Path $Q 'queue.log')
