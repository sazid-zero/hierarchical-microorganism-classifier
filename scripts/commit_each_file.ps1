# Commit each changed/untracked file individually with a descriptive commit message.
# Usage: Open PowerShell at repo root and run:
#   .\scripts\commit_each_file.ps1

$ErrorActionPreference = 'Stop'

# Ensure we're in the repo root (script invoked from repo root by default)
$root = Get-Location
Write-Host "Repository root: $root"

# Get porcelain status lines
$statusLines = (& git status --porcelain) -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
if (-not $statusLines) {
    Write-Host "No changes detected (working tree clean)."
    exit 0
}

foreach ($line in $statusLines) {
    # porcelain format: XY <path>   or '?? <path>'
    # extract the path part robustly
    $pathPart = $null
    if ($line -match '^(?:\?\?|[ MADRCU?!]{2})\s+(.+)$') {
        $pathPart = $Matches[1]
    } else {
        # fallback: take substring after first 3 chars
        if ($line.Length -gt 3) { $pathPart = $line.Substring(3).Trim() }
        else { $pathPart = $line }
    }

    # handle rename entries like "R100 from -> to"
    if ($pathPart -match '->') {
        $parts = $pathPart -split '->'
        $pathPart = $parts[-1].Trim()
    }

    # skip submodule entries
    if ($pathPart -match '^(.+)\s*\(new file\)$') { $pathPart = $Matches[1].Trim() }

    if (-not $pathPart) { continue }

    $filePath = $pathPart
    Write-Host "Processing: $filePath"

    try {
        git add -- "$filePath"
        git commit -m "Update: $filePath" --quiet
        Write-Host "Committed: $filePath"
    } catch {
        Write-Host "No commit made for: $filePath  (maybe no changes staged)"
    }
}

Write-Host "All done."