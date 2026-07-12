#!/usr/bin/env pwsh
<#
.SYNOPSIS
Commit files one-by-one to git with individual commits.

.DESCRIPTION
Stages and commits each changed/untracked file separately, allowing
for a cleaner commit history with separate commits per logical file change.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent)

Write-Host "Repository root: $(Get-Location)" -ForegroundColor Cyan

# Get all untracked and modified files
$files = @()
$files += git status --porcelain | Where-Object { $_ -match '^\?\?' } | ForEach-Object { $_.Substring(3) }
$files += git status --porcelain | Where-Object { $_ -match '^( M| D| A)' } | ForEach-Object { $_.Substring(3) }

if ($files.Count -eq 0) {
    Write-Host "No files to commit." -ForegroundColor Yellow
    Pop-Location
    exit 0
}

Write-Host "Found $($files.Count) file(s) to commit." -ForegroundColor Green

$commit_count = 0
foreach ($file in $files) {
    $file = $file.Trim()
    if ([string]::IsNullOrWhiteSpace($file)) { continue }
    
    Write-Host "`nStaging: $file" -ForegroundColor Cyan
    git add $file
    
    # Infer a commit message from the filename
    $basename = Split-Path -Leaf $file
    $dirname = Split-Path -Parent $file
    
    $msg = switch -Wildcard ($basename) {
        "*requirements*" { "chore: update dependencies ($basename)" }
        "*README*" { "docs: update README with reconstruction status and usage" }
        "*LICENSE*" { "chore: add MIT license" }
        "*ci.yml" { "ci: add GitHub Actions workflow for smoke tests" }
        "commit*.ps1" { "chore: add git commit script" }
        "run_hierarchical.py" { "feat: implement hierarchical inference runner with confidence-based routing" }
        "train_resnet50*.py" { "refactor: replace TPU-specific ResNet50 training with portable CLI" }
        "train_resnet152*.py" { "feat: add ResNet152V2 single-stage training script" }
        "*callbacks*" { "refactor: centralize Keras callbacks" }
        "*layers_head*" { "refactor: add reusable classification head builder" }
        "*resnet50*" { "feat: implement ResNet50 single-stage model builder" }
        "*resnet152*" { "feat: implement ResNet152V2 single-stage model builder" }
        "prepare_dataset.py" { "feat: add portable dataset preparation CLI with synthetic augmentation" }
        "utils_io.py" { "feat: add dataset I/O utilities" }
        "merge_all_classes.py" { "feat: add dataset merging utility" }
        "*test*.py" { "test: add smoke tests for module imports and file existence" }
        default { "chore: update $basename" }
    }
    
    Write-Host "Committing with message: $msg" -ForegroundColor Yellow
    git commit -m $msg
    $commit_count++
}

Write-Host "`nCompleted: $commit_count file(s) committed individually." -ForegroundColor Green
git log --oneline -n $commit_count

Pop-Location
