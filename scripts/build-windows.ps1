[CmdletBinding()]
param(
    [ValidateSet("x64")][string]$Architecture = "x64",
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [int]$Jobs = [Math]::Min([Environment]::ProcessorCount, 8)
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$pythonCommand = Get-Command python -ErrorAction Stop
$buildArgs = @("$PSScriptRoot/build.py", "--platform", "windows", "--arch", $Architecture, "--jobs", "$Jobs")
if ($BuildDir) { $buildArgs += @("--build-dir", $BuildDir) }
if ($OutputDir) { $buildArgs += @("--output-dir", $OutputDir) }
& $pythonCommand.Source @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Windows build failed with exit code $LASTEXITCODE" }
