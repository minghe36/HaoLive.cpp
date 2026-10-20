[CmdletBinding()]
param(
    [ValidateSet("x64")][string]$Architecture = "x64",
    [ValidateSet("cpu", "vulkan", "cuda126")][string]$Variant = "cpu",
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [int]$Jobs = [Math]::Min([Environment]::ProcessorCount, 8)
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($Variant -eq "cuda126") {
    $vswhere = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    $vswherePath = if ($vswhere) { $vswhere.Source } else { Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe" }
    $vsRoot = & $vswherePath -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsRoot) { throw "Visual Studio C++ installation not found" }
    $devCmd = Join-Path $vsRoot "Common7/Tools/VsDevCmd.bat"
    $environmentCommand = '"' + $devCmd + '" -arch=x64 -host_arch=x64 -vcvars_ver=14.29 >nul && set'
    $variables = & $env:ComSpec /d /s /c $environmentCommand
    if ($LASTEXITCODE -ne 0) { throw "Cannot initialize the v142 x64 toolchain" }
    foreach ($line in $variables) {
        $pair = $line -split "=", 2
        if ($pair.Count -eq 2 -and $pair[0]) {
            [Environment]::SetEnvironmentVariable($pair[0], $pair[1], "Process")
        }
    }
    Get-Command cl.exe, nvcc.exe, ninja.exe -ErrorAction Stop | Out-Null
}
$pythonCommand = Get-Command python -ErrorAction Stop
$buildArgs = @("$PSScriptRoot/build.py", "--platform", "windows", "--arch", $Architecture, "--variant", $Variant, "--jobs", "$Jobs")
if ($BuildDir) { $buildArgs += @("--build-dir", $BuildDir) }
if ($OutputDir) { $buildArgs += @("--output-dir", $OutputDir) }
& $pythonCommand.Source @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Windows build failed with exit code $LASTEXITCODE" }
