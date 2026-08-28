[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = (Join-Path $HOME "Workspace\Active\japanese-speaking-assessment-poc"),

    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Installer = Join-Path $PSScriptRoot "apply_bundle.py"
$Arguments = @($Installer, $Target)
if ($Force) { $Arguments += "--force" }
if ($DryRun) { $Arguments += "--dry-run" }

& python @Arguments
exit $LASTEXITCODE
