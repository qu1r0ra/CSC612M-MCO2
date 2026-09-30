param(
    [Parameter(Mandatory = $true)]
    [string]$Executable
)

$ErrorActionPreference = "Continue"
$output = & $Executable 2>&1 | ForEach-Object { "$_" }
$exitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
$diagnostic = $output -join [Environment]::NewLine

if ($exitCode -eq 0) {
    throw "AddressSanitizer did not fail on the deliberate one-byte overread."
}
if ($diagnostic -notmatch "AddressSanitizer: heap-buffer-overflow") {
    throw "Overread probe exited $exitCode without the expected heap-buffer-overflow diagnostic: $diagnostic"
}
if ($diagnostic -notmatch "READ of size 1") {
    throw "Overread probe did not report the expected one-byte read: $diagnostic"
}

Write-Output "AddressSanitizer caught the deliberate one-byte heap overread."
exit 0
