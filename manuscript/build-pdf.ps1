$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# asciidoctor-mathematical uses native Unix libraries and does not build on a
# standard Windows Ruby setup. Prefer WSL2 so this command genuinely renders
# the LaTeX equations instead of silently emitting raw TeX.
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'The STEM renderer requires Linux libraries. Install WSL2 with Ubuntu, install the dependencies listed in README.md, then rerun this script; alternatively download the PDF artifact from the repository GitHub Actions workflow.'
}

$linuxPath = (& wsl.exe wslpath -a $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxPath) {
    throw 'Could not translate this folder path into WSL. Open the repository from a WSL-accessible drive and retry.'
}
$command = "cd '$linuxPath' && bash ./build-pdf.sh"
& wsl.exe bash -lc $command
if ($LASTEXITCODE -ne 0) { throw 'PDF build failed inside WSL. Review the error output above.' }
Write-Host "Created $PSScriptRoot\CH03_Dadwal_World_Models_in_Action.pdf"