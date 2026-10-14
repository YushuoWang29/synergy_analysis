param(
    [string]$Python = "python",
    [string]$XeLaTeX = "xelatex"
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo

$pythonCommand = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python was not found. Activate hand_synergy or pass -Python with its executable path."
}
$analysisPython = $pythonCommand.Source

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked $analysisPython @("course_project\scripts\run_course_final_analysis.py")

Invoke-Checked $analysisPython @("course_project\scripts\build_course_paper.py")

$texCommand = Get-Command $XeLaTeX -ErrorAction SilentlyContinue

if ($texCommand) {
    $texArguments = @("-interaction=nonstopmode", "-halt-on-error", "-output-directory=course_project\paper", "course_project\paper\course_paper.tex")
    Invoke-Checked $texCommand.Source $texArguments
    Invoke-Checked $texCommand.Source $texArguments
    Write-Host "Done. Course paper: course_project\paper\course_paper.pdf"
} else {
    Write-Warning "XeLaTeX was not found. TeX source was generated at course_project\paper\course_paper.tex, but the PDF was not rebuilt."
}
