$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot
$env:STREAMLIT_CONFIG_DIR = "$PSScriptRoot\.streamlit"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

"" | & "$PSScriptRoot\.venv\Scripts\python.exe" `
    -m streamlit run "$PSScriptRoot\src\ai_document_assistant\ui.py" `
    --browser.gatherUsageStats=false
