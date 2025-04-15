# All data agents startup PowerShell script

# Configuration file list with relative paths
$configs = @(
    "config/purchase_orders_config.yaml",
    "config/payments_config.yaml",
    "config/vendors_config.yaml",
    "config/expenses_config.yaml",
    "config/employees_config.yaml",
    "config/communications_config.yaml",
    "config/calls_config.yaml",
    "config/requests_config.yaml"
)

# Get script directory and project root
$scriptDir = $PSScriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptDir)))
Write-Host "Project root directory: $projectRoot"

# Start each agent in a separate window with -NoExit parameter to keep the window open
foreach ($config in $configs) {
    $fullConfigPath = Join-Path $scriptDir $config
    # Use full path for the Python module
    $pythonCommand = "cd '$projectRoot'; python -m samples.python.agents.data_agent --config '$fullConfigPath'"
    Write-Host "Starting agent with config: $fullConfigPath"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $pythonCommand
}

Write-Host "All data agents have been started in separate windows. The windows will remain open until manually closed." 