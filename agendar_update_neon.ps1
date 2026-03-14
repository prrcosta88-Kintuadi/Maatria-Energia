# agendar_update_neon.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Registra uma tarefa no Agendador de Tarefas do Windows para rodar
# update_neon.py todo dia às 00h15.
#
# Como usar (copie e cole linha a linha no PowerShell):
#   1. Abra o PowerShell como Administrador
#   2. Execute: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser   -> [S]
#   3. Execute: Unblock-File .\agendar_update_neon.ps1
#   4. Navegue até a pasta do projeto: cd C:\repos\Kintuadi-Energy
#   5. Execute: .\agendar_update_neon.ps1
# ─────────────────────────────────────────────────────────────────────────────

# ── Configurações — ajuste se necessário ─────────────────────────────────────
$TaskName    = "KintuadiEnergia_UpdateNeon"
$ProjectDir  = (Get-Location).Path          # pasta atual do projeto
$PythonExe   = (Get-Command python).Source  # detecta python do PATH
$ScriptFile  = Join-Path $ProjectDir "update_neon.py"
$LogFile     = Join-Path $ProjectDir "logs\task_scheduler.log"
# ─────────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗"
Write-Host "║   KINTUADI ENERGY — Agendador de Tarefas Windows         ║"
Write-Host "╚══════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "Diretório do projeto : $ProjectDir"
Write-Host "Python detectado     : $PythonExe"
Write-Host "Script               : $ScriptFile"
Write-Host "Horário              : 00:15 todos os dias"
Write-Host ""

# Verificar se o script existe
if (-not (Test-Path $ScriptFile)) {
    Write-Error "update_neon.py não encontrado em $ProjectDir. Rode o script a partir da pasta do projeto."
    exit 1
}

# Remover tarefa anterior se existir
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removendo tarefa anterior '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Criar os objetos da tarefa
$Action  = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "$ScriptFile" `
    -WorkingDirectory $ProjectDir

$Trigger = New-ScheduledTaskTrigger -Daily -At "00:15"

# -StartWhenAvailable: roda assim que possível caso o PC estivesse desligado às 00h15
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable `
    -WakeToRun:$false

$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Highest

# Registrar
Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $Action `
    -Trigger   $Trigger `
    -Settings  $Settings `
    -Principal $Principal `
    -Description "Atualiza tabelas Neon e triggera rebuild no Render — Kintuadi Energy" `
    | Out-Null

Write-Host "✅ Tarefa '$TaskName' criada com sucesso!"
Write-Host ""
Write-Host "Para verificar: Abra o Agendador de Tarefas (taskschd.msc)"
Write-Host "Para testar agora: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Para remover: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
Write-Host ""
