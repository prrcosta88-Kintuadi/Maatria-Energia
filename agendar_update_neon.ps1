# agendar_update_neon.ps1
# ─────────────────────────────────────────────────────────────────────────────
# Registra tarefas no Agendador de Tarefas do Windows para rodar:
#   1. update_neon.py todo dia às 00h15
#   2. refresh do AdaptivePLDForwardEngine toda terça-feira às 22h00
#
# Como usar (copie e cole linha a linha no PowerShell):
#   1. Abra o PowerShell como Administrador
#   2. Execute: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser   -> [S]
#   3. Execute: Unblock-File .\agendar_update_neon.ps1
#   4. Navegue até a pasta do projeto: cd C:\repos\Kintuadi-Energy
#   5. Execute: .\agendar_update_neon.ps1
# ─────────────────────────────────────────────────────────────────────────────

# ── Configurações — ajuste se necessário ─────────────────────────────────────
$DailyTaskName    = "KintuadiEnergia_UpdateNeon"
$AdaptiveTaskName = "KintuadiEnergia_AdaptivePLDForwardWeekly"
$ProjectDir       = (Get-Location).Path          # pasta atual do projeto
$PythonExe        = (Get-Command python).Source  # detecta python do PATH
$ScriptFile       = Join-Path $ProjectDir "update_neon.py"
$LogFile          = Join-Path $ProjectDir "logs\task_scheduler.log"
$DailyArgs        = "`"$ScriptFile`" --allow-neon-failure --dir data"
$AdaptiveArgs     = "`"$ScriptFile`" --retrain-forecast --adaptive-forward-only --force-weekly-retrain --dir data"
# ─────────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗"
Write-Host "║   KINTUADI ENERGY — Agendador de Tarefas Windows         ║"
Write-Host "╚══════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "Diretório do projeto : $ProjectDir"
Write-Host "Python detectado     : $PythonExe"
Write-Host "Script               : $ScriptFile"
Write-Host "Tarefa diária        : 00:15 todos os dias"
Write-Host "Tarefa adaptive      : 22:00 toda terça-feira"
Write-Host ""

# Verificar se o script existe
if (-not (Test-Path $ScriptFile)) {
    Write-Error "update_neon.py não encontrado em $ProjectDir. Rode o script a partir da pasta do projeto."
    exit 1
}

# Remover tarefas anteriores se existirem
if (Get-ScheduledTask -TaskName $DailyTaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removendo tarefa anterior '$DailyTaskName'..."
    Unregister-ScheduledTask -TaskName $DailyTaskName -Confirm:$false
}
if (Get-ScheduledTask -TaskName $AdaptiveTaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removendo tarefa anterior '$AdaptiveTaskName'..."
    Unregister-ScheduledTask -TaskName $AdaptiveTaskName -Confirm:$false
}

# Criar os objetos das tarefas
$DailyAction = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $DailyArgs `
    -WorkingDirectory $ProjectDir

$AdaptiveAction = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument $AdaptiveArgs `
    -WorkingDirectory $ProjectDir

$DailyTrigger = New-ScheduledTaskTrigger -Daily -At "00:15"
$AdaptiveTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At "22:00"

# -StartWhenAvailable: roda assim que possível caso o PC estivesse desligado às 00h15
$DailySettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable `
    -WakeToRun:$false

$AdaptiveSettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 20) `
    -StartWhenAvailable `
    -WakeToRun:$false

$Principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Highest

# Registrar tarefa diária
Register-ScheduledTask `
    -TaskName  $DailyTaskName `
    -Action    $DailyAction `
    -Trigger   $DailyTrigger `
    -Settings  $DailySettings `
    -Principal $Principal `
    -Description "Atualiza tabelas Neon e triggera rebuild no Render — Kintuadi Energy" `
    | Out-Null

# Registrar tarefa semanal do adaptive forward
Register-ScheduledTask `
    -TaskName  $AdaptiveTaskName `
    -Action    $AdaptiveAction `
    -Trigger   $AdaptiveTrigger `
    -Settings  $AdaptiveSettings `
    -Principal $Principal `
    -Description "Recalcula semanalmente o Adaptive PLD Forward Engine e persiste o snapshot no DuckDB/Neon AUTH — Kintuadi Energy" `
    | Out-Null

Write-Host "✅ Tarefa '$DailyTaskName' criada com sucesso!"
Write-Host "✅ Tarefa '$AdaptiveTaskName' criada com sucesso!"
Write-Host ""
Write-Host "Para verificar: Abra o Agendador de Tarefas (taskschd.msc)"
Write-Host "Para testar a diária: Start-ScheduledTask -TaskName '$DailyTaskName'"
Write-Host "Para testar a weekly adaptive: Start-ScheduledTask -TaskName '$AdaptiveTaskName'"
Write-Host "Para remover a diária: Unregister-ScheduledTask -TaskName '$DailyTaskName' -Confirm:`$false"
Write-Host "Para remover a weekly adaptive: Unregister-ScheduledTask -TaskName '$AdaptiveTaskName' -Confirm:`$false"
Write-Host ""
