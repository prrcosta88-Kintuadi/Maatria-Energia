@echo off
echo ========================================
echo   INSTALAÇÃO KINTUADI ENERGY
echo ========================================

REM 1. Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python não encontrado!
    echo Instale Python 3.8+ em: https://python.org
    pause
    exit /b 1
)

REM 2. Instala dependências
echo.
echo 📦 Instalando dependências...
pip install -r requirements.txt

REM 3. Cria .env se não existir
if not exist ".env" (
    echo.
    echo ⚙️  Criando arquivo de configuração...
    copy .env.example .env >nul
    echo ✅ Arquivo .env criado!
    echo 📝 Edite o arquivo .env com suas credenciais ONS
)

REM 4. Cria pastas
mkdir data >nul 2>&1
mkdir logs >nul 2>&1

echo.
echo 🎉 Instalação completa!
echo.
echo ▶️  Execute: python run_collector.py
echo.
pause