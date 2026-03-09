# Dockerfile Multi-Stage Otimizado para Render.com
# Opção 1: Descompacta JSON no build (RECOMENDADO PARA SEU CASO)
#
# Benefícios:
# - Evita timeout na inicialização
# - JSON já está descompactado pronto para usar
# - Imagem final é self-contained
# - Build é rápido em builds posteriores (cache Docker)
#
# Build: docker build -t kintuadi:latest .
# Run:   docker run -p 8501:8501 kintuadi:latest

# ============================================================================
# STAGE 1: BUILDER - Descompacta JSON (não vai pra imagem final)
# ============================================================================

FROM python:3.10-slim as builder

WORKDIR /build

# Copia arquivo gzip APENAS neste stage
COPY data/core_analysis_latest.json.gz .

# Descompacta JSON
# - Comando 'gunzip' descompacta eficientemente
# - Saída vai para arquivo que será copiado para stage 2
RUN echo "📦 Descompactando JSON (pode levar 30-60 segundos)..." && \
    gunzip -v -c core_analysis_latest.json.gz > core_analysis_latest.json && \
    echo "✅ JSON descompactado com sucesso" && \
    ls -lh core_analysis_latest.json && \
    # Valida JSON
    python -c "import json; json.load(open('core_analysis_latest.json')); print('✅ JSON válido')" || true


# ============================================================================
# STAGE 2: RUNTIME - Imagem final (leve e rápida)
# ============================================================================

FROM python:3.10-slim

WORKDIR /app

# ============================================================================
# 1. INSTALA DEPENDÊNCIAS PYTHON
# ============================================================================

RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --retries 5 -q setuptools wheel

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt && \
    echo "✅ Dependências Python instaladas"


# ============================================================================
# 2. COPIA CÓDIGO FONTE
# ============================================================================

# Scripts principais
COPY app.py .
COPY dashboard_integrado.py .
COPY run_collector.py .
COPY setup_env.py .
COPY check_data.py .

# Pasta scripts (lógica central)
COPY scripts/ scripts/

# Configuração Streamlit
COPY streamlit/ streamlit/

# Outros
COPY docker-compose.yml .


# ============================================================================
# 3. COPIA JSON DESCOMPACTADO DO STAGE 1
# ============================================================================

# Cria diretório data antes de copiar
RUN mkdir -p data logs audit_logs

# Copia o JSON já descompactado (só 410MB, não 37MB+410MB)
COPY --from=builder /build/core_analysis_latest.json data/core_analysis_latest.json

# Valida que arquivo chegou e é acessível
RUN ls -lh data/core_analysis_latest.json && \
    echo "✅ JSON pronto no container"


# ============================================================================
# 4. CONFIGURAÇÃO FINAL
# ============================================================================

# Expõe porta Streamlit
EXPOSE 8501

# Variáveis de ambiente Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
ENV PYTHONUNBUFFERED=1

# Outros (opcional, para performance)
ENV STREAMLIT_LOGGER_LEVEL=warning
ENV STREAMLIT_CLIENT_SHOWERRORDETAILS=false

# ============================================================================
# 5. HEALTH CHECK (Essencial para Render)
# ============================================================================
# 
# Render usa isso para verificar se app está vivo
# Se falhar 3 vezes, container é reiniciado
#

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; \
        try: \
            r = requests.get('http://localhost:8501/_stcore/health', timeout=5); \
            exit(0 if r.status_code == 200 else 1); \
        except: \
            exit(1)"


# ============================================================================
# 6. STARTUP
# ============================================================================

CMD ["streamlit", "run", "app.py"]


# ============================================================================
# NOTAS SOBRE LAYERS
# ============================================================================
#
# Ordem das linhas IMPORTA para cache Docker:
#
# 1. requirements.txt (muda raramente, cache é reusado)
# 2. Código Python (muda frequentemente, invalida cache)
# 3. JSON (muda com frequência, mas COPIA é rápida)
#
# Benefício: Se só código mudar, não precisa reinstalar pip packages
#
# ============================================================================

# ============================================================================
# DEBUGGING - Adicione se der problema
# ============================================================================
#
# Para ver o que tá acontecendo no build:
#
# docker build --progress=plain -t kintuadi:latest .
#
# Para debugar container:
#
# docker run -it kintuadi:latest /bin/bash
#
# Dentro do container:
#
# ls -la data/
# python -c "import json; d=json.load(open('data/core_analysis_latest.json')); print(len(str(d)))"
# streamlit run app.py --logger.level=debug
#
# ============================================================================
