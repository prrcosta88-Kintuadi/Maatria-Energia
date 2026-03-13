FROM python:3.10-slim AS runtime

WORKDIR /app

# 1) Dependências Python
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --retries 5 -q setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt && \
    echo "✅ Dependências Python instaladas"

# 2) Código-fonte
COPY app.py .
COPY db_neon.py .
COPY scripts/ scripts/
COPY streamlit/ streamlit/

# 3) Diretórios runtime
RUN mkdir -p data logs

# 4) Configuração runtime
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_LOGGER_LEVEL=warning
ENV STREAMLIT_CLIENT_SHOWERRORDETAILS=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; \
        r = requests.get('http://localhost:8501/_stcore/health', timeout=5); \
        raise SystemExit(0 if r.status_code == 200 else 1)"

CMD ["streamlit", "run", "app.py"]
