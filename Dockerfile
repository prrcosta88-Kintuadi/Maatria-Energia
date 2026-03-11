# Dockerfile para deploy no Render.com usando artefato Parquet (Git LFS)
# - Evita dependência de JSON .gz no build
# - Mantém imagem menor e build mais simples

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
COPY dashboard_integrado.py .
COPY run_collector.py .
COPY setup_env.py .
COPY check_data.py .
COPY scripts/ scripts/
COPY streamlit/ streamlit/
COPY docker-compose.yml .

# 3) Dados necessários em runtime
RUN mkdir -p data logs audit_logs

# Copia o artefato versionado para deploy
COPY data/core_analysis_latest.parquet data/core_analysis_latest.parquet

# Validação simples do artefato
RUN ls -lh data/core_analysis_latest.parquet && \
    python - <<'PY'
import duckdb
from pathlib import Path
p = Path('data/core_analysis_latest.parquet')
head = p.read_bytes()[:200]
if b'git-lfs.github.com/spec/v1' in head:
    raise SystemExit('❌ Arquivo em data/core_analysis_latest.parquet é ponteiro LFS; objeto real não foi baixado no build.')
con = duckdb.connect(database=':memory:')
con.execute("SELECT COUNT(*) FROM read_parquet('data/core_analysis_latest.parquet')")
print('✅ Parquet válido para leitura')
PY

# 4) Configuração runtime
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_LOGGER_LEVEL=warning
ENV STREAMLIT_CLIENT_SHOWERRORDETAILS=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; \
        r = requests.get('http://localhost:8501/_stcore/health', timeout=5); \
        raise SystemExit(0 if r.status_code == 200 else 1)"

CMD ["streamlit", "run", "app.py"]
