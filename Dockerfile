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

# Copia os parquets de seção (arquivos Git normais, não LFS)
COPY data/core_section_advanced_metrics.parquet data/core_section_advanced_metrics.parquet
COPY data/core_section_economic.parquet         data/core_section_economic.parquet
COPY data/core_section_operacao.parquet         data/core_section_operacao.parquet
COPY data/core_section_ccee.parquet             data/core_section_ccee.parquet
COPY data/core_section_renewables.parquet       data/core_section_renewables.parquet

# Validação dos parquets de seção
RUN python3 - <<'PY'
import duckdb
from pathlib import Path

sections = ["advanced_metrics", "economic", "operacao", "ccee", "renewables"]
ok = True
for s in sections:
    p = Path(f"data/core_section_{s}.parquet")
    if not p.exists():
        print(f"ERRO: {p} nao encontrado")
        ok = False
        continue
    head = p.read_bytes()[:512]
    if b"git-lfs.github.com/spec/v1" in head or b"oid sha256:" in head:
        print(f"ERRO: {p} e ponteiro LFS ({p.stat().st_size} bytes)")
        ok = False
        continue
    try:
        con = duckdb.connect(database=":memory:")
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]
        con.close()
        print(f"OK: {p} ({p.stat().st_size:,} bytes, {count} linha(s))")
    except Exception as e:
        print(f"ERRO: {p} falhou: {e}")
        ok = False

if not ok:
    raise SystemExit("Um ou mais parquets de secao sao invalidos.")
print("Todos os parquets de secao validados com sucesso.")
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
