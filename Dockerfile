# ── Base image ─────────────────────────────────────────────────
FROM python:3.11.9-slim

# ── Metadata ───────────────────────────────────────────────────
LABEL maintainer="Shubham Bhardwaj"
LABEL description="NIDS — Network Intrusion Detection System"
LABEL version="1.0.0"

# ── System dependencies ────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer caching) ───────────────────
COPY api/requirements.txt .

# ── Install Python dependencies ────────────────────────────────
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# ── Copy project files ─────────────────────────────────────────
COPY api/main.py .
COPY model/ ./model/

# ── Expose port ────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" \
    || exit 1

# ── Start API ──────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
