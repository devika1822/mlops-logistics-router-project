# Multi-stage Dockerfile – Cargo Payload Track (Student 3)
# Build context : tracks/cost
# Exposes port  : 8000  (docker-compose maps → 8003)

# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

# Install into a prefix so the runtime image stays lean
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only the installed packages (no pip, no build tools)
COPY --from=builder /install /usr/local

# Copy application source files
COPY main.py predict.py ./

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_STAGE=Production

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
