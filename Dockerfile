# XG3 Badminton Microservice — Production Dockerfile
# Service: xg3-badminton (port 8034)

# ─── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends  curl\
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Add prometheus_client for /metrics endpoint
RUN pip install --no-cache-dir --prefix=/install "prometheus_client>=0.17,<1.0"

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps (libgomp for LightGBM/XGBoost + curl for HEALTHCHECK).
# curl is REQUIRED because Coolify v4 runs `curl -fsS http://localhost/...` as
# the deploy-time healthcheck — without it the probe returns "curl: not found"
# and Coolify rolls back to the previous container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN useradd -r -u 1001 -s /bin/false xg3
WORKDIR /app
RUN chown xg3:xg3 /app

# Copy application source
COPY --chown=xg3:xg3 . /app

# Remove test, dev, and cache files from the image
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name "*.pyc" -delete && \
    rm -rf /app/tests /app/.git

USER xg3

# Expose service port + metrics port
EXPOSE 8034

# Environment defaults (override via docker run -e or Railway env)
ENV SERVICE_ENV=production \
    PORT=8034 \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Trivial HEALTHCHECK that always succeeds within 2s. Coolify v4 deploy-time
# rolling-update probe needs the container to report 'healthy' fast or it
# rolls back. Real /health monitoring happens at L7 via gateway proxy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8034}/health/live || exit 1

# Run with uvicorn — production settings.
# Shell form so $PORT (set by Coolify env, defaults 8034) is interpolated;
# this MUST match the HEALTHCHECK port above so the deploy-time probe hits
# the actual listening socket.
CMD sh -c "exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8034} --workers 2 --loop uvloop --log-level warning --no-access-log"
