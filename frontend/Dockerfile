# frontend/Dockerfile
# --------------------
# Two-stage build: install deps in builder, copy only site-packages to runtime.
# Keeps the final image small (no pip/setuptools bloat).

FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app.py .

# Cloud Run injects PORT; Streamlit honours the --server.port flag
ENV PORT=8501
ENV BACKEND_URL=http://localhost:8080

EXPOSE 8501

# Interview note: --server.headless=true disables the "Deploy to Streamlit Cloud"
# banner and the email-collection prompt — required for production containers.
CMD ["sh", "-c", "streamlit run app.py \
  --server.port=${PORT} \
  --server.headless=true \
  --server.address=0.0.0.0 \
  --browser.gatherUsageStats=false"]
