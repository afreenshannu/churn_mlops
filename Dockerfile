# # ── Stage 1: base ─────────────────────────────────────────────────────────────
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install dependencies first (cached layer — only re-runs if requirements change)
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy source code
# COPY src/ ./src/
# COPY app/ ./app/
# COPY models/ ./models/
# COPY data/processed/feature_columns.json ./data/processed/feature_columns.json

# # Expose the port FastAPI will run on
# EXPOSE 8000

# # Run the app with uvicorn
# CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir pandas numpy scikit-learn fastapi "uvicorn[standard]" pydantic

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/processed/feature_columns.json ./data/processed/feature_columns.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]