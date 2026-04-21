FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source modules
COPY config.py state.py models.py helpers.py health.py cache.py \
     ollama.py search.py classifier.py agents.py pipeline.py \
     streaming.py main.py tool_registry.py slash_commands.py ./

# Copy config
COPY config.yaml .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
