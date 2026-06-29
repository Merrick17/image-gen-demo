FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch — install before requirements.txt to pin CPU variant
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

# Single worker — multiple workers would each load the 8+ GB model into RAM
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
