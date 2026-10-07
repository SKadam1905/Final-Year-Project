# Backend API + inference loop, CPU-only PyTorch (small image, slow inference).
# For the demo, run the backend natively in the venv to use the GPU; Docker Desktop GPU
# passthrough needs extra WSL 2 + NVIDIA setup.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Shared libraries opencv-python needs at import time
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Same layout as the repo, so config/ and data/ resolve to /repo/config and /repo/data
WORKDIR /repo/backend
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
