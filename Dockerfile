FROM python:3.12-slim

# `opencv-python-headless` ne lie pas libGL ; seule libglib2.0-0 est requise.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Sans cette limite, OpenCV lance un thread par cœur et les workers Uvicorn
# se battent pour le CPU.
ENV OMP_NUM_THREADS=2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Modèle embarqué par défaut ; le compose peut monter ./models par-dessus.
RUN bash scripts/download_model.sh

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
