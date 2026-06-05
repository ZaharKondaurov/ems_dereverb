FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyPI first (do not use --index-url for torch: it hides PyPI and breaks fastapi)
COPY requirements-web.txt /app/requirements-web.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-web.txt \
    && pip install --no-cache-dir torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

COPY . /app

ENV FSPEN_PRESET=fspen_48khz_overlap
ENV FSPEN_DEVICE=cpu
ENV FSPEN_CHUNK_MS=500

EXPOSE 7860

CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "7860"]
