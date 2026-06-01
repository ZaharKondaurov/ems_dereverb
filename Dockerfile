FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU (override in Space if GPU)
COPY requirements-web.txt /app/requirements-web.txt
RUN pip install --no-cache-dir -r requirements-web.txt \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . /app

ENV FSPEN_CHECKPOINT=checkpoints/fspen_chkp/TrainConfig_48kHz_enc_ext_1986#0.pt
ENV FSPEN_DEVICE=cpu
ENV FSPEN_CONFIG=TrainConfig_48kHz_enc_ext

EXPOSE 7860

CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "7860"]
