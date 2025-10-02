# docker/Dockerfile.cpu.distributed-server
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e .[server]  # marker-api deps

ENV TORCH_DEVICE=cpu
ENV HF_HOME=/data/hf
ENV PORT=8080

CMD ["python", "distributed_server.py", "--host", "0.0.0.0", "--port", "8080"]
