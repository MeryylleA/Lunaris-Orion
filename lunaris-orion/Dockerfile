# Use uma imagem base do NVIDIA CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Evita interações durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Copia o resto do código
COPY . .

# Cria diretórios necessários
RUN mkdir -p models cache

# Configura variáveis de ambiente padrão
ENV MODEL_DEVICE=cuda \
    MODEL_PRECISION=float16 \
    MODEL_MAX_BATCH_SIZE=4 \
    PYTHONUNBUFFERED=1

# Expõe a porta da API
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Labels úteis
LABEL maintainer="Lunaris Team" \
      version="1.0.0" \
      description="Lunaris Orion API - Gerador de Pixel Art usando IA" 