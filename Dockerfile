# syntax=docker/dockerfile:1.6

# ==============================================================================
# Estágio 1: "Builder" - Focado em instalar dependências Python
# Este estágio serve apenas para criar um ambiente com todas as bibliotecas
# Python instaladas, que serão copiadas para o estágio final.
# ==============================================================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Usa o cache do pip para acelerar builds futuros
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt


# ==============================================================================
# Estágio 2: "Final" - Imagem de produção limpa, segura e robusta
# ==============================================================================
FROM python:3.10-slim

# Diretiva: Instalar dependências de sistema no estágio final para maior robustez.
# Adicionadas libs comuns para OpenCV e FFmpeg.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Cria um usuário não-root por segurança
RUN addgroup --system app && adduser --system --group app

WORKDIR /app

# Copia as dependências Python já instaladas do estágio 'builder'
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Diretiva: Mover script principal para a raiz e manter 'src' como pacote.
# O --chown garante que o usuário 'app' tenha permissão sobre os arquivos.
COPY --chown=app:app src/main_uni.py .
COPY --chown=app:app src ./src
COPY --chown=app:app config.json .
COPY --chown=app:app runs ./runs

# Diretiva: Manter PYTHONPATH para garantir que 'src' seja importável.
ENV PYTHONPATH=/app

# Muda para o usuário não-root
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

# Diretiva: O ENTRYPOINT agora chama o main_uni.py na raiz.
# O -u garante que os prints() apareçam nos logs do Docker sem atraso.
ENTRYPOINT ["python", "-u", "main_uni.py"]

# Diretiva: O CMD padrão agora é o subcomando 'api', expondo em 0.0.0.0.
# Isso torna o container um serviço por padrão, sem bloquear stdin.
CMD ["api", "--host", "0.0.0.0", "--port", "8000"]