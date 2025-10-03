# Usa uma imagem base do Python
FROM python:3.10-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia APENAS o arquivo de requisitos
COPY requirements.txt .

# Instala todas as bibliotecas que seu projeto precisa
RUN pip install --no-cache-dir -r requirements.txt

# Copia TODO o resto do seu projeto para dentro do container
# Isso vai recriar sua estrutura de pastas exatamente como está no seu PC
COPY . .

# Informa ao Docker que sua API usa a porta 8000
EXPOSE 8000

# O comando para iniciar sua API quando o container ligar
# Assumindo que seu arquivo principal está em 'src/api/main_api.py'
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "src.api.main_api:app"]