# 🚗 Vox Vehicle Color Detection System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-red.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema avançado de **detecção e rastreamento de veículos** com **classificação de cores** usando **YOLOv11** e **ByteTrack**. Detecta múltiplos carros simultaneamente e classifica suas cores com alta precisão.

## ✨ **Características Principais**

### **Detecção Multi-Car**
- **Detecta todos os carros** na imagem/vídeo (não apenas o maior)
- **Rastreamento persistente** com ByteTrack
- **Classificação individual** de cada veículo detectado
- **Confiança configurável** para detecção e classificação

### **Classificação de Cores**
- **10 cores suportadas**: azul, branco, cinza, marrom, prata, preto, rosa, verde, vermelho, amarelo
- **Modelos treinados** especificamente para veículos brasileiros
- **Confiança ajustável** via interface web
- **Visualização colorida** com numeração automática

### **API Completa**
- **Interface web interativa** com upload de arquivos
- **Processamento assíncrono** com progresso em tempo real
- **Múltiplos formatos** suportados (MP4, JPG, PNG)
- **Histórico completo** de processamentos

### **Poetry Ready**
- **Containerização completa** com Docker e Docker Compose
- **Suporte GPU/CPU** configurável
- **Health checks** automáticos
- **Volumes persistentes** para dados

## 🏗️ **Arquitetura do Sistema**

```
Vox/
├── 📋 requirements.txt              # Dependências Python
├── 📊 config.json                   # Configurações principais
├── 📁 src/
│   ├── ⚙️ main_uni.py               # Código fonte
│   ├── 🌐 api/main_api.py          # API FastAPI multi-car
│   ├── 🎯 tracking/track.py        # Motor de tracking otimizado
│   ├── 🎓 train/train.py           # Pipeline de treinamento (desativado)
│   ├── 🔧 utils/utils.py           # Utilitários
│   └── 📊 cli/commands.py          # Comandos CLI
├── 📂 runs/                         # Modelos treinados
│   ├── yolo11n_detection_detect3/weights/best.pt
│   ├── yolo11n_classification_colors_n3/weights/best.pt
│   ├── yolo11s_classification_colors_s3/weights/best.pt
│   └── yolo11s_detection_detect3/weights/best.pt
└── 📁 outras configs/               # Configurações adicionais
    ├── full config.json
    └── gpu config.json
```

## 🚀 **Quick Start com Poetry**

### **1. Clone e Configure**
```bash
git clone <repository>
cd Vox

# Configurar permissões
chmod +x scripts/deploy.sh
```

### **2. Ative o ambiente Conda e Instale o Poetry**
```bash
conda create -n vox python=3.10 -y
conda activate vox
pip install poetry
```

### **3. Instale dependências**
```bash
poetry install --no-root
poetry add requests[use_chardet-on-py3]
poetry install --no-root
poetry run python -c "import requests; print(requests.__version__)"
poetry install --no-root
```

### **4. Como Executar**
```bash
#Modo Interativo 
poetry run python src/main_uni.py
#Modo API 
poetry run uvicorn src.api.main_api:app --host 127.0.0.1 --port 8000 --reload
```

## 📦 **Instalação Manual**

### **Pré-requisitos**
- Python 3.11+
- CUDA 11.8+ (opcional, para GPU)
- FFmpeg
- Git

### **1. Instalar Dependências**
```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### **2. Verificar Modelos**
```bash
# Modelos disponíveis no projeto
runs/yolo11n_detection_detect3/weights/best.pt       # Detector nano
runs/yolo11s_detection_detect3/weights/best.pt       # Detector small
runs/yolo11n_classification_colors_n3/weights/best.pt # Classifier nano
runs/yolo11s_classification_colors_s3/weights/best.pt # Classifier small
```

### **3. Executar**
```bash
# API Web
python main_uni.py api

# Modo interativo
python main_uni.py

# Processar vídeo direto
python main_uni.py track \
  --video input.mp4 \
  --det-weights runs/yolo11n_detection_detect3/weights/best.pt \
  --cls-weights runs/yolo11n_classification_colors_n3/weights/best.pt
```


## 🎯 **Casos de Uso**

### **1. Monitoramento de Tráfego**
- Contagem de veículos por cor
- Análise de fluxo de tráfego
- Estatísticas em tempo real

### **2. Segurança e Vigilância**
- Identificação de veículos suspeitos
- Tracking de veículos específicos
- Análise forense de vídeos

### **3. Pesquisa de Mercado**
- Preferências de cores por região
- Análise de tendências automotivas
- Estudos de comportamento

### **4. Smart Cities**
- Integração com semáforos inteligentes
- Otimização de rotas
- Planejamento urbano

## 📊 **Performance**

### **Benchmarks Atuais (Sistema do Desenvolvedor)**
- **Detector**: YOLO11n (320px) - ~15ms por frame
- **Classificador**: YOLO11n/s (128px) - ~8ms por crop
- **Throughput**: **45-55 FPS** com frame_stride 2-3
- **Memory Usage**: ~1.5GB RAM
- **CPU**: 12 threads otimizadas

### **Configurações de Performance**
```json
{
  "performance": {
    "frame_stride": 2,           // Processa 1 a cada 2 frames
    "detection_interval": 3,     // Detecta a cada 3 frames
    "batch_size": 10,           // Classifica 10 carros por vez
    "num_threads_cpu": 12,      // Máximo paralelismo CPU
    "enable_cache": true,       // Cache inteligente ativo
    "cache_size": 500,          // 500 classificações em cache
    "cache_ttl": 20             // Cache expira em 20 frames
  }
}
```

### **Otimizações Implementadas**
- ✅ **Smart Caching** - Reduz classificações redundantes em 40-60%
- ✅ **Batch Processing** - Processa múltiplos carros simultaneamente
- ✅ **Frame Striding** - Processa 1 a cada N frames
- ✅ **ONNX Fallback** - Inferência otimizada quando disponível
- ✅ **Multi-threading** - Paralelização CPU otimizada
- ✅ **Async Processing** - Processamento não-bloqueante
- ✅ **Memory Pooling** - Reutilização eficiente de buffers
- ✅ **Detection Interval** - Detecção espaçada para performance
 build-gpu`, `make run-gpu`, `make up-gpu`  
- **Utilitários**: `make logs`, `make status`, `make health`, `make shell`
- **Testes**: `make test`, `make lint`, `make format`
- **Limpeza**: `make clean`, `make clean-all`

## 📈 **Roadmap**

### **v2.1 (Próxima Release)**
- [ ] Suporte a Treino/Retreino fechando o ciclo CI/CD
- [ ] Suporte a streaming em tempo real
- [ ] Dashboard de analytics avançado
- [ ] Exportação para banco de dados
- [ ] API de estatísticas históricas

## 🔧 **Desenvolvimento**

### **Setup de Desenvolvimento**
```bash
# Ambiente de desenvolvimento com hot reload
make dev

# Acesso:
# API: http://localhost:8000
# Jupyter (se habilitado): http://localhost:8888
```

### **Comandos Make Disponíveis**
```bash
make help              # Lista todos os comandos
make build             # Build da imagem
make run               # Executa container
make up                # Sobe todos os serviços
make down              # Para todos os serviços
make logs              # Mostra logs
make test              # Executa testes
make clean             # Limpeza básica
make deploy            # Deploy automatizado
make health            # Verifica saúde da API
```

### **Estrutura de Comandos**
- **Desenvolvimento**: `make dev`, `make dev-build`, `make dev-logs`
- **Produção**: `make build`, `make run`, `make up`, `make deploy`
- **GPU**: `makene CI/CD automatizado** 
- [x] **Treinamento desativado** (modo produção)
- [ ] Kubernetes deployment
- [ ] Multi-tenant support
- [ ] Machine Learning drift detection

## 📝 **Status Atual**

### **v2.0.0** (Atual - Modo Produção)
- ✅ **Multi-car detection** - Detecta todos os carros
- ✅ **Docker containerization** - Deploy simplificado
- ✅ **Web interface** - UI completa e intuitiva
- ✅ **Async processing** - Processamento não-bloqueante
- ✅ **Smart caching** - Performance otimizada (45-55 FPS)
- ✅ **Health monitoring** - Monitoramento automático
- 🚫 **Training disabled** - Modo produção (CI/CD planejado)

## 🤝 **Suporte**

### **Informações do Projeto**
- **Nome**: Vox Vehicle Color Detection System
- **Versão**: 2.0.0
- **Modo**: Produção (Treinamento Desativado)
- **Framework**: YOLOv11 + ByteTrack
- **API**: FastAPI + Interface Web

### **Suporte Técnico**
Para questões técnicas, consulte:
- 📚 **Documentação**: `/docs` endpoint da API
- 🔧 **Config**: Arquivo `config.json` centralizadoFIRST

## 📄 **Licença**

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 **Agradecimentos**

- **Ultralytics** - YOLO11 implementation
- **ByteTrack** - Multi-object tracking
- **FastAPI** - Modern web framework
- **OpenCV** - Computer vision library
- **PyTorch** - Deep learning framework

---

<div align="center">

**🚗 Vox Vehicle Color Detection System - Built with ❤️ for automotive AI**

[![Performance](https://img.shields.io/badge/Performance-45--55%20FPS-green.svg)]()
[![Models](https://img.shields.io/badge/Models-YOLO11n%2Fs-blue.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Production%20Ready-blue.svg)]()

</div>
