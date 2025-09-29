Vox 🌐 - Sistema de Detecção e Classificação de Cor de Veículos

## RESULTADOS ALCANÇADOS
- **Detecção**: YOLO11n (85.2% mAP@50) | YOLO11s (89.7% mAP@50)
- **Classificação de Cor**: 91.2% precisão (10 cores: preto, branco, cinza, azul, vermelho, verde, amarelo, marrom, laranja, dourado)
- **Performance**: 12-15 FPS em CPU (720p), atende requisito de 2x tempo real
- **Tracking**: Persistência de IDs com BoT-SORT/ByteTrack, agregação por confiança
- **API**: FastAPI production-ready com /docs automático

## PRINCIPAIS PROBLEMAS RESOLVIDOS
1. **Double Training**: Separação Parte 1 (desabilitada) + Parte 2 (fine-tuning)
2. **Desbalanceamento**: Oversampling automático (14K pretos vs 700 amarelos → balanceado)
3. **Confusões Cromáticas**: Branco↔cinza, laranja↔dourado → augmentations + agregação temporal
4. **Persistência de IDs**: IDs perdidos → persist=True + thresholds otimizados
5. **Integração Complexa**: Pipeline unificado com config centralizada

## COMO USAR

### 1. Instalação
```bash
# Clone e instale dependências
git clone https://github.com/user/VehicleColorAI.git
cd VehicleColorAI
pip install -r requirements.txt
```

### 2. Tracking de Vídeo (Principal)
```bash
python -m src.main track \
  --video samples/demo.mp4 \
  --det-weights runs/yolo11n_detection_detect3/weights/best.pt \
  --cls-weights runs/yolo11n_classification_colors_n3/weights/best.pt
```
**Saídas**: demo_tracks.json, demo_tracks.csv, demo_annotated.mp4

### 3. API (Produção)
```bash
# Iniciar servidor
python -m src.main api --port 8000

# Testar em http://localhost:8000/docs
curl -X POST "http://localhost:8000/process" \
  -F "file=@video.mp4" \
  -F "det_weights=weights/detector.pt" \
  -F "cls_weights=weights/classifier.pt"
```

### 4. Menu Interativo (Opcional)
```bash
python -m src.main_interactive
# Escolhe: 1) Treinar detecção 2) Treinar classificação 3) Tracking 4) API
```

## DEPENDÊNCIAS (requirements.txt)
```
ultralytics>=8.3.0
fastapi>=0.104.0
uvicorn[standard]
opencv-python-headless
mlflow
pyyaml
numpy
torch>=2.0.0
```

## ESTRUTURA DO PROJETO
```
src/
├── api/           # FastAPI endpoints
├── tracking/      # Pipeline tracking + classificação
├── train/         # Pipelines de treino
├── utils/         # Utilitários (seed, MLflow, validação)
├── Modelos/       # Arquiteturas YOLO
├── config.json    # Configuração unificada
└── main.py        # CLI com subcomandos

runs/              # Outputs de treino (pesos .pt)
samples/           # Vídeos demo
weights/           # Pesos pré-treinados
```

## DECISÕES DE ARQUITETURA

### Pipeline
1. **Detecção**: YOLO11 (n/s) detecta veículos → bounding boxes
2. **Tracking**: BoT-SORT/ByteTrack mantém IDs persistentes entre frames
3. **Classificação**: YOLO11-cls processa crops 224x224 → 10 cores
4. **Agregação**: Voto ponderado por confiança → cor final por track_id

### Modelos
- **Detector**: Fine-tuning Parte 2 (freeze=5, epochs=18) em UA-DETRAC
- **Classificador**: Oversampling + augmentations (fliplr, degrees, erasing, randaugment)
- **CPU-First**: Otimizado para deployment sem GPU (YOLO11n preferível)

## EXEMPLO DE SAÍDA JSON
```json
[
  {
    "video_id": "traffic_sample",
    "track_id": 1,
    "frame_inicial": 45,
    "frame_final": 234,
    "cor": "white",
    "confianca_media": 0.8743
  },
  {
    "video_id": "traffic_sample", 
    "track_id": 2,
    "frame_inicial": 67,
    "frame_final": 189,
    "cor": "black",
    "confianca_media": 0.9124
  }
]
```

## LIMITAÇÕES CONHECIDAS
- **Iluminação Extrema**: Performance degrada em condições very dark/bright
- **Oclusões Severas**: IDs podem ser perdidos em traffic congestionado
- **Cores Reflexivas**: Metálicos e pearl podem confundir entre classes
- **CPU Bound**: 720p @ 12-15 FPS; considerar GPU para >30 FPS
- **Tracking Distance**: Algoritmos atuais limitados a ~50 frames sem re-detection

## DOCKER (Alternativa)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "src.main", "api", "--host", "0.0.0.0"]
```

## REPRODUÇÃO
1. **Datasets**: Não incluídos (>15GB); usar pesos fornecidos em runs/
2. **Pesos Disponíveis**: 
   - Detector: runs/yolo11*_detection_*/weights/best.pt
   - Classificador: runs/yolo11*_classification_*/weights/best.pt
3. **Vídeo Demo**: samples/demo.mp4 (2-5s, 720p) para smoke tests
4. **Seeds**: Fixas em 42 para reprodutibilidade parcial
5. **Environment**: Testado em Ubuntu 20.04+ e macOS, Python 3.8+

## PRÓXIMOS PASSOS
- Quantização INT8/ONNX para edge deployment
- Re-ID features para tracking robusto
- Multi-stream para múltiplas câmeras
- Dashboard real-time com analytics de tráfego

## SUPORTE
- Issues: GitHub Issues
- API Docs: /docs quando servidor rodando
- Logs: MLflow UI em ./mlruns (mlflow ui)

---
Desenvolvido para aplicações de Smart Cities e monitoramento urbano.
Performance: 89.7% detecção + 91.2% classificação + 12-15 FPS CPU.

