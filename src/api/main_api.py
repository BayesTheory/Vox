from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import tempfile, shutil, time, threading, json, os
from typing import Optional, Dict, List
from uuid import uuid4

import numpy as np
import cv2
from ultralytics import YOLO
from src.tracking.track import process_video_tracking

app = FastAPI(
    title="Vox Color Detection API",
    version="2.0.0",
    description="API para tracking de veículos e classificação de cor - Múltiplos carros"
)

# Configurações
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

JOBS: Dict[str, Dict] = {}

#TUDO JUNTO É PÉSSIMO(CSS+HTML+JS) MAS FOI FEITO AS PRESSAS ESSA PARTE DO CÓDIGO
# ==================== UI COMPLETA ATUALIZADA ====================
HTML_UI = """
<!doctype html>
<html lang="pt-br">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Vox Color Detection - API (Multi-Car)</title>
    <style>
        :root { 
            color-scheme: light dark;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --success: #059669;
            --danger: #dc2626;
            --warning: #d97706;
            --border: #e5e7eb;
            --bg-card: #f9fafb;
            --text: #111827;
            --text-light: #6b7280;
        }
        
        * { box-sizing: border-box; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; line-height: 1.6; 
            background: #ffffff; color: var(--text);
        }
        
        .container { max-width: 1200px; margin: 0 auto; }
        
        h1 { color: var(--primary); margin-bottom: 30px; text-align: center; }
        h2 { color: var(--text); border-bottom: 2px solid var(--border); padding-bottom: 10px; }
        
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: box-shadow 0.2s;
        }
        .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: 600; color: var(--text); }
        
        input[type="text"], input[type="file"], input[type="number"] {
            width: 100%; padding: 10px 12px; border: 2px solid var(--border);
            border-radius: 8px; font-size: 14px; transition: border-color 0.2s;
        }
        input:focus { outline: none; border-color: var(--primary); }
        
        .btn {
            padding: 10px 20px; margin-right: 10px; margin-bottom: 10px;
            border: none; border-radius: 8px; font-size: 14px; font-weight: 600;
            cursor: pointer; text-decoration: none; display: inline-block;
            text-align: center; transition: all 0.2s;
        }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: var(--primary-hover); }
        .btn-secondary { background: #6b7280; color: white; }
        .btn-secondary:hover { background: #4b5563; }
        .btn-danger { background: var(--danger); color: white; }
        .btn-danger:hover { background: #b91c1c; }
        
        .progress-container { margin: 15px 0; }
        .progress-label { font-size: 14px; margin-bottom: 5px; }
        .progress-bar {
            width: 100%; height: 8px; background: #e5e7eb;
            border-radius: 4px; overflow: hidden;
        }
        .progress-fill {
            height: 100%; background: var(--success);
            width: 0%; transition: width 0.3s;
        }
        
        .result-card {
            margin-top: 20px; padding: 20px;
            background: white; border: 1px solid var(--border);
            border-radius: 8px; display: none;
        }
        
        .color-chip {
            display: inline-block; padding: 5px 12px;
            border-radius: 20px; font-size: 12px; font-weight: 600;
            margin: 2px; border: 1px solid #ccc;
        }
        
        .img-preview {
            max-width: 100%; max-height: 300px;
            border: 1px solid var(--border); border-radius: 8px;
            margin: 10px 0; display: none;
        }
        
        .video-player {
            width: 100%; max-height: 400px;
            border: 1px solid var(--border); border-radius: 8px;
        }
        
        .history-item {
            background: white; border: 1px solid var(--border);
            border-radius: 8px; padding: 15px; margin-bottom: 10px;
        }
        
        .stats { display: flex; gap: 20px; margin: 10px 0; }
        .stat { text-align: center; }
        .stat-value { font-size: 20px; font-weight: bold; color: var(--primary); }
        .stat-label { font-size: 12px; color: var(--text-light); }
        
        .alert {
            padding: 12px 16px; border-radius: 8px; margin: 10px 0;
            border-left: 4px solid;
        }
        .alert-success { background: #ecfdf5; border-color: var(--success); color: #065f46; }
        .alert-error { background: #fef2f2; border-color: var(--danger); color: #991b1b; }
        
        .loading { opacity: 0.6; pointer-events: none; }
        
        .car-detection {
            border: 2px solid var(--border); border-radius: 8px;
            padding: 15px; margin: 10px 0; background: white;
        }
        
        .car-detection h4 {
            margin-top: 0; color: var(--primary);
            display: flex; align-items: center; gap: 10px;
        }
        
        .confidence-bar {
            height: 6px; background: #e5e7eb; border-radius: 3px;
            overflow: hidden; margin: 5px 0;
        }
        
        .confidence-fill {
            height: 100%; background: var(--success);
            border-radius: 3px; transition: width 0.3s;
        }
        
        .bbox-info {
            font-size: 12px; color: var(--text-light);
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 VOX Detection API - Multi-Car</h1>
        
        <div class="grid">
            <!-- Classificação de Imagem -->
            <div class="card">
                <h2>📸 Classificar Imagem (Múltiplos Carros)</h2>
                <form id="img-form">
                    <div class="form-group">
                        <label>Imagem (JPG/PNG):</label>
                        <input type="file" name="file" id="img-input" accept="image/*" required/>
                    </div>
                    <div class="form-group">
                        <label>Detector:</label>
                        <input type="text" name="det_weights" value="runs/yolo11n_detection_detect3/weights/best.pt"/>
                    </div>
                    <div class="form-group">
                        <label>Classificador:</label>
                        <input type="text" name="cls_weights" value="runs/yolo11s_classification_colors_s3/weights/best.pt"/>
                    </div>
                    <div class="form-group">
                        <label>Device:</label>
                        <input type="text" name="device" value="cpu"/>
                    </div>
                    <div class="form-group">
                        <label>Confiança mínima detecção:</label>
                        <input type="number" name="det_conf" value="0.25" min="0.1" max="1.0" step="0.05"/>
                    </div>
                    <div class="form-group">
                        <label>Confiança mínima classificação:</label>
                        <input type="number" name="cls_conf" value="0.3" min="0.1" max="1.0" step="0.05"/>
                    </div>
                    <button type="submit" class="btn btn-primary">🔍 Classificar Todos</button>
                    <button type="button" id="img-clear" class="btn btn-secondary">🗑️ Limpar</button>
                </form>
                <img id="img-preview" class="img-preview"/>
                <div id="img-result" class="result-card"></div>
            </div>
            
            <!-- Processamento de Vídeo -->
            <div class="card">
                <h2>🎥 Processar Vídeo</h2>
                <form id="vid-form">
                    <div class="form-group">
                        <label>Vídeo (MP4):</label>
                        <input type="file" name="file" accept="video/mp4" required/>
                    </div>
                    <div class="form-group">
                        <label>Detector:</label>
                        <input type="text" name="det_weights" value="runs/yolo11n_detection_detect3/weights/best.pt"/>
                    </div>
                    <div class="form-group">
                        <label>Classificador:</label>
                        <input type="text" name="cls_weights" value="runs/yolo11s_classification_colors_s3/weights/best.pt"/>
                    </div>
                    <div class="form-group">
                        <label>Config:</label>
                        <input type="text" name="config_path" value="config.json"/>
                    </div>
                    
                    <div class="progress-container">
                        <div class="progress-label">Upload: <span id="upload-pct">0%</span></div>
                        <div class="progress-bar"><div id="upload-bar" class="progress-fill"></div></div>
                    </div>
                    
                    <div class="progress-container">
                        <div class="progress-label">Processamento: <span id="process-pct">0%</span> | ETA: <span id="process-eta">--</span>s</div>
                        <div class="progress-bar"><div id="process-bar" class="progress-fill"></div></div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">🚀 Processar</button>
                    <button type="button" id="vid-clear" class="btn btn-secondary">🗑️ Limpar</button>
                </form>
                <div id="vid-result" class="result-card"></div>
            </div>
        </div>
        
        <!-- Histórico -->
        <div class="card">
            <h2>📋 Histórico de Processamentos</h2>
            <div id="history-container">
                <div class="stats" id="stats"></div>
                <div id="history-list"></div>
            </div>
        </div>
        
        <!-- Links Úteis -->
        <div class="card">
            <h2>🔗 API Documentation</h2>
            <a href="/docs" target="_blank" class="btn btn-primary">📚 Swagger UI</a>
            <a href="/openapi.json" target="_blank" class="btn btn-secondary">📄 OpenAPI JSON</a>
            <a href="/health" target="_blank" class="btn btn-secondary">💊 Health Check</a>
        </div>
    </div>

    <script>
        // Utilitários
        function showAlert(message, type = 'success') {
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.textContent = message;
            document.body.appendChild(alert);
            setTimeout(() => alert.remove(), 5000);
        }

        function getColorStyle(colorName) {
            const colors = {
                'preto': { bg: '#000000', fg: '#ffffff' },
                'branco': { bg: '#ffffff', fg: '#111111' },
                'cinza_prata': { bg: '#c0c0c0', fg: '#111111' },
                'cinza': { bg: '#808080', fg: '#ffffff' },
                'azul': { bg: '#1e40af', fg: '#ffffff' },
                'vermelho': { bg: '#b91c1c', fg: '#ffffff' },
                'verde': { bg: '#166534', fg: '#ffffff' },
                'amarelo': { bg: '#fde047', fg: '#111111' },
                'marrom': { bg: '#7f5539', fg: '#ffffff' },
                'dourado': { bg: '#daa520', fg: '#111111' },
                'laranja': { bg: '#ea580c', fg: '#ffffff' },
                'prata': { bg: '#c0c0c0', fg: '#111111' },
                'rosa': { bg: '#ec4899', fg: '#ffffff' }
            };
            const name = (colorName || '').toLowerCase();
            for (const [key, value] of Object.entries(colors)) {
                if (name.includes(key)) return value;
            }
            return { bg: '#e5e7eb', fg: '#111111' };
        }

        // Preview da imagem
        const imgInput = document.getElementById('img-input');
        const imgPreview = document.getElementById('img-preview');
        
        imgInput.addEventListener('change', () => {
            const file = imgInput.files?.[0];
            if (!file) {
                imgPreview.style.display = 'none';
                return;
            }
            imgPreview.src = URL.createObjectURL(file);
            imgPreview.style.display = 'block';
        });

        // Classificação de imagem - MÚLTIPLOS CARROS
        document.getElementById('img-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const form = e.target;
            const formData = new FormData(form);
            const resultDiv = document.getElementById('img-result');
            
            form.classList.add('loading');
            resultDiv.style.display = 'none';
            
            try {
                const response = await fetch('/classify-image-multi', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail || 'Erro na classificação');
                }
                
                // Construir HTML para múltiplos carros
                let carsHtml = '';
                if (data.cars && data.cars.length > 0) {
                    carsHtml = data.cars.map((car, index) => {
                        const colorStyle = getColorStyle(car.color);
                        const confPercent = (car.confidence * 100).toFixed(1);
                        return `
                            <div class="car-detection">
                                <h4>🚗 Carro ${index + 1}</h4>
                                <div><strong>Cor:</strong> 
                                    <span class="color-chip" style="background:${colorStyle.bg};color:${colorStyle.fg};">${car.color}</span>
                                </div>
                                <div><strong>Confiança:</strong> ${confPercent}%</div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${confPercent}%"></div>
                                </div>
                                <div class="bbox-info">BBox: [${car.bbox.join(', ')}]</div>
                            </div>
                        `;
                    }).join('');
                }
                
                resultDiv.innerHTML = `
                    <h3>✅ Resultado da Classificação Multi-Car</h3>
                    <div><strong>Arquivo:</strong> ${data.filename}</div>
                    <div><strong>Carros detectados:</strong> ${data.total_cars}</div>
                    <div><strong>Carros classificados:</strong> ${data.classified_cars}</div>
                    
                    ${carsHtml}
                    
                    <div style="margin-top: 15px;">
                        <a href="${data.annotated_image_url}" target="_blank" class="btn btn-primary">📥 Baixar Imagem Anotada</a>
                    </div>
                `;
                resultDiv.style.display = 'block';
                showAlert(`${data.total_cars} carros detectados, ${data.classified_cars} classificados!`);
                
            } catch (error) {
                showAlert(error.message, 'error');
            } finally {
                form.classList.remove('loading');
            }
        });

        // Limpar formulário de imagem
        document.getElementById('img-clear').addEventListener('click', () => {
            document.getElementById('img-form').reset();
            imgPreview.style.display = 'none';
            document.getElementById('img-result').style.display = 'none';
        });

        // Processamento de vídeo
        const uploadBar = document.getElementById('upload-bar');
        const uploadPct = document.getElementById('upload-pct');
        const processBar = document.getElementById('process-bar');
        const processPct = document.getElementById('process-pct');
        const processEta = document.getElementById('process-eta');

        function resetProgress() {
            uploadBar.style.width = '0%';
            uploadPct.textContent = '0%';
            processBar.style.width = '0%';
            processPct.textContent = '0%';
            processEta.textContent = '--';
        }

        document.getElementById('vid-form').addEventListener('submit', (e) => {
            e.preventDefault();
            const form = e.target;
            const formData = new FormData(form);
            const resultDiv = document.getElementById('vid-result');
            
            form.classList.add('loading');
            resultDiv.style.display = 'none';
            resetProgress();

            const xhr = new XMLHttpRequest();

            // Progress do upload
            xhr.upload.onprogress = (evt) => {
                if (evt.lengthComputable) {
                    const pct = (evt.loaded / evt.total) * 100;
                    uploadBar.style.width = pct + '%';
                    uploadPct.textContent = pct.toFixed(1) + '%';
                }
            };

            xhr.onload = () => {
                try {
                    const data = JSON.parse(xhr.responseText);
                    if (!data.job_id) {
                        throw new Error(data.detail || xhr.responseText);
                    }

                    const jobId = data.job_id;
                    showAlert('Upload concluído! Iniciando processamento...');

                    // Polling do progresso
                    const pollInterval = setInterval(async () => {
                        try {
                            const statusResponse = await fetch(`/status/${jobId}`);
                            const status = await statusResponse.json();

                            const progress = status.progress || 0;
                            const eta = status.eta || '--';

                            processBar.style.width = progress + '%';
                            processPct.textContent = progress.toFixed(1) + '%';
                            processEta.textContent = eta;

                            if (status.status === 'done') {
                                clearInterval(pollInterval);
                                form.classList.remove('loading');

                                const result = status.result;
                                const colorChips = Object.entries(result.summary || {})
                                    .map(([color, count]) => {
                                        const style = getColorStyle(color);
                                        return `<span class="color-chip" style="background:${style.bg};color:${style.fg};">${color}: ${count}</span>`;
                                    }).join(' ');

                                resultDiv.innerHTML = `
                                    <h3>✅ Processamento Concluído</h3>
                                    <div><strong>Total de Tracks:</strong> ${result.total_tracks}</div>
                                    <div style="margin: 10px 0;">${colorChips}</div>
                                    <div style="margin-top: 15px;">
                                        <a href="${result.json_url}" target="_blank" class="btn btn-primary">📄 JSON</a>
                                        <a href="${result.csv_url}" target="_blank" class="btn btn-primary">📊 CSV</a>
                                        <a href="${result.video_url}" target="_blank" class="btn btn-primary">🎥 Vídeo</a>
                                    </div>
                                `;
                                resultDiv.style.display = 'block';
                                showAlert('Vídeo processado com sucesso!');
                                loadHistory();
                            }

                            if (status.status === 'error') {
                                clearInterval(pollInterval);
                                form.classList.remove('loading');
                                throw new Error(status.error);
                            }

                        } catch (error) {
                            clearInterval(pollInterval);
                            form.classList.remove('loading');
                            showAlert('Erro no processamento: ' + error.message, 'error');
                        }
                    }, 2000);

                } catch (error) {
                    form.classList.remove('loading');
                    showAlert('Erro: ' + error.message, 'error');
                }
            };

            xhr.onerror = () => {
                form.classList.remove('loading');
                showAlert('Erro de conexão', 'error');
            };

            xhr.open('POST', '/process_async');
            xhr.send(formData);
        });

        // Limpar formulário de vídeo
        document.getElementById('vid-clear').addEventListener('click', () => {
            document.getElementById('vid-form').reset();
            document.getElementById('vid-result').style.display = 'none';
            resetProgress();
        });

        // Carregar histórico
        async function loadHistory() {
            try {
                const response = await fetch('/history');
                const data = await response.json();
                const items = data.items || [];

                // Estatísticas
                const totalJobs = items.length;
                const totalTracks = items.reduce((sum, item) => sum + (item.total_tracks || 0), 0);
                const recentJobs = items.filter(item => 
                    (Date.now() / 1000 - item.created_at) < 7 * 24 * 3600
                ).length;

                document.getElementById('stats').innerHTML = `
                    <div class="stat">
                        <div class="stat-value">${totalJobs}</div>
                        <div class="stat-label">Total Jobs</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${totalTracks}</div>
                        <div class="stat-label">Total Tracks</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${recentJobs}</div>
                        <div class="stat-label">Esta Semana</div>
                    </div>
                `;

                // Lista de histórico
                const historyHTML = items.map(item => {
                    const date = new Date(item.created_at * 1000).toLocaleString('pt-BR');
                    const colorChips = Object.entries(item.summary || {})
                        .map(([color, count]) => {
                            const style = getColorStyle(color);
                            return `<span class="color-chip" style="background:${style.bg};color:${style.fg};">${color}: ${count}</span>`;
                        }).join(' ');

                    return `
                        <div class="history-item">
                            <div><strong>ID:</strong> ${item.id}</div>
                            <div><strong>Data:</strong> ${date}</div>
                            <div><strong>Tracks:</strong> ${item.total_tracks || 0}</div>
                            <div style="margin: 10px 0;">${colorChips}</div>
                            <div>
                                <a href="${item.json_url}" target="_blank" class="btn btn-secondary">JSON</a>
                                <a href="${item.csv_url}" target="_blank" class="btn btn-secondary">CSV</a>
                                <a href="${item.video_url}" target="_blank" class="btn btn-secondary">Vídeo</a>
                                <button onclick="deleteJob('${item.id}')" class="btn btn-danger">Excluir</button>
                            </div>
                        </div>
                    `;
                }).join('');

                document.getElementById('history-list').innerHTML = historyHTML || '<p>Nenhum processamento encontrado.</p>';

            } catch (error) {
                showAlert('Erro ao carregar histórico: ' + error.message, 'error');
            }
        }

        // Excluir job
        async function deleteJob(jobId) {
            if (!confirm('Tem certeza que deseja excluir este processamento?')) return;

            try {
                await fetch(`/delete/${jobId}`, { method: 'POST' });
                showAlert('Processamento excluído com sucesso!');
                loadHistory();
            } catch (error) {
                showAlert('Erro ao excluir: ' + error.message, 'error');
            }
        }

        // Carregar histórico na inicialização
        loadHistory();

        // Auto-refresh do histórico a cada 30 segundos
        setInterval(loadHistory, 30000);
    </script>
</body>
</html>
"""

# ==================== MODELOS ====================
class HealthResponse(BaseModel):
    status: str
    message: str

class TrackingResponse(BaseModel):
    json_path: str
    csv_path: str
    video_annotated: str
    total_tracks: int
    summary: dict
    json_url: Optional[str] = None
    csv_url: Optional[str] = None
    video_url: Optional[str] = None

class CarDetection(BaseModel):
    color: str
    confidence: float
    bbox: List[int]

class MultiCarResponse(BaseModel):
    filename: str
    total_cars: int
    classified_cars: int
    cars: List[CarDetection]
    annotated_image_path: str
    annotated_image_url: str

# ==================== ROTAS BÁSICAS ====================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui", status_code=307)

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=HTML_UI, status_code=200)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="API funcionando normalmente - Multi-Car Support")

# ==================== HISTÓRICO ====================
def list_history() -> List[Dict]:
    items = []
    for path in sorted(RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        meta_file = path / "meta.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                items.append(data)
            except:
                continue
    return items

@app.get("/history")
def history():
    return {"items": list_history()}

@app.post("/delete/{item_id}")
def delete_item(item_id: str):
    folder = RESULTS_DIR / item_id
    if not folder.exists():
        raise HTTPException(404, "Item não encontrado")
    shutil.rmtree(str(folder), ignore_errors=True)
    return {"deleted": item_id}

# ==================== CLASSIFICAÇÃO DE MÚLTIPLOS CARROS ====================
@app.post("/classify-image-multi", response_model=MultiCarResponse)
async def classify_image_multi(
    file: UploadFile = File(...),
    det_weights: str = Form("runs/yolo11n_detection_detect3/weights/best.pt"),
    cls_weights: str = Form("runs/yolo11s_classification_colors_s3/weights/best.pt"),
    device: str = Form("cpu"),
    det_conf: float = Form(0.25),
    cls_conf: float = Form(0.3)
):
    """✅ NOVA ROTA - DETECTA E CLASSIFICA TODOS OS CARROS NA IMAGEM"""
    
    # Validação
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        raise HTTPException(400, "Formato não suportado. Use: jpg, jpeg, png, bmp, webp")
    
    # Lê e decodifica imagem
    img_bytes = await file.read()
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        raise HTTPException(400, "Não foi possível decodificar a imagem")

    # Carrega detector
    try:
        detector = YOLO(det_weights)
    except Exception as e:
        raise HTTPException(400, f"Erro carregando detector: {e}")

    # ✅ DETECTA TODOS OS VEÍCULOS (não só o maior)
    det_results = detector.predict(
        source=img_np, 
        imgsz=640, 
        device=device, 
        conf=det_conf,  # Usar confiança configurável
        iou=0.7, 
        verbose=False
    )[0]
    
    if det_results.boxes is None or len(det_results.boxes.xyxy) == 0:
        raise HTTPException(400, f"Nenhum veículo detectado na imagem (conf >= {det_conf})")

    # ✅ PROCESSAR TODAS AS DETECÇÕES
    xyxy = det_results.boxes.xyxy.cpu().numpy() if hasattr(det_results.boxes.xyxy, "cpu") else np.asarray(det_results.boxes.xyxy)
    confidences = det_results.boxes.conf.cpu().numpy() if hasattr(det_results.boxes.conf, "cpu") else np.asarray(det_results.boxes.conf)
    
    total_cars = len(xyxy)
    cars_data = []
    
    # Carrega classificador
    try:
        classifier = YOLO(cls_weights)
    except Exception as e:
        raise HTTPException(400, f"Erro carregando classificador: {e}")

    # ✅ IMAGEM ANOTADA - PREPARAR CANVAS
    annotated = img_np.copy()
    
    # ✅ PROCESSAR CADA DETECÇÃO
    for i, (bbox, det_confidence) in enumerate(zip(xyxy, confidences)):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validar bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_np.shape[1], x2)
        y2 = min(img_np.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue  # Skip bbox inválido
        
        # Extrair crop
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        try:
            # ✅ CLASSIFICAR COR DO CROP
            cls_results = classifier.predict(
                source=crop, 
                imgsz=224, 
                device=device, 
                verbose=False
            )[0]
            
            top1_idx = int(cls_results.probs.top1)
            top1_conf = float(cls_results.probs.top1conf.cpu().item()) if hasattr(cls_results.probs.top1conf, "cpu") else float(cls_results.probs.top1conf)
            color_name = classifier.names[top1_idx]
            
            # ✅ APENAS ADICIONAR SE CONFIANÇA SUFICIENTE
            if top1_conf >= cls_conf:
                cars_data.append(CarDetection(
                    color=color_name,
                    confidence=round(top1_conf, 4),
                    bbox=[x1, y1, x2, y2]
                ))
                
                # ✅ ANOTAR NA IMAGEM - COR BASEADA NO ÍNDICE
                colors = [
                    (0, 255, 0),    # Verde
                    (255, 0, 0),    # Azul
                    (0, 0, 255),    # Vermelho
                    (255, 255, 0),  # Ciano
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Amarelo
                    (128, 0, 128),  # Roxo
                    (255, 165, 0),  # Laranja
                    (0, 128, 255),  # Azul claro
                    (128, 255, 0),  # Verde lima
                ]
                color_bgr = colors[len(cars_data) - 1 % len(colors)]
                
                # Desenhar bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3)
                
                # Label
                label = f"#{len(cars_data)} {color_name} ({top1_conf:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                ty = max(25, y1 - 8)
                
                # Fundo do texto
                cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), color_bgr, -1)
                cv2.putText(annotated, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Erro classificando carro {i+1}: {e}")
            continue
    
    classified_cars = len(cars_data)
    
    # ✅ SALVAR IMAGEM ANOTADA
    timestamp = int(time.time())
    out_name = f"multi_car_{timestamp}_{uuid4().hex[:6]}_annotated.jpg"
    out_path = RESULTS_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    # ✅ RETORNAR DADOS ESTRUTURADOS
    return MultiCarResponse(
        filename=file.filename,
        total_cars=total_cars,
        classified_cars=classified_cars,
        cars=cars_data,
        annotated_image_path=str(out_path),
        annotated_image_url=f"/results/{out_name}"
    )

# ==================== CLASSIFICAÇÃO ORIGINAL (MANTIDA PARA COMPATIBILIDADE) ====================
@app.post("/classify-image")
async def classify_image_single(
    file: UploadFile = File(...),
    det_weights: str = Form("runs/yolo11n_detection_detect3/weights/best.pt"),
    cls_weights: str = Form("runs/yolo11s_classification_colors_s3/weights/best.pt"),
    device: str = Form("cpu")
):
    """Classificação original - apenas o carro com maior área (mantida para compatibilidade)"""
    
    # Validação
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        raise HTTPException(400, "Formato não suportado. Use: jpg, jpeg, png, bmp, webp")
    
    # Lê e decodifica imagem
    img_bytes = await file.read()
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        raise HTTPException(400, "Não foi possível decodificar a imagem")

    # Carrega detector
    try:
        detector = YOLO(det_weights)
    except Exception as e:
        raise HTTPException(400, f"Erro carregando detector: {e}")

    # Detecta veículos
    det_results = detector.predict(source=img_np, imgsz=640, device=device, conf=0.25, iou=0.7, verbose=False)[0]
    if det_results.boxes is None or len(det_results.boxes.xyxy) == 0:
        raise HTTPException(400, "Nenhum veículo detectado na imagem")

    # Pega maior detecção
    xyxy = det_results.boxes.xyxy.cpu().numpy() if hasattr(det_results.boxes.xyxy, "cpu") else np.asarray(det_results.boxes.xyxy)
    areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = map(int, xyxy[idx])

    # Carrega classificador
    try:
        classifier = YOLO(cls_weights)
    except Exception as e:
        raise HTTPException(400, f"Erro carregando classificador: {e}")

    # Classifica crop
    crop = img_np[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop.size == 0:
        raise HTTPException(400, "Crop inválido")

    cls_results = classifier.predict(source=crop, imgsz=224, device=device, verbose=False)[0]
    top1_idx = int(cls_results.probs.top1)
    top1_conf = float(cls_results.probs.top1conf.cpu().item()) if hasattr(cls_results.probs.top1conf, "cpu") else float(cls_results.probs.top1conf)
    color_name = classifier.names[top1_idx]

    # Salva imagem anotada
    out_name = f"img_{int(time.time())}_{uuid4().hex[:6]}_annotated.jpg"
    out_path = RESULTS_DIR / out_name
    
    annotated = img_np.copy()
    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 3)
    label = f"{color_name} ({top1_conf:.2f})"
    (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    ty = max(25, y1-8)
    cv2.rectangle(annotated, (x1, ty-th-6), (x1+tw+10, ty+4), (0,255,0), -1)
    cv2.putText(annotated, label, (x1+5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.imwrite(str(out_path), annotated)

    return {
        "filename": file.filename,
        "color": color_name,
        "confidence": round(top1_conf, 4),
        "bbox": [x1, y1, x2, y2],
        "annotated_image_path": str(out_path),
        "annotated_image_url": f"/results/{out_name}"
    }

# ==================== PROCESSAMENTO ASSÍNCRONO ====================
def _video_processing_job(job_id: str, temp_video: str, det_weights: str, cls_weights: str, config_path: str):
    try:
        # Conta frames para progresso
        cap = cv2.VideoCapture(temp_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        
        JOBS[job_id].update({"status": "running", "progress": 0.0, "eta": None})
        start_time = time.time()

        def progress_callback(processed_frames: int, total_frames_to_process: int):
            if total_frames_to_process > 0:
                progress = min(100.0, (processed_frames / total_frames_to_process) * 100)
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0
                remaining_frames = total_frames_to_process - processed_frames
                eta = int(remaining_frames / fps) if fps > 0 else None
                JOBS[job_id].update({"progress": round(progress, 1), "eta": eta})

        # Cria pasta do job
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Processa vídeo
        result = process_video_tracking(
            video_path=temp_video,
            det_weights=det_weights,
            cls_weights=cls_weights,
            config_path=config_path,
            out_dir=str(job_dir),
            progress_cb=progress_callback,
        )
        
        # Remove arquivo temporário
        Path(temp_video).unlink(missing_ok=True)

        # Constrói URLs
        output_files = result.get("output_files", {})
        json_path = output_files.get("json", "")
        csv_path = output_files.get("csv", "")
        video_path = result.get("video_annotated", "")
        
        # URLs públicas
        json_name = Path(json_path).name if json_path else ""
        csv_name = Path(csv_path).name if csv_path else ""
        video_name = Path(video_path).name if video_path else ""

        result.update({
            "json_url": f"/results/{job_id}/{json_name}" if json_name else "",
            "csv_url": f"/results/{job_id}/{csv_name}" if csv_name else "",
            "video_url": f"/results/{job_id}/{video_name}" if video_name else "",
        })

        # Salva metadata
        meta = {
            "id": job_id,
            "created_at": int(start_time),
            "det_weights": det_weights,
            "cls_weights": cls_weights,
            "json_url": result["json_url"],
            "csv_url": result["csv_url"],
            "video_url": result["video_url"],
            "summary": result.get("color_distribution", {}),
            "total_tracks": result.get("total_tracks", 0)
        }
        (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        JOBS[job_id].update({"status": "done", "result": result, "progress": 100.0, "eta": 0})

    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})
        try:
            Path(temp_video).unlink(missing_ok=True)
        except:
            pass

@app.post("/process_async")
async def process_video_async(
    file: UploadFile = File(...),
    det_weights: str = Form("runs/yolo11n_detection_detect3/weights/best.pt"),
    cls_weights: str = Form("runs/yolo11s_classification_colors_s3/weights/best.pt"),
    config_path: str = Form("config.json"),
):
    # Validações
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(400, "Formato não suportado. Use: mp4, mov, avi")
    
    if not Path(det_weights).exists():
        raise HTTPException(400, f"Detector não encontrado: {det_weights}")
    
    if not Path(cls_weights).exists():
        raise HTTPException(400, f"Classificador não encontrado: {cls_weights}")
    
    if not Path(config_path).exists():
        raise HTTPException(400, f"Config não encontrado: {config_path}")

    # Salva arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_video = tmp.name

    # Cria job
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "eta": None, "created": time.time()}

    # Inicia thread
    thread = threading.Thread(
        target=_video_processing_job,
        args=(job_id, temp_video, det_weights, cls_weights, config_path),
        daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    return {k: v for k, v in job.items() if k != "created"}

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main_api:app", host="127.0.0.1", port=8000, reload=True)