# src/api/main_api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
import tempfile
import shutil
from typing import Optional

from src.tracking.track import process_video_tracking

app = FastAPI(
    title="Vehicle Color Detection API",
    description="API para tracking de veículos e classificação de cor em vídeos",
    version="1.0.0"
)

class HealthResponse(BaseModel):
    status: str
    message: str

class TrackingResponse(BaseModel):
    json_path: str
    csv_path: str
    video_annotated: str
    total_tracks: int
    summary: dict

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Endpoint de saúde da API"""
    return HealthResponse(status="ok", message="API funcionando")

@app.post("/process", response_model=TrackingResponse)
async def process_video(
    file: UploadFile = File(..., description="Arquivo de vídeo .mp4"),
    det_weights: str = Form(..., description="Caminho dos pesos do detector"),
    cls_weights: str = Form(..., description="Caminho dos pesos do classificador"),
    config_path: str = Form("src/config.json", description="Caminho do arquivo de configuração")
):
    """
    Processa vídeo com tracking + classificação de cor.
    Retorna caminhos dos artefatos gerados.
    """
    # Validações
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Apenas vídeos .mp4, .avi, .mov são suportados")
    
    det_path = Path(det_weights)
    cls_path = Path(cls_weights)
    if not det_path.exists():
        raise HTTPException(400, f"Detector não encontrado: {det_weights}")
    if not cls_path.exists():
        raise HTTPException(400, f"Classificador não encontrado: {cls_weights}")
    
    try:
        # Salva upload temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_video = tmp.name
        
        # Processa
        result = process_video_tracking(
            video_path=temp_video,
            det_weights=det_weights,
            cls_weights=cls_weights,
            config_path=config_path
        )
        
        # Cleanup
        Path(temp_video).unlink()
        
        return TrackingResponse(
            json_path=result["json"],
            csv_path=result["csv"],
            video_annotated=result["video_annotated"],
            total_tracks=result["total_tracks"],
            summary=result["summary"]
        )
        
    except Exception as e:
        # Cleanup em caso de erro
        if 'temp_video' in locals():
            Path(temp_video).unlink(missing_ok=True)
        raise HTTPException(500, f"Erro no processamento: {str(e)}")

@app.get("/download/{file_type}/{filename}")
def download_file(file_type: str, filename: str):
    """Download de artefatos gerados (JSON, CSV, MP4)"""
    if file_type not in ["json", "csv", "video"]:
        raise HTTPException(400, "Tipo deve ser: json, csv ou video")
    
    # Lógica de download seria implementada aqui
    # Por segurança, validar caminhos e permissões
    raise HTTPException(501, "Download direto não implementado nesta versão")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
