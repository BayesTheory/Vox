# src/api/main_api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import tempfile, shutil, time, threading, json
from typing import Optional, Dict, List
from uuid import uuid4

import numpy as np
import cv2
from ultralytics import YOLO

from src.tracking.track import process_video_tracking  # detecta->track (ByteTrack)->classifica por box [web:629]

app = FastAPI(
    title="Vehicle Color Detection API",
    version="1.0.0",
    description="API para tracking de veículos e classificação de cor em vídeos e imagens"
)

# Artefatos persistentes publicados em /results
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")  # publica estáticos [web:1074]

# Jobs em memória para progresso
JOBS: Dict[str, Dict] = {}

# ---------- UI INLINE (HTML + CSS + JS) ----------
HTML_UI = """
<!doctype html><html lang="pt-br"><head><meta charset="utf-8"/>
<title>Vehicle Color Detection</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:24px; line-height:1.45; }
section { margin:18px 0; padding:14px; border:1px solid #e5e5e5; border-radius:10px; }
label { display:block; margin:8px 0; }
input[type="text"],input[type="file"]{width:100%;max-width:760px;padding:6px 8px;}
button{padding:8px 14px;margin-right:6px;cursor:pointer;color:#111;background:#fff;border:1px solid #ccc;border-radius:8px;}
a.button{padding:8px 12px;border:1px solid #ccc;border-radius:8px;text-decoration:none;color:#111;background:#fff;margin-right:8px;}
.bar{width:100%;max-width:760px;height:10px;background:#eee;border-radius:8px;overflow:hidden;margin:8px 0;}
.fill{height:100%;width:0%;background:#3b82f6;transition:width .2s;}
.card{border:1px solid #ddd;border-radius:12px;padding:14px;margin-top:12px;}
video{width:100%;max-width:760px;border:1px solid #ddd;border-radius:10px;}
pre{background:#f7f7f7;padding:12px;border-radius:8px;overflow:auto;}
.chips{display:flex;flex-wrap:wrap;gap:6px;}
.chip{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid #ccc;}
</style>
</head><body>
<h1>Vehicle Color Detection</h1>

<section>
  <h2>Classificar Imagem (com caixa)</h2>
  <form id="img-form">
    <label>Imagem (jpg/png)<input type="file" name="file" id="img-input" accept="image/*" required/></label>
    <label>Detector<input type="text" name="det_weights" value="runs/yolo11n_detection_detect3/weights/best.pt"/></label>
    <label>Classificador<input type="text" name="cls_weights" value="C:/Users/Rian/Desktop/Vox/runs/yolo11s_classification_colors_s/weights/best.pt"/></label>
    <label>Device<input type="text" name="device" value="cpu"/></label>
    <button type="submit">Classificar</button><button type="button" id="img-clear">Limpar</button>
  </form>
  <img id="img-preview" style="max-width:360px;display:none;border:1px solid #ddd;border-radius:6px;"/>
  <div id="img-card" class="card" style="display:none;"></div>
</section>

<section>
  <h2>Processar Vídeo (Detecção + Tracking + Cor)</h2>
  <form id="vid-form">
    <label>Vídeo (.mp4)<input type="file" name="file" accept="video/mp4" required/></label>
    <label>Detector<input type="text" name="det_weights" value="runs/yolo11n_detection_detect3/weights/best.pt"/></label>
    <label>Classificador<input type="text" name="cls_weights" value="C:/Users/Rian/Desktop/Vox/runs/yolo11s_classification_colors_s/weights/best.pt"/></label>
    <label>Config<input type="text" name="config_path" value="src/config.json"/></label>
    <div>Upload: <span id="u-pct">0%</span><div class="bar"><div id="u-bar" class="fill"></div></div></div>
    <div>Processamento: <span id="p-pct">0%</span> ETA: <span id="p-eta">--</span>s<div class="bar"><div id="p-bar" class="fill"></div></div></div>
    <button type="submit">Enviar e Processar</button><button type="button" id="vid-clear">Limpar</button>
  </form>
  <div id="vid-card" class="card" style="display:none;"></div>
</section>

<section>
  <h2>Histórico</h2>
  <div id="hist"></div>
</section>

<section><h2>Links úteis</h2><ul><li><a class="button" href="/docs" target="_blank">Swagger /docs</a></li><li><a class="button" href="/openapi.json" target="_blank">OpenAPI JSON</a></li></ul></section>

<script>
// Preview imagem
const imgInput=document.getElementById('img-input'), imgPreview=document.getElementById('img-preview');
imgInput.addEventListener('change',()=>{const f=imgInput.files?.[0]; if(!f){imgPreview.style.display='none';return;} imgPreview.src=URL.createObjectURL(f); imgPreview.style.display='block';});
// Limpar imagem
document.getElementById('img-clear').addEventListener('click',()=>{document.getElementById('img-form').reset(); imgPreview.style.display='none'; document.getElementById('img-card').style.display='none';});

// Mapa de cores com contraste
function colorStyle(name){
  const n=(name||'').toLowerCase();
  const map={
    "preto": {bg:"#000000", fg:"#ffffff"},
    "branco":{bg:"#ffffff", fg:"#111111"},
    "cinza_prata": {bg:"#c0c0c0", fg:"#111111"},
    "prata": {bg:"#c0c0c0", fg:"#111111"},
    "cinza": {bg:"#808080", fg:"#ffffff"},
    "azul":  {bg:"#1e40af", fg:"#ffffff"},
    "vermelho":{bg:"#b91c1c", fg:"#ffffff"},
    "verde": {bg:"#166534", fg:"#ffffff"},
    "amarelo":{bg:"#fde047", fg:"#111111"},
    "marrom":{bg:"#7f5539", fg:"#ffffff"},
    "dourado":{bg:"#daa520", fg:"#111111"},
    "laranja":{bg:"#ea580c", fg:"#ffffff"}
  };
  for (const k of Object.keys(map)){ if(n.includes(k)) return map[k]; }
  return {bg:"#e5e7eb", fg:"#111111"};
}

// Classificar imagem
document.getElementById('img-form').addEventListener('submit',async(e)=>{
  e.preventDefault();
  const fd=new FormData(e.target);
  const r=await fetch('/classify-image',{method:'POST',body:fd});
  const data=await r.json();
  const card=document.getElementById('img-card');
  card.style.display='block';
  const st=colorStyle(data.color||'');
  const link = data.annotated_image_url ? `<a class="button" href="${data.annotated_image_url}" target="_blank">Baixar imagem anotada</a>` : '';
  card.innerHTML = `
    <div><b>Arquivo:</b> ${data.filename||'-'}</div>
    <div><b>Cor:</b> <span class="chip" style="background:${st.bg};color:${st.fg};border-color:#999;">${data.color}</span></div>
    <div><b>Confiança:</b> ${(data.confidence*100).toFixed(2)}%</div>
    ${data.bbox ? `<div><b>Box:</b> [${data.bbox.join(', ')}]</div>` : ''}
    <div style="margin-top:8px;">${link}</div>
  `;
});

// Upload + processamento (vídeo) com barras
const uBar=document.getElementById('u-bar'), uPct=document.getElementById('u-pct'), pBar=document.getElementById('p-bar'), pPct=document.getElementById('p-pct'), pEta=document.getElementById('p-eta');
document.getElementById('vid-clear').addEventListener('click',()=>{document.getElementById('vid-form').reset(); uBar.style.width='0%'; uPct.textContent='0%'; pBar.style.width='0%'; pPct.textContent='0%'; pEta.textContent='--'; document.getElementById('vid-card').style.display='none';});

document.getElementById('vid-form').addEventListener('submit',(e)=>{
  e.preventDefault();
  const fd=new FormData(e.target);
  const xhr=new XMLHttpRequest();
  xhr.upload.onprogress=(evt)=>{ if(evt.lengthComputable){const pct=evt.loaded/evt.total*100; uBar.style.width=pct.toFixed(1)+'%'; uPct.textContent=pct.toFixed(1)+'%'; }}; // progresso de upload [web:1126]
  xhr.onload=()=>{ try{
    const data=JSON.parse(xhr.responseText); if(!data.job_id){alert(xhr.responseText);return;}
    const job=data.job_id;
    const timer=setInterval(async()=>{
      const s=await fetch('/status/'+job).then(r=>r.json()); // polling de status [web:1120]
      const pct=s.progress||0; const eta=s.eta??'--';
      pBar.style.width=pct.toFixed(1)+'%'; pPct.textContent=pct.toFixed(1)+'%'; pEta.textContent=eta;
      if(s.status==='done'){
        clearInterval(timer);
        const r=s.result;
        const chips = Object.entries(r.summary||{}).map(([k,v])=>{ const st=colorStyle(k); return `<span class="chip" style="background:${st.bg};color:${st.fg};border-color:#999;">${k}: ${v}</span>`; }).join(' ');
        const links = `
          <a class="button" href="${r.json_url}" target="_blank">Baixar JSON</a>
          <a class="button" href="${r.csv_url}" target="_blank">Baixar CSV</a>
          <a class="button" href="${r.video_url}" target="_blank">Baixar Vídeo Anotado</a>
        `;
        const card=document.getElementById('vid-card'); card.style.display='block';
        card.innerHTML = `<div><b>Total de tracks:</b> ${r.total_tracks}</div><div class="chips" style="margin:8px 0;">${chips}</div>${links}`;
        loadHist();
      }
      if(s.status==='error'){ clearInterval(timer); alert('Erro: '+s.error); }
    },1000);
  }catch(e){ alert('Erro: '+e); } };
  xhr.open('POST','/process_async'); xhr.send(fd);
});

// Histórico
async function loadHist(){
  const data = await fetch('/history').then(r=>r.json());
  const hist = document.getElementById('hist');
  hist.innerHTML = (data.items||[]).map(it => `
    <div class="card">
      <div><b>ID:</b> ${it.id}</div>
      <div class="chips">${Object.entries(it.summary||{}).map(([k,v])=>{const st=colorStyle(k); return `<span class="chip" style="background:${st.bg};color:${st.fg};border-color:#999;">${k}: ${v}</span>`}).join(' ')}</div>
      <a class="button" href="${it.json_url}" target="_blank">JSON</a>
      <a class="button" href="${it.csv_url}" target="_blank">CSV</a>
      <a class="button" href="${it.video_url}" target="_blank">Vídeo</a>
      <button onclick="fetch('/delete/${it.id}',{method:'POST'}).then(()=>loadHist())">Excluir</button>
    </div>
  `).join('');
}
loadHist();
</script>
</body></html>
"""

# ---------- MODELOS ----------
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

# ---------- ROTAS BÁSICAS ----------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui", status_code=307)  # redireciona raiz para UI [web:1074]

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=HTML_UI, status_code=200)  # serve a página inline [web:1074]

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="API funcionando")  # health simples [web:1074]

# ---------- HISTÓRICO ----------
def list_history() -> List[Dict]:
    items = []
    for p in sorted(RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        meta = p / "meta.json"
        if meta.exists():
            try:
                items.append(json.loads(meta.read_text(encoding="utf-8")))
            except:
                pass
    return items  # lista metadados de jobs [web:1074]

@app.get("/history")
def history():
    return {"items": list_history()}  # expõe histórico para UI [web:1074]

@app.post("/delete/{item_id}")
def delete_item(item_id: str):
    folder = RESULTS_DIR / item_id
    if not folder.exists():
        raise HTTPException(404, "Item não encontrado")
    import shutil as _sh
    _sh.rmtree(str(folder), ignore_errors=True)
    return {"deleted": item_id}  # permite remoção manual de jobs [web:1074]

# ---------- CLASSIFICAÇÃO DE IMAGEM (com detecção p/ box) ----------
@app.post("/classify-image")
async def classify_image(
    file: UploadFile = File(..., description="Imagem do carro (jpg/png)"),
    det_weights: str = Form(..., description="Pesos do detector (best.pt)"),
    cls_weights: str = Form(..., description="Pesos do classificador (-cls.pt ou best.pt)"),
    device: str = Form("cpu", description="cpu ou cuda:0")
):
    # valida extensão
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        raise HTTPException(400, "Formatos suportados: jpg, jpeg, png, bmp, webp")  # valida file [web:1074]
    img_bytes = await file.read()
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        raise HTTPException(400, "Não foi possível decodificar a imagem")  # decodifica para NumPy [web:405]

    # detecta veículo para criar box
    try:
        det = YOLO(det_weights)
    except Exception as e:
        raise HTTPException(400, f"Erro carregando detector: {e}")
    det_res = det.predict(source=img_np, imgsz=640, device=device, conf=0.25, iou=0.7, verbose=False)[0]  # detecções [web:405]
    if det_res.boxes is None or det_res.boxes.xyxy is None or len(det_res.boxes.xyxy) == 0:
        raise HTTPException(400, "Nenhum veículo detectado na imagem")  # sem boxes [web:405]
    xyxy = det_res.boxes.xyxy
    xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
    areas = (xyxy_np[:,2]-xyxy_np[:,0]) * (xyxy_np[:,3]-xyxy_np[:,1])
    idx = int(np.argmax(areas))
    x1,y1,x2,y2 = map(int, xyxy_np[idx])

    # classifica cor no crop
    try:
        cls = YOLO(cls_weights)
    except Exception as e:
        raise HTTPException(400, f"Erro carregando classificador: {e}")
    crop = img_np[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if crop.size == 0:
        raise HTTPException(400, "Crop inválido")  # valida crop [web:405]
    cls_res = cls.predict(source=crop, imgsz=224, device=device, verbose=False)[0]  # classificação [web:405]
    top1_idx = int(cls_res.probs.top1)
    top1_conf = float(cls_res.probs.top1conf.cpu().item()) if hasattr(cls_res.probs.top1conf, "cpu") else float(cls_res.probs.top1conf)
    color_name = cls.names[top1_idx]

    # desenha e salva imagem anotada persistente
    out_name = f"img_{int(time.time())}_{uuid4().hex[:6]}_annotated.jpg"
    out_path = RESULTS_DIR / out_name
    vis = img_np.copy()
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,0), 2)
    label = f"{color_name} {top1_conf:.2f}"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    ty = max(18, y1-8)
    cv2.rectangle(vis, (x1, ty-th-6), (x1+tw+10, ty+4), (0,0,0), -1)
    cv2.putText(vis, label, (x1+5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imwrite(str(out_path), vis)

    return {
        "filename": file.filename,
        "color": color_name,
        "confidence": round(top1_conf, 6),
        "bbox": [x1,y1,x2,y2],
        "annotated_image_path": str(out_path),
        "annotated_image_url": f"/results/{out_name}"
    }  # retorna link público [web:1074]

# ---------- PROCESSAMENTO ASSÍNCRONO COM PROGRESSO ----------
def _video_job(job_id: str, temp_video: str, det_weights: str, cls_weights: str, config_path: str):
    try:
        cap = cv2.VideoCapture(temp_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        JOBS[job_id].update({"status": "running", "progress": 0.0, "eta": None})
        start = time.time()

        def progress_cb(idx: int):
            if total > 0:
                p = min(100.0, 100.0 * (idx + 1) / total)
                elapsed = max(1e-3, time.time() - start)
                fps = (idx + 1) / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total - (idx + 1))
                eta = int(remaining / fps) if fps > 0 else None
                JOBS[job_id].update({"progress": round(p, 1), "eta": eta})  # progresso p/ polling [web:1120]

        # Pasta persistente do job
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        result = process_video_tracking(
            video_path=temp_video,
            det_weights=det_weights,
            cls_weights=cls_weights,
            config_path=config_path,
            out_dir=str(job_dir),
            progress_cb=progress_cb,
        )  # tracking + classificação por box (ByteTrack configurado no track.py) [web:629]
        Path(temp_video).unlink(missing_ok=True)

        # URLs e meta
        json_name = Path(result["json"]).name
        csv_name  = Path(result["csv"]).name
        vid_name  = Path(result["video_annotated"]).name
        result.update({
            "json_url": f"/results/{job_id}/{json_name}",
            "csv_url":  f"/results/{job_id}/{csv_name}",
            "video_url":f"/results/{job_id}/{vid_name}",
        })
        meta = {
            "id": job_id,
            "created_at": int(start),
            "det_weights": det_weights,
            "cls_weights": cls_weights,
            "json_url": result["json_url"],
            "csv_url":  result["csv_url"],
            "video_url": result["video_url"],
            "summary": result.get("summary", {}),
            "total_tracks": result.get("total_tracks", 0)
        }
        (job_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")  # salva meta [web:1074]
        JOBS[job_id].update({"status": "done", "result": result, "progress": 100.0, "eta": 0})
    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})
        try:
            Path(temp_video).unlink(missing_ok=True)
        except:
            pass  # erro tratado [web:1120]

@app.post("/process_async")
async def process_video_async(
    file: UploadFile = File(..., description="Arquivo de vídeo .mp4"),
    det_weights: str = Form(..., description="Pesos do detector (best.pt)"),
    cls_weights: str = Form(..., description="Pesos do classificador (-cls.pt/best.pt)"),
    config_path: str = Form("src/config.json", description="Config"),
):
    if not file.filename.lower().endswith((".mp4",".mov",".avi")):
        raise HTTPException(400, "Apenas .mp4/.mov/.avi")
    if not Path(det_weights).exists():
        raise HTTPException(400, f"Detector não encontrado: {det_weights}")
    if not Path(cls_weights).exists():
        raise HTTPException(400, f"Classificador não encontrado: {cls_weights}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_video = tmp.name  # arquivo temporário do upload [web:1120]

    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "eta": None, "created": time.time()}
    t = threading.Thread(target=_video_job, args=(job_id, temp_video, det_weights, cls_weights, config_path), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued"}  # resposta imediata p/ UI iniciar polling [web:1120]

@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job_id não encontrado")
    return {k: v for k, v in job.items() if k != "created"}  # status para polling [web:1120]

# Execução direta (dev)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main_api:app", host="127.0.0.1", port=8000, reload=True)  # dev com reload [web:1074]
