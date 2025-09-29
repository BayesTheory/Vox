# src/tracking/track.py
from pathlib import Path
from typing import Dict, List, Optional
import json
import csv
import cv2
from collections import defaultdict
from ultralytics import YOLO

def process_video_tracking(
    video_path: str,
    det_weights: str,
    cls_weights: str,
    config_path: str = "src/config.json",
    out_dir: Optional[str] = None,
) -> Dict:
    """
    Tracking de veículos com classificação de cor usando config unificado.
    Retorna paths dos artefatos gerados.
    """
    from src.utils.utils import load_config
    
    cfg = load_config(config_path)
    tracking_cfg = cfg["tracking"]
    
    src = Path(video_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {src}")
    
    out_root = Path(out_dir) if out_dir else src.parent / "tracking_results"
    out_root.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    
    # Carrega modelos
    det_model = YOLO(det_weights)
    cls_model = YOLO(cls_weights)
    
    # Setup vídeo
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Vídeo anotado
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = out_root / f"{stem}_annotated.mp4"
    out_vid = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
    
    # Tracking stream
    stream = det_model.track(
        source=str(src),
        tracker=tracking_cfg["tracker"],
        persist=True,
        stream=True,
        device="cpu",  # Força CPU para deployment
        imgsz=tracking_cfg["det_imgsz"],
        conf=tracking_cfg["conf_thres"],
        iou=tracking_cfg["iou_thres"],
        verbose=False,
        save=False  # Não salva arquivos automáticos
    )
    
    # Agregação por track_id
    tracks = defaultdict(lambda: {
        "start_frame": None,
        "end_frame": None,
        "color_votes": defaultdict(float),
        "confidences": []
    })
    
    frame_idx = -1
    for result in stream:
        frame_idx += 1
        frame = result.orig_img
        boxes = getattr(result, "boxes", None)
        
        if boxes is None or boxes.xyxy is None:
            out_vid.write(frame)
            continue
        
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else []
        
        if len(ids) == 0:
            out_vid.write(frame)
            continue
        
        for (x1, y1, x2, y2), track_id in zip(xyxy, ids):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Crop válido
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Classificação
            cls_result = cls_model.predict(
                source=crop, 
                imgsz=tracking_cfg["cls_imgsz"], 
                device="cpu", 
                verbose=False
            )
            
            probs = cls_result[0].probs
            color = cls_model.names[int(probs.top1)]
            conf = float(probs.top1conf)
            
            # Agregação
            track = tracks[int(track_id)]
            if track["start_frame"] is None:
                track["start_frame"] = frame_idx
            track["end_frame"] = frame_idx
            track["color_votes"][color] += conf
            track["confidences"].append(conf)
            
            # Anotação visual
            label = f"ID:{int(track_id)} {color} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out_vid.write(frame)
    
    out_vid.release()
    cap.release()
    
    # Exporta resultados finais
    records = []
    for track_id, track in tracks.items():
        if not track["confidences"]:
            continue
        
        # Cor final por voto ponderado
        final_color = max(track["color_votes"].items(), key=lambda x: x[1])[0]
        mean_conf = sum(track["confidences"]) / len(track["confidences"])
        
        records.append({
            "video_id": stem,
            "track_id": int(track_id),
            "frame_inicial": int(track["start_frame"]),
            "frame_final": int(track["end_frame"]),
            "cor": final_color,
            "confianca_media": round(mean_conf, 4)
        })
    
    # Salva JSON e CSV
    json_path = out_root / f"{stem}_tracks.json"
    csv_path = out_root / f"{stem}_tracks.csv"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "video_annotated": str(out_video_path),
        "total_tracks": len(records),
        "summary": {track["cor"]: len([r for r in records if r["cor"] == track["cor"]]) 
                   for track in records} if records else {}
    }
