# src/tracking/track.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict
import json
import csv
import time
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.utils.utils import load_config


# ======================================================================
# HSV / K-Means color classification
# ======================================================================

COLOR_LABELS = [
    "amarelo", "azul", "branco", "cinza_prata", "dourado",
    "laranja", "marrom", "preto", "verde", "vermelho"
]

def map_hsv_to_color_name(hsv_color: np.ndarray) -> str:
    """
    Mapeia HSV (OpenCV: H[0..179], S[0..255], V[0..255]) para um dos rótulos do dataset.
    Regras heurísticas: acromáticos por S/V, depois faixas de H para cores cromáticas.
    """
    if hsv_color is None or hsv_color.shape[0] != 3:
        return "cinza_prata"

    h, s, v = float(hsv_color[0]), float(hsv_color[1]), float(hsv_color[2])

    # Acromáticos
    if s < 40:
        if v < 60:
            return "preto"
        elif v > 195:
            return "branco"
        else:
            return "cinza_prata"

    # Cromáticos por H
    # Faixas ajustadas para o dataset
    if (h >= 0 and h <= 8) or (h >= 170 and h <= 179):
        return "vermelho"
    if 9 <= h <= 20:
        return "laranja"
    if 21 <= h <= 33:
        # amarelo/dourado: separar por valor (dourado tende a V um pouco menor)
        return "dourado" if v < 160 else "amarelo"
    if 34 <= h <= 85:
        return "verde"
    if 86 <= h <= 125:
        return "azul"
    # Tons quentes escuros: marrom em S alto e V médio-baixo
    if 10 <= h <= 25 and v < 140:
        return "marrom"

    # Fallback
    return "cinza_prata"


def get_dominant_color_kmeans(crop: np.ndarray, k: int = 4) -> Tuple[np.ndarray, float]:
    """
    Encontra cor dominante em HSV via K-Means, ponderando por S e V para evitar ruído acromático.
    Retorna (HSV_dominante, confiança).
    """
    if crop is None or crop.size == 0:
        return np.array([0, 0, 0], dtype=np.float32), 0.0

    # Redimensionar para acelerar
    h, w = crop.shape[:2]
    scale = 120.0 / max(h, w) if max(h, w) > 120 else 1.0
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Filtra pixels muito escuros/brilhantes com saturação baixa que confundem (opcional)
    mask = (hsv[:, :, 1] > 20) | (hsv[:, :, 2] > 40)
    pixels = hsv[mask].reshape(-1, 3).astype(np.float32)
    if pixels.size == 0:
        pixels = hsv.reshape(-1, 3).astype(np.float32)

    # K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

    # Pondera contagem por S e V (clusters mais saturados/visíveis ganham peso)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    s_weight = np.clip(centers[:, 1] / 255.0, 0.5, 1.0)
    v_weight = np.clip(centers[:, 2] / 255.0, 0.5, 1.0)
    weighted = counts * (0.6 * s_weight + 0.4 * v_weight)

    idx = int(np.argmax(weighted))
    dom = centers[idx]
    conf = float(counts[idx] / max(1, counts.sum()))
    return dom, conf


def classify_colors_hsv(crops: List[np.ndarray], infos: List[Tuple]) -> Dict[int, Tuple[str, float]]:
    """Classifica por HSV/K-Means."""
    new_labels: Dict[int, Tuple[str, float]] = {}
    for crop, info in zip(crops, infos):
        tid = info[4]
        hsv_color, confidence = get_dominant_color_kmeans(crop)
        color_name = map_hsv_to_color_name(hsv_color)
        new_labels[tid] = (color_name, confidence)
    return new_labels


def classify_colors_ai(
    cls_model: YOLO,
    crops: List[np.ndarray],
    infos: List[Tuple],
    imgsz: int,
    device: str,
    half: bool,
) -> Dict[int, Tuple[str, float]]:
    """Classifica por modelo YOLO-cls com batch e fallback para batch=1 se ONNX for estático."""
    new_labels: Dict[int, Tuple[str, float]] = {}
    cls_names = cls_model.names

    try:
        results = cls_model.predict(
            source=crops,
            imgsz=imgsz,
            device=device,
            half=half,
            verbose=False,
            batch=len(crops)
        )
    except Exception as e:
        if "invalid dimensions" in str(e) or "Got invalid dimensions" in str(e) or "Expected:" in str(e):
            results = []
            for c in crops:
                ri = cls_model.predict(source=c, imgsz=imgsz, device=device, half=half, verbose=False, batch=1)
                results.append(ri[0] if isinstance(ri, list) else ri)
        else:
            raise

    for r0, info in zip(results, infos):
        tid = info[4]
        probs = r0.probs
        top1_idx = int(probs.top1)
        # alguns backends já retornam float nativo
        top1_conf = float(probs.top1conf.cpu().item()) if hasattr(probs.top1conf, "cpu") else float(probs.top1conf)
        color_name = cls_names[top1_idx]
        new_labels[tid] = (color_name, top1_conf)
    return new_labels


# ======================================================================
# ONNX helpers
# ======================================================================

def onnx_static_imgsz(model: YOLO) -> Optional[int]:
    """Descobre H==W estático do ONNX (None se dinâmico)."""
    try:
        sess = getattr(model.model, "session", None)
        if sess is None:
            return None
        inputs = sess.get_inputs()
        if not inputs:
            return None
        shape = inputs[0].shape  # [N, C, H, W]
        if len(shape) != 4:
            return None
        H, W = shape[2], shape[3]
        if isinstance(H, int) and isinstance(W, int) and H == W and H > 0:
            return int(H)
        return None
    except Exception:
        return None


def load_yolo_prefer_onnx(
    weights_path: str,
    imgsz: int,
    task: str,
    prefer_onnx: bool = True,
    onnx_simplify: bool = True,
    onnx_dynamic: bool = False,
    verbose: bool = False,
) -> Tuple[YOLO, str, Optional[Path]]:
    """
    Carrega YOLO preferindo ONNX; exporta se necessário; retorna (modelo, backend, path_onnx).
    Usa task explícito e retorna o caminho exportado para permitir renomear se quiser.
    """
    wp = Path(weights_path)
    if not wp.exists():
        raise FileNotFoundError(f"Pesos não encontrados: {wp}")

    onnx_path = None

    if prefer_onnx:
        # 1) Tenta carregar .onnx com mesmo stem
        candidate = wp.with_suffix(".onnx")
        if candidate.exists():
            try:
                model = YOLO(str(candidate), task=task)
                return model, "onnx", candidate
            except Exception as e:
                if verbose:
                    print(f"[WARN] Falha ao carregar {candidate}: {e}")

        # 2) Exporta e carrega
        try:
            model_pt = YOLO(str(wp), task=task)
            exported = model_pt.export(
                format="onnx", imgsz=imgsz, simplify=onnx_simplify, dynamic=onnx_dynamic, verbose=verbose
            )  # docs: cria 'stem.onnx' e retorna path
            onnx_path = Path(exported) if exported else wp.with_suffix(".onnx")
            model = YOLO(str(onnx_path), task=task)
            return model, "onnx", onnx_path
        except Exception as e:
            if verbose:
                print(f"[WARN] Export ONNX falhou ({wp}): {e}. Fallback para .pt.")

    # 3) Fallback .pt
    model = YOLO(str(wp), task=task)
    return model, "pt", None


# ======================================================================
# Main
# ======================================================================

def process_video_tracking(
    video_path: str,
    det_weights: str,
    cls_weights: Optional[str],
    config_path: str = "src/config.json",
    out_dir: Optional[str] = None,
    device: str = "auto",
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict:

    cfg = load_config(config_path)
    tcfg = cfg.get("tracking", {})

    # Modo de cor
    color_mode = tcfg.get("color_classifier_mode", "hsv").lower()
    if color_mode not in ("hsv", "ai"):
        raise ValueError("color_classifier_mode deve ser 'hsv' ou 'ai'.")

    # ONNX flags
    prefer_onnx = bool(tcfg.get("prefer_onnx", True))
    onnx_simplify = bool(tcfg.get("onnx_simplify", True))
    onnx_dynamic_cls = bool(tcfg.get("onnx_dynamic_cls", True))
    verbose_export = bool(tcfg.get("onnx_verbose", False))

    # Tracker/params
    tracker_yaml = tcfg.get("tracker", "bytetrack.yaml")
    det_imgsz_cpu = int(tcfg.get("det_imgsz_cpu", 320))
    det_imgsz_gpu = int(tcfg.get("det_imgsz_gpu", 416))
    conf_thres = float(tcfg.get("conf_thres", 0.25))
    iou_thres = float(tcfg.get("iou_thres", 0.7))
    cls_imgsz = int(tcfg.get("cls_imgsz", 224))
    sample_every = int(tcfg.get("sample_every", 5))
    min_cls_conf = float(tcfg.get("min_cls_conf", 0.6))
    refresh_every = int(tcfg.get("refresh_every", 40))
    area_change_pct = float(tcfg.get("area_change_pct", 0.3))

    # Device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    use_half = (device != "cpu")

    # IO
    src = Path(video_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {src}")
    out_root = Path(out_dir).resolve() if out_dir else src.parent / "tracking_results"
    out_root.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {src}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = out_root / f"{stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_src, (w, h))

    # Load models
    det_imgsz_base = det_imgsz_gpu if use_half else det_imgsz_cpu
    det_model, det_backend, det_onnx_path = load_yolo_prefer_onnx(
        det_weights, det_imgsz_base, task="detect",
        prefer_onnx=prefer_onnx, onnx_simplify=onnx_simplify, onnx_dynamic=False, verbose=verbose_export
    )

    cls_model, cls_backend, cls_onnx_path = None, "hsv", None
    if color_mode == "ai":
        if not cls_weights:
            raise ValueError("cls_weights obrigatório quando color_classifier_mode='ai'.")
        cls_model, cls_backend, cls_onnx_path = load_yolo_prefer_onnx(
            cls_weights, cls_imgsz, task="classify",
            prefer_onnx=prefer_onnx, onnx_simplify=onnx_simplify, onnx_dynamic=onnx_dynamic_cls, verbose=verbose_export
        )

    # Harmoniza imgsz efetivo com ONNX estático (se houver)
    det_imgsz_eff = det_imgsz_base
    if det_backend == "onnx":
        s = onnx_static_imgsz(det_model)
        if s:
            det_imgsz_eff = s

    cls_imgsz_eff = cls_imgsz
    if color_mode == "ai" and cls_backend == "onnx":
        s = onnx_static_imgsz(cls_model)
        if s:
            cls_imgsz_eff = s

    # Warm-up
    try:
        _ = det_model.predict(np.zeros((det_imgsz_eff, det_imgsz_eff, 3), dtype=np.uint8),
                              device=device, imgsz=det_imgsz_eff, half=use_half, verbose=False)
        if color_mode == "ai":
            _ = cls_model.predict(np.zeros((cls_imgsz_eff, cls_imgsz_eff, 3), dtype=np.uint8),
                                  device=device, imgsz=cls_imgsz_eff, half=use_half, verbose=False)
    except Exception:
        pass

    # Tracking stream
    stream = det_model.track(
        source=str(src),
        tracker=tracker_yaml,
        persist=True,
        stream=True,
        device=device,
        imgsz=det_imgsz_eff,
        conf=conf_thres,
        iou=iou_thres,
        half=use_half,
        verbose=False,
        save=False,
        # Opcional pro desempenho:
        # classes=[0],
        # max_det=50,
        # vid_stride=1,
    )

    # States
    def id_color(i: int) -> Tuple[int, int, int]:
        rng = np.random.default_rng(abs(int(i)) + 12345)
        return tuple(int(c) for c in rng.integers(64, 255, size=3))

    tracks = defaultdict(lambda: {"start_frame": None, "end_frame": None, "color_votes": defaultdict(float), "confidences": []})
    last_label: Dict[int, Tuple[str, float]] = {}
    last_cls_frame: Dict[int, int] = {}
    last_area: Dict[int, int] = {}

    frame_idx = -1
    smooth_fps = 0.0
    t_prev = time.time()

    # Loop
    for result in stream:
        frame_idx += 1
        frame = result.orig_img
        boxes = getattr(result, "boxes", None)

        # FPS
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        inst_fps = 1.0 / dt
        smooth_fps = 0.9 * smooth_fps + 0.1 * inst_fps if smooth_fps > 0 else inst_fps
        t_prev = t_now

        need_cls_infos: List[Tuple[int, int, int, int, int]] = []
        on_sample = (frame_idx % sample_every == 0)

        if boxes is not None and boxes.xyxy is not None and boxes.id is not None:
            xyxy = boxes.xyxy
            ids = boxes.id
            xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
            ids = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)

            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
                if x2 <= x1 or y2 <= y1:
                    continue

                # area change
                area = (x2 - x1) * (y2 - y1)
                area_changed = False
                if tid in last_area:
                    a0 = max(1, last_area[tid])
                    area_changed = (abs(area - a0) / a0) >= area_change_pct
                last_area[tid] = area

                # decide classificar
                is_new = (tid not in last_label)
                low_conf = (not is_new) and (last_label[tid][1] < min_cls_conf)
                need_refresh = (tid in last_cls_frame) and ((frame_idx - last_cls_frame[tid]) >= refresh_every)

                if on_sample and (is_new or low_conf or need_refresh or area_changed):
                    need_cls_infos.append((x1, y1, x2, y2, tid))

            # classificação
            if need_cls_infos:
                crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2, _) in need_cls_infos]

                if color_mode == "ai":
                    new_labels = classify_colors_ai(cls_model, crops, need_cls_infos, imgsz=cls_imgsz_eff, device=device, half=use_half)
                else:
                    new_labels = classify_colors_hsv(crops, need_cls_infos)

                for tid, (color_name, conf) in new_labels.items():
                    tracks[tid]["color_votes"][color_name] += conf
                    tracks[tid]["confidences"].append(conf)
                    last_label[tid] = (color_name, conf)
                    last_cls_frame[tid] = frame_idx

            # desenha
            for (x1, y1, x2, y2), tid in zip(xyxy.astype(int), ids.astype(int)):
                c = id_color(int(tid))
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                if int(tid) in last_label:
                    color_name, conf = last_label[int(tid)]
                    label = f"ID:{int(tid)} {color_name} {conf:.2f}"
                    ty = max(18, y1 - 8)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # header
        progress_pct = 100.0 * (frame_idx + 1) / max(1, total_frames)
        header = f"FPS:{smooth_fps:.1f} | Prog:{progress_pct:.1f}% | det:{det_backend}:{det_imgsz_eff} cls:{'hsv' if color_mode=='hsv' else cls_backend}:{cls_imgsz_eff}"
        (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, 10), (10 + tw + 16, 10 + th + 16), (0, 0, 0), -1)
        cv2.putText(frame, header, (18, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()

    # Saídas agregadas
    records: List[Dict] = []
    for track_id, t in tracks.items():
        if not t["confidences"]:
            continue
        final_color = max(t["color_votes"].items(), key=lambda kv: kv[1])[0]
        mean_conf = float(sum(t["confidences"]) / len(t["confidences"]))
        records.append({
            "video_id": stem,
            "track_id": int(track_id),
            "frame_inicial": int(t["start_frame"]) if t["start_frame"] is not None else 0,
            "frame_final": int(t["end_frame"]) if t["end_frame"] is not None else 0,
            "cor": final_color,
            "confianca_media": round(mean_conf, 6),
        })

    json_path = out_root / f"{stem}_tracks.json"
    csv_path = out_root / f"{stem}_tracks.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["video_id", "track_id", "frame_inicial", "frame_final", "cor", "confianca_media"])
        writer_csv.writeheader()
        writer_csv.writerows(records)

    summary: Dict[str, int] = defaultdict(int)
    for r in records:
        summary[r["cor"]] += 1

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "video_annotated": str(out_video_path),
        "total_tracks": len(records),
        "summary": dict(summary),
    }
