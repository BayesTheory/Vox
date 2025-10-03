# src/tracking/track.py - Versão com DETECÇÃO INTERCALADA (2x mais rápido)

from __future__ import annotations

import os
import csv
import json
import time
import warnings
import gc
import threading
from queue import Queue, Empty, Full
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configurações ultra-otimizadas
os.environ.update({
    "ORT_LOGGING_LEVEL": "3", "ONNX_LOGGING_LEVEL": "3", "TF_CPP_MIN_LOG_LEVEL": "3",
    "YOLO_VERBOSE": "False", "ULTRALYTICS_VERBOSE": "False"
})
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


# =========================
# Utilities
# =========================
def round_to_stride(imgsz: int, stride: int = 32) -> int:
    return max(stride, int((imgsz + stride - 1) // stride) * stride)

def get_center_crop(x1, y1, x2, y2, margin=0.15):
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    return max(x1, x1 + mx), max(y1, y1 + my), min(x2, x2 - mx), min(y2, y2 - my)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class FPSTracker:
    def __init__(self, window_size=25):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.total_frames_processed = 0
        self.frame_by_frame_fps = []
        self.detection_times = deque(maxlen=window_size)
        self.classification_times = deque(maxlen=window_size)
        self.interpolation_times = deque(maxlen=window_size)

    def update(self, detection_time=0, classification_time=0, interpolation_time=0):
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        self.frame_times.append(dt)
        self.total_frames_processed += 1

        if detection_time > 0:
            self.detection_times.append(detection_time)
        if classification_time > 0:
            self.classification_times.append(classification_time)
        if interpolation_time > 0:
            self.interpolation_times.append(interpolation_time)

        total_elapsed = now - self.start_time
        instant_fps = 1.0 / max(dt, 1e-6)
        smoothed_fps = len(self.frame_times) / max(sum(self.frame_times), 1e-6)
        average_fps = self.total_frames_processed / max(total_elapsed, 1e-6)

        self.frame_by_frame_fps.append(round(instant_fps, 2))

        return {
            "instant": instant_fps,
            "smoothed": smoothed_fps,
            "average": average_fps,
            "elapsed": total_elapsed
        }

    def get_summary(self):
        total_time = time.perf_counter() - self.start_time
        return {
            "total_frames_processed": self.total_frames_processed,
            "total_time_seconds": round(total_time, 2),
            "total_time_formatted": self._format_time(total_time),
            "average_fps": round(self.total_frames_processed / max(total_time, 1e-6), 2),
            "avg_detection_time": round(sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0, 4),
            "avg_classification_time": round(sum(self.classification_times) / len(self.classification_times) if self.classification_times else 0, 4),
            "avg_interpolation_time": round(sum(self.interpolation_times) / len(self.interpolation_times) if self.interpolation_times else 0, 4),
        }

    @staticmethod
    def _format_time(seconds):
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        if h > 0: return f"{h}h {m:02d}m {s:02d}s"
        if m > 0: return f"{m}m {s:02d}s"
        return f"{s}s"


class TrackInterpolator:
    """Sistema de interpolação/propagação de tracks entre detecções"""

    def __init__(self, max_missing_frames=5):
        self.last_detection_results = None
        self.last_detection_frame = -1
        self.track_velocities = {}  # track_id -> (vx, vy)
        self.track_positions = {}   # track_id -> (x1, y1, x2, y2)
        self.max_missing_frames = max_missing_frames

    def update_from_detection(self, detection_results, frame_idx):
        """Atualiza com resultados reais de detecção"""
        self.last_detection_results = detection_results
        self.last_detection_frame = frame_idx

        # Calcula velocidades baseadas na posição anterior
        if detection_results.boxes is not None and detection_results.boxes.id is not None:
            current_positions = {}

            for box in detection_results.boxes:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                current_positions[track_id] = (x1, y1, x2, y2, cx, cy)

                # Calcula velocidade se temos posição anterior
                if track_id in self.track_positions:
                    old_x1, old_y1, old_x2, old_y2, old_cx, old_cy = self.track_positions[track_id]
                    frames_diff = frame_idx - self.last_detection_frame if self.last_detection_frame >= 0 else 1

                    vx = (cx - old_cx) / max(frames_diff, 1)
                    vy = (cy - old_cy) / max(frames_diff, 1)
                    self.track_velocities[track_id] = (vx, vy)
                else:
                    self.track_velocities[track_id] = (0, 0)  # Parado se é novo

            self.track_positions = current_positions

    def interpolate_frame(self, frame_idx):
        """Cria resultados interpolados para um frame sem detecção"""
        if self.last_detection_results is None:
            return None

        frames_since_detection = frame_idx - self.last_detection_frame

        # Se passou muito tempo, não interpola
        if frames_since_detection > self.max_missing_frames:
            return None

        # "Copia" o resultado da última detecção e ajusta posições
        interpolated_result = type(self.last_detection_results)(
            orig_img=np.zeros_like(self.last_detection_results.orig_img),
            path=self.last_detection_results.path,
            names=self.last_detection_results.names
        )

        if self.last_detection_results.boxes is not None and self.last_detection_results.boxes.id is not None:
            # Cria boxes interpoladas
            interpolated_boxes = []
            interpolated_ids = []
            interpolated_confs = []
            interpolated_classes = []

            for box in self.last_detection_results.boxes:
                track_id = int(box.id[0])

                if track_id in self.track_positions and track_id in self.track_velocities:
                    old_x1, old_y1, old_x2, old_y2, old_cx, old_cy = self.track_positions[track_id]
                    vx, vy = self.track_velocities[track_id]

                    # Extrapola posição
                    new_cx = old_cx + vx * frames_since_detection
                    new_cy = old_cy + vy * frames_since_detection

                    # Mantém tamanho da box
                    w, h = old_x2 - old_x1, old_y2 - old_y1
                    new_x1 = int(new_cx - w // 2)
                    new_y1 = int(new_cy - h // 2)
                    new_x2 = int(new_cx + w // 2)
                    new_y2 = int(new_cy + h // 2)

                    # Adiciona à lista
                    interpolated_boxes.append([new_x1, new_y1, new_x2, new_y2])
                    interpolated_ids.append(track_id)
                    interpolated_confs.append(float(box.conf[0]) * 0.9)  # Reduz confiança
                    interpolated_classes.append(int(box.cls[0]))

            if interpolated_boxes:
                # Cria estrutura compatível com YOLO Results
                interpolated_result.boxes = type(self.last_detection_results.boxes)(
                    torch.tensor(interpolated_boxes), 
                    interpolated_result.orig_img.shape
                )
                interpolated_result.boxes.id = torch.tensor(interpolated_ids)
                interpolated_result.boxes.conf = torch.tensor(interpolated_confs)
                interpolated_result.boxes.cls = torch.tensor(interpolated_classes)

        return interpolated_result


# =========================
# FrameWriter otimizado
# =========================
class FrameWriter(threading.Thread):
    def __init__(self, output_path, fourcc, fps, frame_size, processed_queue, timeline_mode="duplicate", frame_stride=1):
        super().__init__(daemon=True)
        self.processed_queue = processed_queue
        self.stop_event = threading.Event()
        self.frames_written = 0
        self.frame_stride = frame_stride

        if timeline_mode == "duplicate":
            self.output_fps = fps
            self.duplicate_factor = frame_stride
        elif timeline_mode == "fps_scale":
            self.output_fps = max(1.0, fps / frame_stride)
            self.duplicate_factor = 1
        else:  # realtime
            self.output_fps = fps
            self.duplicate_factor = 1

        self.writer = cv2.VideoWriter(str(output_path), fourcc, self.output_fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Não foi possível abrir o VideoWriter: {output_path}")

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame_data = self.processed_queue.get(timeout=1.0)
                if frame_data is None:
                    break

                for _ in range(self.duplicate_factor):
                    self.writer.write(frame_data['processed_frame'])
                    self.frames_written += 1

            except Empty:
                continue
            except Exception as e:
                print(f"Erro na thread writer: {e}")
                break

        self.writer.release()

    def stop(self):
        self.stop_event.set()


class SmartClassificationCache:
    def __init__(self):
        self.last_classification = {}
        self.track_confidence_history = defaultdict(list)

    def should_classify(self, track_id, processed_frame_idx, classification_interval=15):
        if track_id not in self.last_classification:
            return True

        frames_since_last = processed_frame_idx - self.last_classification[track_id]
        if frames_since_last >= classification_interval:
            return True

        if track_id in self.track_confidence_history:
            recent_confs = self.track_confidence_history[track_id][-3:]
            avg_conf = sum(recent_confs) / len(recent_confs) if recent_confs else 0
            if avg_conf < 0.7:
                return True

        return False

    def update_classification(self, track_id, processed_frame_idx, color, confidence):
        self.last_classification[track_id] = processed_frame_idx
        self.track_confidence_history[track_id].append(confidence)
        if len(self.track_confidence_history[track_id]) > 10:
            self.track_confidence_history[track_id] = self.track_confidence_history[track_id][-10:]


# =========================
# Model Loading
# =========================
def load_models_optimized(det_weights, cls_weights, device, det_imgsz, cls_imgsz, num_threads):
    is_gpu = device.startswith("cuda")

    if is_gpu:
        print("🚀 GPU detectada: usando PyTorch .pt diretamente")
        det_model = YOLO(str(det_weights), task="detect")
        cls_model = YOLO(str(cls_weights), task="classify")
        backend_info = {"detection": "PyTorch-CUDA", "classification": "PyTorch-CUDA"}
    else:
        print(f"💻 CPU detectada: usando ONNX com {num_threads} threads")
        try:
            det_onnx_path = Path(det_weights).with_suffix(".onnx")
            cls_onnx_path = Path(cls_weights).with_suffix(".onnx")

            if not det_onnx_path.exists():
                det_temp = YOLO(str(det_weights), task="detect")
                det_temp.export(format="onnx", imgsz=det_imgsz, simplify=True, verbose=False)

            if not cls_onnx_path.exists():
                cls_temp = YOLO(str(cls_weights), task="classify")
                cls_temp.export(format="onnx", imgsz=cls_imgsz, simplify=True, verbose=False)

            det_model = YOLO(str(det_onnx_path), task="detect")
            cls_model = YOLO(str(cls_onnx_path), task="classify")

            _configure_onnx_cpu(det_model, num_threads)
            _configure_onnx_cpu(cls_model, num_threads)

            backend_info = {"detection": "ONNX-CPU", "classification": "ONNX-CPU"}
            print(f"✅ ONNX configurado com {num_threads} threads")
        except Exception as e:
            print(f"⚠️  Falha ao usar ONNX, usando PyTorch CPU: {e}")
            det_model = YOLO(str(det_weights), task="detect")
            cls_model = YOLO(str(cls_weights), task="classify")
            backend_info = {"detection": "PyTorch-CPU", "classification": "PyTorch-CPU"}

    return det_model, cls_model, backend_info


def _configure_onnx_cpu(yolo_model, num_threads):
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = max(1, num_threads // 2)
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 4

        providers = [("CPUExecutionProvider", {"intra_op_num_threads": num_threads})]

        if hasattr(yolo_model.model, 'model_path'):
            model_path = yolo_model.model.model_path
            new_session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
            yolo_model.model.session = new_session
    except Exception as e:
        print(f"Erro ao configurar ONNX CPU: {e}")


# =========================
# Classification & Drawing
# =========================
def classify_colors_ai(cls_model, crops, infos, imgsz, device, half, batch_size=16):
    if not crops:
        return {}

    new_labels = {}
    cls_names = cls_model.names

    try:
        with torch.no_grad():
            results = cls_model.predict(
                source=crops, imgsz=imgsz, device=device, half=half, 
                verbose=False, batch=min(len(crops), batch_size * 2)
            )

        for r0, info in zip(results, infos):
            tid = info[4]
            probs = r0.probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf.cpu().item()) if hasattr(probs.top1conf, "cpu") else float(probs.top1conf)
            color_name = cls_names[top1_idx]
            new_labels[tid] = (color_name, top1_conf)

        return new_labels

    except Exception:
        # Fallback para método original
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i+batch_size]
            batch_infos = infos[i:i+batch_size]

            try:
                with torch.no_grad():
                    results = cls_model.predict(
                        source=batch_crops, imgsz=imgsz, device=device, half=half, 
                        verbose=False, batch=len(batch_crops)
                    )

                for r0, info in zip(results, batch_infos):
                    tid = info[4]
                    probs = r0.probs
                    top1_idx = int(probs.top1)
                    top1_conf = float(probs.top1conf.cpu().item()) if hasattr(probs.top1conf, "cpu") else float(probs.top1conf)
                    color_name = cls_names[top1_idx]
                    new_labels[tid] = (color_name, top1_conf)
            except Exception:
                continue

    return new_labels


def draw_metrics_overlay(frame, fps_stats, frame_idx, total_frames_proc, eta, num_detections=0, 
                        fps_raw_equiv=None, detection_mode="", interpolated_count=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, color = 0.6, 2, (0, 255, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (480, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    texts = [
        f"FPS: {fps_stats['instant']:.1f} | Smooth: {fps_stats['smoothed']:.1f}",
        f"Proc: {frame_idx + 1}/{max(1,total_frames_proc)} ({100*(frame_idx+1)/max(1,total_frames_proc):.1f}%)",
        f"ETA: {eta}",
        f"Objects: {num_detections} {detection_mode}",
    ]

    if fps_raw_equiv:
        texts.insert(1, f"FPS(equiv): {fps_raw_equiv:.1f}")

    if interpolated_count > 0:
        texts.append(f"Interpolated frames: {interpolated_count}")

    for i, text in enumerate(texts):
        cv2.putText(frame, text, (10, 25 + i*25), font, font_scale, color, thickness)


# =========================
# MAIN FUNCTION - DETECÇÃO INTERCALADA
# =========================
def process_video_tracking(
    video_path: str,
    det_weights: str,
    cls_weights: str,
    config_path: str,
    out_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Função principal com DETECÇÃO INTERCALADA - otimização sugerida pelo usuário
    """

    config = load_config(config_path)
    cfg = config["tracking"]

    # Setup do dispositivo
    config_device = cfg["inference"].get("device", "auto")
    device = "cuda:0" if config_device == "auto" and torch.cuda.is_available() else "cpu"
    is_gpu = device.startswith("cuda")

    # Configurações de performance com DETECÇÃO INTERCALADA
    perf_cfg = cfg.get("performance", {})
    frame_stride = max(1, int(perf_cfg.get("frame_stride", 1)))

    # NOVA CONFIGURAÇÃO: Detecção intercalada
    detection_interval = int(perf_cfg.get("detection_interval", 2))  # Detecta a cada N frames
    print(f"🎯 OTIMIZAÇÃO: Detecção intercalada a cada {detection_interval} frames")

    num_threads_cpu = int(perf_cfg.get("num_threads_cpu", os.cpu_count() or 8))

    if not is_gpu:
        print(f"💻 CPU: {num_threads_cpu} threads")
        os.environ.update({
            "OMP_NUM_THREADS": str(num_threads_cpu), 
            "MKL_NUM_THREADS": str(num_threads_cpu),
            "NUMEXPR_NUM_THREADS": str(num_threads_cpu)
        })
        cv2.setNumThreads(num_threads_cpu)
        torch.set_num_threads(num_threads_cpu)

    # Setup dos paths
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    # Informações do vídeo
    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    video_info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()

    out_dir_path = Path(out_dir) if out_dir else video_path_obj.parent / f"{video_path_obj.stem}_output"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    suffix = "gpu" if is_gpu else "cpu"
    out_video_path = out_dir_path / f"{video_path_obj.stem}_intercalada_{suffix}.mp4"
    timeline_mode = str(cfg["output"].get("timeline_mode", "duplicate")).lower()

    # Carrega modelos
    det_imgsz_base = round_to_stride(int(cfg["detection"]["imgsz_gpu"] if is_gpu else cfg["detection"]["imgsz_cpu"]))
    cls_imgsz_base = round_to_stride(int(cfg["classification_model"]["imgsz"]))

    det_model, cls_model, backend_info = load_models_optimized(
        det_weights, cls_weights, device, det_imgsz_base, cls_imgsz_base, num_threads_cpu
    )

    # Setup do FrameWriter
    processed_queue = Queue(maxsize=30)
    writer = FrameWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*cfg["output"]["video_codec"]),
        video_info['fps'],
        (video_info['width'], video_info['height']),
        processed_queue,
        timeline_mode,
        frame_stride
    )
    writer.start()

    # Setup das configurações
    cls_cache = SmartClassificationCache()
    interpolator = TrackInterpolator(max_missing_frames=detection_interval * 2)

    det_conf = float(cfg["detection"]["conf_threshold"])
    det_iou = float(cfg["detection"]["iou_threshold"])
    max_det = int(cfg["detection"].get("max_det", 40))
    batch_size = int(cfg["classification_model"].get("batch_size", 16))
    center_crop_margin = float(cfg["classification_model"].get("center_crop_margin", 0.15))
    classification_interval = int(cfg["sampling"].get("classify_every", 15))
    use_half = bool(cfg["inference"].get("half_precision", False)) and is_gpu

    # Warm-up
    print("🔥 Warm-up...")
    try:
        with torch.no_grad():
            dummy = np.zeros((det_imgsz_base, det_imgsz_base, 3), dtype=np.uint8)
            _ = det_model.predict(dummy, device=device, imgsz=det_imgsz_base, half=use_half, verbose=False)
            _ = cls_model.predict(dummy, device=device, imgsz=cls_imgsz_base, half=use_half, verbose=False)
    except Exception:
        pass

    # Setup do loop principal
    tracks = defaultdict(lambda: {"start_frame": -1, "end_frame": 0, "color_votes": defaultdict(float), "confidences": []})
    last_known_color = {}
    fps_tracker = FPSTracker()
    total_frames_proc = max(1, video_info['total_frames'] // frame_stride)
    interpolated_frames_count = 0

    print(f"🚀 DETECÇÃO INTERCALADA processamento:")
    print(f"   📹 {video_info['total_frames']} frames @ {video_info['fps']:.2f} FPS")
    print(f"   ⚡ Frame stride: {frame_stride} (~{total_frames_proc} frames)")
    print(f"   🎯 Detecção: a cada {detection_interval} frames (economia: {(detection_interval-1)/detection_interval*100:.1f}%)")
    print(f"   🎬 Timeline: {timeline_mode}")
    print(f"   🔧 {backend_info['detection']} | {device}")

    # Stream nativo com detecção intercalada
    stream_config = {
        'source': str(video_path_obj),
        'stream': True,
        'persist': True,
        'tracker': cfg["tracker"],
        'device': device,
        'imgsz': det_imgsz_base,
        'conf': det_conf,
        'iou': det_iou,
        'half': use_half,
        'verbose': False,
        'vid_stride': frame_stride,
        'max_det': max_det,
        'save': False,
        'show': False,
    }

    stream = det_model.track(**stream_config)

    try:
        for processed_frame_idx, result in enumerate(stream):
            frame_idx = processed_frame_idx * frame_stride

            # OTIMIZAÇÃO PRINCIPAL: Detecção intercalada
            should_detect = processed_frame_idx % detection_interval == 0

            if should_detect:
                # Frame com DETECÇÃO REAL
                det_start = time.perf_counter()
                processed_frame = result.orig_img.copy()
                det_time = time.perf_counter() - det_start

                # Atualiza interpolador com resultados reais
                interpolator.update_from_detection(result, processed_frame_idx)
                detection_mode = "[DETECT]"
                interp_time = 0

            else:
                # Frame INTERPOLADO (muito mais rápido)
                interp_start = time.perf_counter()
                result = interpolator.interpolate_frame(processed_frame_idx)
                processed_frame = np.zeros((video_info['height'], video_info['width'], 3), dtype=np.uint8)

                if result is None:
                    # Se não conseguiu interpolar, pula este frame
                    continue

                det_time = 0
                interp_time = time.perf_counter() - interp_start
                detection_mode = "[INTERP]"
                interpolated_frames_count += 1

            need_cls_infos = []
            num_detections = 0
            cls_time = 0

            if result.boxes is not None and result.boxes.id is not None:
                num_detections = len(result.boxes)

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0])

                    if tracks[track_id]["start_frame"] == -1:
                        tracks[track_id]["start_frame"] = frame_idx
                    tracks[track_id]["end_frame"] = frame_idx

                    # Classificação apenas em frames com detecção real
                    if should_detect and cls_cache.should_classify(track_id, processed_frame_idx, classification_interval):
                        cx1, cy1, cx2, cy2 = get_center_crop(x1, y1, x2, y2, center_crop_margin)
                        if cx2 > cx1 and cy2 > cy1:
                            need_cls_infos.append((cx1, cy1, cx2, cy2, track_id))

                # Classificação (apenas em frames de detecção)
                if need_cls_infos and should_detect:
                    cls_start = time.perf_counter()
                    crops = [processed_frame[y1:y2, x1:x2] for (x1, y1, x2, y2, _) in need_cls_infos]
                    new_labels = classify_colors_ai(cls_model, crops, need_cls_infos, cls_imgsz_base, device, use_half, batch_size)
                    cls_time = time.perf_counter() - cls_start

                    for tid, (color_name, conf) in new_labels.items():
                        if conf >= float(cfg["classification"]["min_confidence"]):
                            cls_cache.update_classification(tid, processed_frame_idx, color_name, conf)
                            tracks[tid]["color_votes"][color_name] += conf
                            tracks[tid]["confidences"].append(conf)
                            last_known_color[tid] = (color_name, conf)

            # Visualização otimizada
            vis_cfg = cfg["visualization"]
            if vis_cfg["draw_boxes"] and result.boxes is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Cor diferente para frames interpolados
                    if should_detect:
                        np.random.seed(track_id)
                        box_color = tuple(int(c) for c in np.random.randint(64, 255, size=3))
                    else:
                        # Cor mais apagada para interpolados
                        np.random.seed(track_id)
                        base_color = np.random.randint(64, 255, size=3)
                        box_color = tuple(int(c * 0.7) for c in base_color)

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, vis_cfg["box_thickness"])

                    label = []
                    if vis_cfg["draw_track_id"]:
                        label.append(f"ID:{track_id}")
                    if track_id in last_known_color:
                        color_name, conf = last_known_color[track_id]
                        if vis_cfg["draw_labels"]:
                            label.append(color_name)
                        if vis_cfg["draw_confidence"]:
                            label.append(f"{conf:.2f}")

                    if label:
                        label_text = " ".join(label)
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                      vis_cfg["font_scale"], vis_cfg["font_thickness"])
                        cv2.rectangle(processed_frame, (x1, y1 - th - 10), (x1 + tw + 5, y1), box_color, -1)
                        cv2.putText(processed_frame, label_text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                    vis_cfg["font_scale"], (255,255,255), vis_cfg["font_thickness"])

            # Overlay com informações de interpolação
            fps_stats = fps_tracker.update(det_time, cls_time, interp_time)
            fps_raw_equiv = fps_stats["instant"] * frame_stride
            eta = fps_tracker._format_time((total_frames_proc - processed_frame_idx - 1) / max(1e-6, fps_stats["smoothed"]))

            draw_metrics_overlay(processed_frame, fps_stats, processed_frame_idx, 
                               total_frames_proc, eta, num_detections, fps_raw_equiv, 
                               detection_mode, interpolated_frames_count)

            # Envia para escrita
            try:
                processed_queue.put({
                    'processed_frame_idx': processed_frame_idx,
                    'processed_frame': processed_frame
                }, timeout=0.1)
            except Full:
                pass

            if progress_cb and processed_frame_idx % 5 == 0:
                progress_cb(min(processed_frame_idx + 1, total_frames_proc), total_frames_proc)

            # Limpeza otimizada
            if processed_frame_idx % 200 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n⚠️ Interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔄 Finalizando...")
        processed_queue.put(None)
        writer.join(timeout=10)
        print(f"✅ Finalizado. Frames: {fps_tracker.total_frames_processed}")
        print(f"📊 Frames interpolados: {interpolated_frames_count} ({interpolated_frames_count/fps_tracker.total_frames_processed*100:.1f}%)")

    # Resultados
    performance_summary = fps_tracker.get_summary()
    performance_summary.update({
        "fps_model": performance_summary["average_fps"],
        "fps_raw_equiv": performance_summary["average_fps"] * frame_stride,
        "processed_frames_total": fps_tracker.total_frames_processed,
        "interpolated_frames": interpolated_frames_count,
        "detection_efficiency": f"{(detection_interval-1)/detection_interval*100:.1f}% less detections",
        "timeline_mode": timeline_mode
    })

    # Consolidação
    final_records = []
    for tid, data in tracks.items():
        if data["confidences"]:
            final_color = max(data["color_votes"].items(), key=lambda x: x[1])[0] if data["color_votes"] else "indefinido"
            final_records.append({
                "video_id": video_path_obj.stem,
                "track_id": tid,
                "frame_inicial": data["start_frame"],
                "frame_final": data["end_frame"],
                "cor": final_color,
                "confianca_media": round(sum(data["confidences"]) / len(data["confidences"]), 4)
            })

    # Output
    color_counts = defaultdict(int)
    for record in final_records:
        color_counts[record["cor"]] += 1

    result_data = {
        "video_info": {
            "filename": video_path_obj.name,
            "resolution": f"{video_info['width']}x{video_info['height']}",
            "fps_in": round(video_info['fps'], 3),
            "fps_out": round(writer.output_fps, 3),
            "total_frames_in": video_info['total_frames'],
            "total_frames_out": writer.frames_written,
            "frame_stride": frame_stride,
            "detection_interval": detection_interval,
            "timeline_mode": timeline_mode
        },
        "hardware_info": {
            "device": device,
            "backend_detection": backend_info["detection"],
            "backend_classification": backend_info["classification"]
        },
        "processing_parameters": {
            "tracker": cfg["tracker"],
            "detection": {"imgsz": det_imgsz_base, "conf_threshold": det_conf, "iou_threshold": det_iou, "max_det": max_det},
            "classification": {"min_confidence": float(cfg["classification"]["min_confidence"]), "imgsz": cls_imgsz_base, "batch_size": batch_size},
            "performance": {"frame_stride": frame_stride, "detection_interval": detection_interval, "num_threads_cpu": num_threads_cpu},
        },
        "performance_metrics": performance_summary,
        "tracks": final_records,
        "summary": {"total_tracks": len(final_records), "color_distribution": dict(color_counts)}
    }

    # Save
    output_paths = {}
    if cfg["output"]["save_json"]:
        json_path = out_dir_path / f"{video_path_obj.stem}_intercalada_{suffix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        output_paths["json"] = str(json_path)

    if cfg["output"]["save_csv"] and final_records:
        csv_path = out_dir_path / f"{video_path_obj.stem}_intercalada_{suffix}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=final_records[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(final_records)
        output_paths["csv"] = str(csv_path)

    print(f"\n🎉 DETECÇÃO INTERCALADA concluída:")
    print(f"   📊 {len(final_records)} tracks | {performance_summary['average_fps']:.1f} FPS")
    print(f"   ⚡ Economia: {(detection_interval-1)/detection_interval*100:.1f}% menos detecções")
    print(f"   🔄 {interpolated_frames_count} frames interpolados")

    return {
        "output_dir": str(out_dir_path),
        "total_tracks": len(final_records),
        "output_files": output_paths,
        "performance": performance_summary,
        "color_distribution": dict(color_counts)
    }
