# src/tracking/track.py - VERSÃO CONFIGURÁVEL SEM HARDCODE

from __future__ import annotations

import os
import csv
import json
import time
import warnings
import threading
import psutil
from queue import Queue, Empty, Full
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# ✅ CONFIGURAÇÕES BÁSICAS
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

    def update(self, detection_time=0, classification_time=0):
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        self.frame_times.append(dt)
        self.total_frames_processed += 1

        if detection_time > 0:
            self.detection_times.append(detection_time)
        if classification_time > 0:
            self.classification_times.append(classification_time)

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
        }

    @staticmethod
    def _format_time(seconds):
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        if h > 0: return f"{h}h {m:02d}m {s:02d}s"
        if m > 0: return f"{m}m {s:02d}s"
        return f"{s}s"

# =========================
# FrameWriter
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

# =========================
# Model Loading - CONFIGURÁVEL
# =========================

def get_hardware_info():
    """Detecta informações de hardware"""
    info = {
        "has_cuda": torch.cuda.is_available(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_count": psutil.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / (1024**3)
    }
    
    if info["has_cuda"]:
        info["gpu_name"] = torch.cuda.get_device_name()
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info

def apply_performance_config(config: Dict[str, Any], hardware_info: Dict[str, Any]):
    """✅ APLICA CONFIGURAÇÕES BASEADAS NO HARDWARE E CONFIG"""
    
    perf_cfg = config["tracking"]["performance"]
    inference_cfg = config["tracking"]["inference"]
    
    # ✅ DETECÇÃO AUTOMÁTICA DE DEVICE
    config_device = inference_cfg.get("device", "auto")
    if config_device == "auto":
        device = "cuda:0" if hardware_info["has_cuda"] else "cpu"
    else:
        device = config_device
    
    is_gpu = device.startswith("cuda")
    
    # ✅ APLICAR CONFIGURAÇÕES AUTOMÁTICAS SE NÃO ESPECIFICADAS
    
    # Threading
    if perf_cfg.get("num_threads_cpu", 0) == 0 and not is_gpu:
        optimal_threads = min(hardware_info["cpu_cores"], 8)
        perf_cfg["num_threads_cpu"] = optimal_threads
        print(f"🧵 Auto-config threads: {optimal_threads}")
    
    # Batch size
    cls_cfg = config["tracking"]["classification_model"]
    if cls_cfg.get("batch_size", 0) == 0:
        if is_gpu:
            gpu_memory = hardware_info.get("gpu_memory_gb", 4)
            if gpu_memory >= 12:
                cls_cfg["batch_size"] = 16
            elif gpu_memory >= 8:
                cls_cfg["batch_size"] = 8
            else:
                cls_cfg["batch_size"] = 4
        else:
            cls_cfg["batch_size"] = 2  # CPU conservador
        print(f"📦 Auto-config batch size: {cls_cfg['batch_size']}")
    
    # Frame stride
    if perf_cfg.get("frame_stride", 0) == 0:
        perf_cfg["frame_stride"] = 2 if is_gpu else 3
        print(f"⚡ Auto-config frame stride: {perf_cfg['frame_stride']}")
    
    # Image sizes
    det_cfg = config["tracking"]["detection"]
    if det_cfg.get("imgsz_cpu", 0) == 0:
        det_cfg["imgsz_cpu"] = 640 if is_gpu else 416
        print(f"🖼️ Auto-config detection imgsz: {det_cfg['imgsz_cpu']}")
    
    if cls_cfg.get("imgsz", 0) == 0:
        cls_cfg["imgsz"] = 224 if is_gpu else 160
        print(f"🎨 Auto-config classification imgsz: {cls_cfg['imgsz']}")
    
    # Sampling
    sampling_cfg = config["tracking"]["sampling"]
    if sampling_cfg.get("classify_every", 0) == 0:
        sampling_cfg["classify_every"] = 3 if is_gpu else 5
        print(f"🔄 Auto-config classify every: {sampling_cfg['classify_every']}")
    
    return device, is_gpu

def load_models(det_weights: str, cls_weights: str, config: Dict[str, Any], device: str) -> Tuple[YOLO, YOLO, Dict[str, str]]:
    """✅ CARREGAMENTO DE MODELOS CONFIGURÁVEL"""
    
    perf_cfg = config["tracking"]["performance"]
    
    # Verificar se deve forçar PyTorch
    force_pytorch = perf_cfg.get("force_pytorch", False)
    use_onnx = perf_cfg.get("use_onnx", True) and not force_pytorch
    
    is_gpu = device.startswith("cuda")
    
    print(f"🚀 Carregando modelos:")
    print(f"   Device: {device}")
    print(f"   Force PyTorch: {force_pytorch}")
    print(f"   Use ONNX: {use_onnx}")
    
    backend_info = {}
    
    # ✅ CARREGAMENTO CONFIGURÁVEL DO DETECTOR
    if is_gpu or not use_onnx:
        print("📦 Detector: PyTorch (.pt)")
        det_model = YOLO(str(det_weights), task="detect")
        backend_info["detection"] = f"PyTorch-{device.upper()}"
    else:
        print("📦 Detector: Tentando ONNX...")
        try:
            det_onnx_path = Path(det_weights).with_suffix(".onnx")
            
            if not det_onnx_path.exists():
                print("   Exportando para ONNX...")
                det_temp = YOLO(str(det_weights), task="detect")
                det_imgsz = config["tracking"]["detection"]["imgsz_cpu"]
                det_temp.export(format="onnx", imgsz=det_imgsz, simplify=True, verbose=False)
            
            det_model = YOLO(str(det_onnx_path), task="detect")
            
            # Configurar ONNX
            num_threads = perf_cfg.get("num_threads_cpu", 4)
            _configure_onnx_cpu(det_model, num_threads)
            
            backend_info["detection"] = f"ONNX-CPU-{num_threads}T"
            print(f"   ✅ ONNX detector carregado ({num_threads} threads)")
            
        except Exception as e:
            print(f"   ⚠️ ONNX falhou: {e}")
            print("   📦 Fallback para PyTorch")
            det_model = YOLO(str(det_weights), task="detect")
            backend_info["detection"] = "PyTorch-CPU-Fallback"
    
    # ✅ CARREGAMENTO CONFIGURÁVEL DO CLASSIFICADOR
    force_cls_pytorch = perf_cfg.get("force_classification_pytorch", True)  # Padrão True para estabilidade
    
    if is_gpu or not use_onnx or force_cls_pytorch:
        print("🎨 Classificador: PyTorch (.pt) - RECOMENDADO")
        cls_model = YOLO(str(cls_weights), task="classify")
        backend_info["classification"] = f"PyTorch-{device.upper()}-Stable"
    else:
        print("🎨 Classificador: Tentando ONNX...")
        try:
            cls_onnx_path = Path(cls_weights).with_suffix(".onnx")
            
            if not cls_onnx_path.exists():
                print("   Exportando classificador para ONNX...")
                cls_temp = YOLO(str(cls_weights), task="classify")
                cls_imgsz = config["tracking"]["classification_model"]["imgsz"]
                cls_temp.export(format="onnx", imgsz=cls_imgsz, simplify=True, verbose=False)
            
            cls_model = YOLO(str(cls_onnx_path), task="classify")
            
            # Configurar ONNX
            num_threads = perf_cfg.get("num_threads_cpu", 4)
            _configure_onnx_cpu(cls_model, num_threads)
            
            backend_info["classification"] = f"ONNX-CPU-{num_threads}T-EXPERIMENTAL"
            print(f"   ⚠️ ONNX classificador carregado (experimental)")
            
        except Exception as e:
            print(f"   ❌ ONNX classificador falhou: {e}")
            print("   🔧 Forçando PyTorch para classificador")
            cls_model = YOLO(str(cls_weights), task="classify")
            backend_info["classification"] = "PyTorch-CPU-ForceStable"
    
    return det_model, cls_model, backend_info

def _configure_onnx_cpu(yolo_model, num_threads):
    """Configura ONNX para CPU"""
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
        print(f"Erro configurando ONNX: {e}")

# =========================
# Classification - CONFIGURÁVEL
# =========================

def classify_colors_smart(cls_model, crops, infos, config: Dict[str, Any], device: str):
    """✅ CLASSIFICAÇÃO INTELIGENTE BASEADA NO CONFIG"""
    if not crops:
        return {}

    cls_cfg = config["tracking"]["classification_model"]
    cls_min_conf = config["tracking"]["classification"]["min_confidence"]
    
    # Parâmetros do config
    imgsz = cls_cfg["imgsz"]
    batch_size = cls_cfg["batch_size"]
    use_half = config["tracking"]["inference"].get("half_precision", False) and device.startswith("cuda")
    individual_mode = config["tracking"]["performance"].get("force_individual_classification", False)
    
    new_labels = {}
    cls_names = cls_model.names
    
    # ✅ CONVERSÃO CONFIGURÁVEL PARA PIL
    valid_crops = []
    valid_infos = []
    
    min_crop_size = cls_cfg.get("min_crop_size", 10)
    
    for crop, info in zip(crops, infos):
        try:
            # Validar com tamanho configurável
            if crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
                continue
            
            # Conversão BGR -> RGB -> PIL
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = crop
            
            pil_crop = Image.fromarray(crop_rgb)
            valid_crops.append(pil_crop)
            valid_infos.append(info)
            
        except Exception as e:
            continue
    
    if not valid_crops:
        return {}
    
    print(f"🎨 Classificando {len(valid_crops)} crops (batch={batch_size}, individual={individual_mode})")
    
    # ✅ MODO CONFIGURÁVEL: BATCH OU INDIVIDUAL
    if individual_mode or batch_size == 1:
        # Modo individual (mais estável)
        for crop, info in zip(valid_crops, valid_infos):
            try:
                with torch.no_grad():
                    result = cls_model.predict(
                        source=crop,
                        imgsz=imgsz,
                        device=device,
                        half=use_half,
                        verbose=False,
                        batch=False
                    )
                
                tid = info[4] if len(info) > 4 else info[-1]
                
                if result.probs is not None:
                    probs = result.probs
                    top1_idx = int(probs.top1)
                    
                    if hasattr(probs.top1conf, "cpu"):
                        top1_conf = float(probs.top1conf.cpu().item())
                    else:
                        top1_conf = float(probs.top1conf)
                    
                    if top1_conf >= cls_min_conf:
                        color_name = cls_names[top1_idx]
                        new_labels[tid] = (color_name, top1_conf)
                        
            except Exception as e:
                continue
    else:
        # Modo batch (mais rápido se funcionar)
        for i in range(0, len(valid_crops), batch_size):
            batch_crops = valid_crops[i:i+batch_size]
            batch_infos = valid_infos[i:i+batch_size]
            
            try:
                with torch.no_grad():
                    results = cls_model.predict(
                        source=batch_crops,
                        imgsz=imgsz,
                        device=device,
                        half=use_half,
                        verbose=False,
                        batch=len(batch_crops)
                    )
                
                for result, info in zip(results, batch_infos):
                    tid = info[4] if len(info) > 4 else info[-1]
                    
                    if result.probs is not None:
                        probs = result.probs
                        top1_idx = int(probs.top1)
                        
                        if hasattr(probs.top1conf, "cpu"):
                            top1_conf = float(probs.top1conf.cpu().item())
                        else:
                            top1_conf = float(probs.top1conf)
                        
                        if top1_conf >= cls_min_conf:
                            color_name = cls_names[top1_idx]
                            new_labels[tid] = (color_name, top1_conf)
                            
            except Exception as e:
                error_msg = str(e).lower()
                if "dimension" in error_msg or "batch" in error_msg:
                    print(f"⚠️ Erro de batch detectado, alternando para modo individual")
                    # Reprocessar este batch individualmente
                    for crop, info in zip(batch_crops, batch_infos):
                        try:
                            with torch.no_grad():
                                result = cls_model.predict(
                                    source=crop,
                                    imgsz=imgsz,
                                    device=device,
                                    half=use_half,
                                    verbose=False,
                                    batch=False
                                )
                            
                            tid = info[4] if len(info) > 4 else info[-1]
                            
                            if result.probs is not None:
                                probs = result.probs
                                top1_idx = int(probs.top1)
                                
                                if hasattr(probs.top1conf, "cpu"):
                                    top1_conf = float(probs.top1conf.cpu().item())
                                else:
                                    top1_conf = float(probs.top1conf)
                                
                                if top1_conf >= cls_min_conf:
                                    color_name = cls_names[top1_idx]
                                    new_labels[tid] = (color_name, top1_conf)
                                    
                        except Exception:
                            continue
                else:
                    print(f"⚠️ Erro no batch {i//batch_size}: {e}")
                    continue
    
    return new_labels

def draw_metrics_overlay(frame, fps_stats, frame_idx, total_frames_proc, eta, num_detections=0, fps_raw_equiv=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, color = 0.6, 2, (0, 255, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (480, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    texts = [
        f"FPS: {fps_stats['instant']:.1f} | Smooth: {fps_stats['smoothed']:.1f}",
        f"Proc: {frame_idx + 1}/{max(1,total_frames_proc)} ({100*(frame_idx+1)/max(1,total_frames_proc):.1f}%)",
        f"ETA: {eta}",
        f"Objects: {num_detections}",
    ]

    if fps_raw_equiv:
        texts.insert(1, f"Raw FPS: {fps_raw_equiv:.1f}")

    for i, text in enumerate(texts):
        cv2.putText(frame, text, (10, 25 + i*20), font, font_scale, color, thickness)

# =========================
# MAIN FUNCTION - TOTALMENTE CONFIGURÁVEL
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
    ✅ FUNÇÃO PRINCIPAL TOTALMENTE CONFIGURÁVEL PELO CONFIG.JSON
    """
    
    # ✅ CARREGAR CONFIG
    config = load_config(config_path)
    cfg = config["tracking"]
    
    # ✅ DETECTAR HARDWARE E APLICAR CONFIGURAÇÕES
    hardware_info = get_hardware_info()
    device, is_gpu = apply_performance_config(config, hardware_info)
    
    print(f"🎯 TRACKING CONFIGURÁVEL iniciando...")
    print(f"   🔧 Device: {device}")
    print(f"   💻 Hardware: {'GPU' if is_gpu else 'CPU'}")
    print(f"   📊 Config: {config_path}")
    
    # ✅ CONFIGURAR THREADING SE CPU
    if not is_gpu:
        num_threads = cfg["performance"]["num_threads_cpu"]
        os.environ.update({
            "OMP_NUM_THREADS": str(num_threads),
            "MKL_NUM_THREADS": str(num_threads),
            "NUMEXPR_NUM_THREADS": str(num_threads)
        })
        cv2.setNumThreads(num_threads)
        torch.set_num_threads(num_threads)
        print(f"   🧵 CPU Threads: {num_threads}")
    
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
    out_video_path = out_dir_path / f"{video_path_obj.stem}_tracking_{suffix}.mp4"
    timeline_mode = str(cfg["output"]["timeline_mode"])
    
    # ✅ CARREGAR MODELOS COM BASE NO CONFIG
    det_model, cls_model, backend_info = load_models(det_weights, cls_weights, config, device)
    
    # ✅ WARM-UP CONFIGURÁVEL
    enable_warmup = cfg["performance"].get("enable_warmup", True)
    if enable_warmup:
        warmup_iterations = cfg["performance"].get("warmup_iterations", 3)
        print(f"🔥 Warm-up ({warmup_iterations} iterações)...")
        
        det_imgsz = cfg["detection"]["imgsz_cpu"]
        cls_imgsz = cfg["classification_model"]["imgsz"]
        
        dummy_det = np.zeros((det_imgsz, det_imgsz, 3), dtype=np.uint8)
        dummy_cls = Image.fromarray(np.zeros((cls_imgsz, cls_imgsz, 3), dtype=np.uint8))
        
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = det_model.predict(dummy_det, device=device, verbose=False)
                _ = cls_model.predict(dummy_cls, device=device, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Setup do FrameWriter
    processed_queue = Queue(maxsize=30)
    frame_stride = cfg["performance"]["frame_stride"]
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
    
    # ✅ TODAS AS CONFIGURAÇÕES VÊM DO CONFIG
    det_conf = cfg["detection"]["conf_threshold"]
    det_iou = cfg["detection"]["iou_threshold"]
    max_det = cfg["detection"]["max_det"]
    center_crop_margin = cfg["classification_model"]["center_crop_margin"]
    classification_interval = cfg["sampling"]["classify_every"]
    use_half = cfg["inference"]["half_precision"] and is_gpu
    
    # Setup do loop principal
    tracks = defaultdict(lambda: {"start_frame": -1, "end_frame": 0, "color_votes": defaultdict(float), "confidences": []})
    last_known_color = {}
    fps_tracker = FPSTracker()
    total_frames_proc = max(1, video_info['total_frames'] // frame_stride)
    
    classification_counter = 0
    
    print(f"🚀 Configurações aplicadas:")
    print(f"   📹 {video_info['total_frames']} frames @ {video_info['fps']:.2f} FPS")
    print(f"   ⚡ Frame stride: {frame_stride} (~{total_frames_proc} frames)")
    print(f"   🎨 Classify every: {classification_interval} frames")
    print(f"   📦 Batch size: {cfg['classification_model']['batch_size']}")
    print(f"   🔧 Backend: {backend_info['detection']} | {backend_info['classification']}")
    
    # ✅ STREAM YOLO CONFIGURÁVEL
    stream_config = {
        'source': str(video_path_obj),
        'stream': True,
        'persist': True,
        'tracker': cfg["tracker"],
        'device': device,
        'imgsz': cfg["detection"]["imgsz_cpu"],
        'conf': det_conf,
        'iou': det_iou,
        'half': use_half,
        'verbose': False,
        'max_det': max_det,
        'save': False,
        'show': False,
    }
    
    if frame_stride > 1:
        stream_config['vid_stride'] = frame_stride
    
    stream = det_model.track(**stream_config)
    
    try:
        for processed_frame_idx, result in enumerate(stream):
            frame_idx = processed_frame_idx * frame_stride
            
            # ✅ TIMING COM CUDA SYNC
            det_start = time.perf_counter()
            processed_frame = result.orig_img.copy()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            det_time = time.perf_counter() - det_start
            
            # ✅ CLASSIFICAÇÃO CONFIGURÁVEL
            classification_counter += 1
            should_classify = classification_counter % classification_interval == 0
            
            need_cls_infos = []
            num_detections = 0
            cls_time = 0
            
            if result.boxes is not None and result.boxes.id is not None:
                num_detections = len(result.boxes)
                
                for box in result.boxes:
                    if box.id is None:
                        continue
                    
                    try:
                        track_id = int(box.id[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if tracks[track_id]["start_frame"] == -1:
                            tracks[track_id]["start_frame"] = frame_idx
                        tracks[track_id]["end_frame"] = frame_idx
                        
                        if should_classify:
                            cx1, cy1, cx2, cy2 = get_center_crop(x1, y1, x2, y2, center_crop_margin)
                            min_size = cfg["classification_model"].get("min_crop_size", 10)
                            if (cx2 - cx1) >= min_size and (cy2 - cy1) >= min_size:
                                need_cls_infos.append((cx1, cy1, cx2, cy2, track_id))
                                
                    except (TypeError, IndexError, ValueError):
                        continue
                
                # ✅ CLASSIFICAÇÃO CONFIGURÁVEL
                if need_cls_infos:
                    cls_start = time.perf_counter()
                    
                    # Criar crops
                    crops = []
                    valid_infos = []
                    
                    for (x1, y1, x2, y2, tid) in need_cls_infos:
                        # Validar coordenadas
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(processed_frame.shape[1], x2)
                        y2 = min(processed_frame.shape[0], y2)
                        
                        min_size = cfg["classification_model"].get("min_crop_size", 10)
                        if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                            crop = processed_frame[y1:y2, x1:x2].copy()
                            crops.append(crop)
                            valid_infos.append((x1, y1, x2, y2, tid))
                    
                    if crops:
                        new_labels = classify_colors_smart(
                            cls_model, crops, valid_infos, config, device
                        )
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        cls_time = time.perf_counter() - cls_start
                        
                        # Processar resultados
                        for tid, (color_name, conf) in new_labels.items():
                            tracks[tid]["color_votes"][color_name] += conf
                            tracks[tid]["confidences"].append(conf)
                            last_known_color[tid] = (color_name, conf)
            
            # ✅ VISUALIZAÇÃO CONFIGURÁVEL
            vis_cfg = cfg["visualization"]
            if vis_cfg["draw_boxes"] and result.boxes is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    
                    try:
                        track_id = int(box.id[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Cor consistente por track
                        np.random.seed(track_id)
                        box_color = tuple(int(c) for c in np.random.randint(64, 255, size=3))
                        
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
                    except (TypeError, IndexError, ValueError):
                        continue
            
            # Overlay com informações
            fps_stats = fps_tracker.update(det_time, cls_time)
            fps_raw_equiv = fps_stats["instant"] * frame_stride
            eta = fps_tracker._format_time((total_frames_proc - processed_frame_idx - 1) / max(1e-6, fps_stats["smoothed"]))
            
            draw_metrics_overlay(processed_frame, fps_stats, processed_frame_idx,
                               total_frames_proc, eta, num_detections, fps_raw_equiv)
            
            # Envia para escrita
            try:
                processed_queue.put({
                    'processed_frame_idx': processed_frame_idx,
                    'processed_frame': processed_frame
                }, timeout=0.1)
            except Full:
                pass
            
            if progress_cb and processed_frame_idx % 10 == 0:
                progress_cb(min(processed_frame_idx + 1, total_frames_proc), total_frames_proc)
    
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
    
    # Resultados
    performance_summary = fps_tracker.get_summary()
    performance_summary.update({
        "fps_model": performance_summary["average_fps"],
        "fps_raw_equiv": performance_summary["average_fps"] * frame_stride,
        "processed_frames_total": fps_tracker.total_frames_processed,
        "timeline_mode": timeline_mode,
        "config_used": cfg,
        "backend_info": backend_info
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
            "timeline_mode": timeline_mode
        },
        "hardware_info": {
            "device": device,
            "backend_detection": backend_info["detection"],
            "backend_classification": backend_info["classification"],
            "hardware_detected": hardware_info
        },
        "processing_parameters": {
            "tracker": cfg["tracker"],
            "detection": {"imgsz": cfg["detection"]["imgsz_cpu"], "conf_threshold": det_conf, "iou_threshold": det_iou, "max_det": max_det},
            "classification": {"min_confidence": cfg["classification"]["min_confidence"], "imgsz": cfg["classification_model"]["imgsz"], "batch_size": cfg["classification_model"]["batch_size"]},
            "performance": cfg["performance"],
        },
        "performance_metrics": performance_summary,
        "tracks": final_records,
        "summary": {"total_tracks": len(final_records), "color_distribution": dict(color_counts)}
    }
    
    # Save
    output_paths = {}
    if cfg["output"]["save_json"]:
        json_path = out_dir_path / f"{video_path_obj.stem}_tracking_{suffix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        output_paths["json"] = str(json_path)
    
    if cfg["output"]["save_csv"] and final_records:
        csv_path = out_dir_path / f"{video_path_obj.stem}_tracking_{suffix}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=final_records[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(final_records)
        output_paths["csv"] = str(csv_path)
    
    print(f"\n🎉 TRACKING CONFIGURÁVEL concluído:")
    print(f"   📊 {len(final_records)} tracks detectados!")
    print(f"   ⚡ Performance: {performance_summary['average_fps']:.1f} FPS")
    print(f"   🚀 Raw equivalent: {performance_summary['fps_raw_equiv']:.1f} FPS")
    print(f"   🎯 Backend: {backend_info['detection']} + {backend_info['classification']}")
    
    return {
        "json": output_paths.get("json", ""),
        "csv": output_paths.get("csv", ""),
        "video_annotated": str(out_video_path),
        "output_dir": str(out_dir_path),
        "total_tracks": len(final_records),
        "output_files": output_paths,
        "performance": performance_summary,
        "color_distribution": dict(color_counts)
    }