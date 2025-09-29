# src/utils/utils.py
from __future__ import annotations

import os
import json
import time
import random
import contextlib
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

import mlflow

# Importes opcionais (não quebram se ausentes)
try:
    import yaml
except Exception:
    yaml = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


# ----------------------------
# Arquivos e validações
# ----------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def path_exists(path: Path | str) -> bool:
    return Path(path).exists()


def is_dir(path: Path | str) -> bool:
    return Path(path).is_dir()


def list_images(root: Path | str) -> List[Path]:
    root = Path(root)
    out = []
    for ext in IMG_EXTS:
        out.extend(root.rglob(f"*{ext}"))
    return sorted(out)


def validate_dataset_path(dataset_path: str, yaml_file: str) -> Tuple[bool, str]:
    """Valida se a pasta do dataset e o YAML existem (formato Ultralytics)."""
    base = Path(dataset_path)
    if not base.exists():
        return False, f"Dataset path not found: {base}"
    yml = base / yaml_file
    if not yml.exists():
        return False, f"YAML file not found: {yml}"
    return True, f"Dataset OK: {base} | YAML: {yml}"


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, out_path: Path | str):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_yaml(yaml_path: Path | str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML não instalado; pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: Dict[str, Any], yaml_path: Path | str):
    if yaml is None:
        raise RuntimeError("PyYAML não instalado; pip install pyyaml")
    yaml_path = Path(yaml_path)
    ensure_dir(yaml_path.parent)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


# ----------------------------
# Reprodutibilidade e device
# ----------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(preferred: str = "cpu") -> str:
    if preferred and preferred != "auto":
        return preferred
    if torch is not None and torch.cuda.is_available():
        return "0"
    return "cpu"


# ----------------------------
# MLflow: setup e logging
# ----------------------------

def setup_mlflow(config: Dict[str, Any]):
    """Configura MLflow tracking a partir do JSON."""
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "./mlruns")
    experiment_name = config.get("mlflow", {}).get("experiment_name", "YOLO_Training")
    mlflow.set_tracking_uri(tracking_uri)
    # Cria/seleciona o experimento
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        # Em alguns backends, get_experiment_by_name pode falhar se o tracking não existir ainda
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)



@contextlib.contextmanager
def start_run(run_name: str, tags: Optional[Dict[str, str]] = None, run_id: Optional[str] = None):
    """
    Contexto seguro para iniciar/reativar um run do MLflow.
    Se run_id for fornecido, reabre o run anterior; caso contrário, cria novo.
    """
    if run_id:
        with mlflow.start_run(run_id=run_id) as run:
            if tags:
                mlflow.set_tags(tags)
            yield run
    else:
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            yield run

def log_training_config(config: Dict[str, Any], dataset_key: str, model_label: str):
    """
    Registra no MLflow os parâmetros principais do treino e informações do dataset.
    model_label pode ser algo como 'yolo11n' ou 'yolov8s'.
    """
    ds = config.get("datasets", {}).get(dataset_key, {})
    tr = config.get("training", {})

    core_params = {
        "model_label": model_label,
        "dataset_key": dataset_key,
        "ds_name": ds.get("name", dataset_key),
        "ds_task": ds.get("task", ""),
        "ds_path": ds.get("path", ""),
        "ds_yaml": ds.get("yaml", ""),
        "epochs": tr.get("epochs", 0),
        "imgsz": tr.get("imgsz", 0),
        "batch": tr.get("batch", 0),
        "device": tr.get("device", "cpu"),
        "patience": tr.get("patience", 0),
        "save_period": tr.get("save_period", 0),
        "seed": tr.get("seed", 42),
    }

    # Log como params simples no MLflow
    for k, v in core_params.items():
        mlflow.log_param(k, v)

def log_params_flat(d: Dict[str, Any], prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        mlflow.log_param(key, v)


def log_metrics(d: Dict[str, float], step: Optional[int] = None):
    for k, v in d.items():
        mlflow.log_metric(k, float(v), step=step if step is not None else None)


def log_artifacts(paths: Iterable[Path | str], artifact_path: Optional[str] = None):
    for p in paths:
        p = Path(p)
        if p.is_file():
            mlflow.log_artifact(str(p), artifact_path=artifact_path)
        elif p.is_dir():
            mlflow.log_artifacts(str(p), artifact_path=artifact_path)


def get_active_run_id() -> Optional[str]:
    ar = mlflow.active_run()
    return ar.info.run_id if ar else None


# ----------------------------
# Medição de tempo e estágios
# ----------------------------

@contextlib.contextmanager
def timer(name: str = "stage"):
    t0 = time.time()
    yield
    dt = time.time() - t0
    mlflow.log_metric(f"time_{name}_sec", dt)


def describe_stage(stage_cfg: Dict[str, Any]) -> str:
    keys = ["desc", "epochs", "freeze", "imgsz", "lr0", "lrf", "cosine", "mosaic", "hsv_h", "hsv_s", "hsv_v"]
    parts = [f"{k}={stage_cfg[k]}" for k in keys if k in stage_cfg]
    return ", ".join(parts)


def log_stage(stage_idx: int, stage_cfg: Dict[str, Any]):
    mlflow.log_param(f"stage_{stage_idx}", describe_stage(stage_cfg))


# ----------------------------
# Manifestos e QA
# ----------------------------

def dataset_manifest(root: str, yaml_file: str) -> Dict[str, Any]:
    base = Path(root)
    yml = base / yaml_file
    stats = {
        "root": str(base.resolve()),
        "yaml": str(yml.resolve()),
        "exists": base.exists(),
        "yaml_exists": yml.exists(),
        "images_train": len(list_images(base / "images" / "train")),
        "images_val": len(list_images(base / "images" / "val")),
        "labels_train": len(list((base / "labels" / "train").glob("*.txt"))) if (base / "labels" / "train").exists() else 0,
        "labels_val": len(list((base / "labels" / "val").glob("*.txt"))) if (base / "labels" / "val").exists() else 0,
    }
    return stats


def save_manifest(root: str, yaml_file: str, out_path: str | Path):
    man = dataset_manifest(root, yaml_file)
    save_json(man, out_path)
    return man
