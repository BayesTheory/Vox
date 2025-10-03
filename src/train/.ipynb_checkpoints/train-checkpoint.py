# src/train/train.py
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from src.utils.utils import (
    load_config, validate_dataset_path, seed_everything,
    setup_mlflow, start_run, log_training_config, log_metrics, log_artifacts
)

FAMILY_MODULE = {"yolo11": "src.Modelos.yolo11"}
ALIASES = {"YOLO11": "yolo11", "Yolo11": "yolo11"}

def _import_family_module(fam: str):
    name = FAMILY_MODULE[fam]
    return importlib.import_module(name)

def _normalize_family(f: str) -> str:
    return ALIASES.get(f, f)

def _resolve_task_key(task: str) -> str:
    return "detect" if task in ("detect", "detection") else "classify"

def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica overrides recursivamente, preservando estrutura aninhada"""
    result = cfg.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _apply_overrides(result[key], value)
        else:
            result[key] = value
    return result

def train_detection_pipeline(
    config_path: str = "src/config.json",
    variant: str = "n",
    part1_weights: Optional[str] = None,
    part2_weights: Optional[str] = None,
    force_part1: bool = False,
    force_part2: bool = False,
    overrides: Optional[Dict[str, Any]] = None
):
    """
    Pipeline completo de detecção: Parte 1 (opcional) + Parte 2.
    Preparado para CI/CD com controle fino de estágios.
    """
    cfg = load_config(config_path)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    
    family = _normalize_family(cfg["model_family"])
    dataset = cfg["datasets"]["detection"]
    
    # Validação dataset
    ok, msg = validate_dataset_path(dataset["path"], dataset["yaml"])
    if not ok:
        print(f"❌ {msg}")
        return None
    print(f"✅ {msg}")
    
    # Setup básico
    seed_everything(cfg["training"]["seed"])
    setup_mlflow(cfg)
    mod = _import_family_module(family)
    
    results = None
    current_weights = part1_weights or f"yolo11{variant}.pt"  # default pré-treino
    
    # Parte 1 (transfer learning básico) - normalmente desabilitada
    part1_cfg = cfg["stages"]["detect"]["part1"]
    if (part1_cfg.get("enabled", False) or force_part1) and not part2_weights:
        print("🔄 Executando Parte 1 (transfer learning básico)")
        run_name = f"yolo11{variant}_detection_part1"
        stages = [part1_cfg]
        
        training_cfg = {**cfg["training"], "weights": current_weights}
        
        with start_run(run_name=run_name, tags={"family": family, "variant": variant, "task": "detect", "phase": "part1"}):
            log_training_config(cfg, "detection", f"yolo11{variant}")
            results = mod.train(dataset, training_cfg, variant, run_name, stages=stages)
            
            if results and hasattr(results, "results_dict"):
                log_metrics(results.results_dict, prefix="part1_metrics")
            
            # Atualiza weights para Parte 2
            best_path = Path(f"runs/{run_name}/weights/best.pt")
            if best_path.exists():
                current_weights = str(best_path)
                log_artifacts([best_path], artifact_path="part1_artifacts")
        
        print(f"✅ Parte 1 concluída: {run_name}")
    
    # Parte 2 (fine-tuning especializado) - normalmente habilitada
    part2_cfg = cfg["stages"]["detect"]["part2"]
    if part2_cfg.get("enabled", True) or force_part2:
        print("🔄 Executando Parte 2 (fine-tuning especializado)")
        if part2_weights:  # Override para CI/CD
            current_weights = part2_weights
        
        run_name = f"yolo11{variant}_detection_part2"
        stages = [part2_cfg]
        
        training_cfg = {**cfg["training"], "weights": current_weights}
        
        with start_run(run_name=run_name, tags={"family": family, "variant": variant, "task": "detect", "phase": "part2"}):
            log_training_config(cfg, "detection", f"yolo11{variant}")
            results = mod.train(dataset, training_cfg, variant, run_name, stages=stages)
            
            if results and hasattr(results, "results_dict"):
                log_metrics(results.results_dict, prefix="part2_metrics")
            
            best_path = Path(f"runs/{run_name}/weights/best.pt")
            last_path = Path(f"runs/{run_name}/weights/last.pt")
            arts = [p for p in (best_path, last_path) if p.exists()]
            if arts:
                log_artifacts(arts, artifact_path="part2_artifacts")
        
        print(f"✅ Parte 2 concluída: {run_name}")
    
    return results

def train_classification_pipeline(
    config_path: str = "src/config.json",
    variant: str = "n",
    base_weights: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
):
    """
    Pipeline de classificação de cores com balanceamento automático.
    Preparado para CI/CD.
    """
    cfg = load_config(config_path)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    
    family = _normalize_family(cfg["model_family"])
    dataset = cfg["datasets"]["classification"]
    cls_cfg = cfg["stages"]["classify"]
    
    if not cls_cfg.get("enabled", True):
        print("⚠️ Classificação desabilitada no config")
        return None
    
    # Validação dataset (pasta raiz para classify)
    data_root = Path(dataset["path"])
    if not data_root.exists():
        print(f"❌ Dataset de classificação não encontrado: {data_root}")
        return None
    
    # Balanceamento automático se habilitado
    balanced_root = data_root
    balance_cfg = dataset.get("balance", {})
    if balance_cfg.get("enabled", False):
        from src.utils.utils import create_balanced_dataset
        balanced_root = create_balanced_dataset(
            data_root, 
            balance_cfg.get("mode", "oversample"),
            balance_cfg.get("ratio", 0.6)
        )
        print(f"✅ Dataset balanceado criado: {balanced_root}")
    
    # Setup básico
    seed_everything(cfg["training"]["seed"])
    setup_mlflow(cfg)
    mod = _import_family_module(family)
    
    run_name = f"yolo11{variant}_classification"
    weights = base_weights or f"yolo11{variant}-cls.pt"
    
    training_cfg = {**cfg["training"], "weights": weights}
    stages = [cls_cfg]
    
    # Override dataset path para versão balanceada
    dataset_override = {**dataset, "path": str(balanced_root)}
    
    with start_run(run_name=run_name, tags={"family": family, "variant": variant, "task": "classify"}):
        log_training_config(cfg, "classification", f"yolo11{variant}")
        results = mod.train(dataset_override, training_cfg, variant, run_name, stages=stages)
        
        if results and hasattr(results, "results_dict"):
            log_metrics(results.results_dict, prefix="classify_metrics")
        
        best_path = Path(f"runs/{run_name}/weights/best.pt")
        last_path = Path(f"runs/{run_name}/weights/last.pt")
        arts = [p for p in (best_path, last_path) if p.exists()]
        if arts:
            log_artifacts(arts, artifact_path="classify_artifacts")
    
    print(f"✅ Classificação concluída: {run_name}")
    return results

def train_from_config(config_path: str = "src/config.json", overrides: Optional[Dict[str, Any]] = None):
    """Wrapper legacy para compatibilidade"""
    if overrides and overrides.get("task") == "detect":
        variant = overrides.get("variant", "n")
        return train_detection_pipeline(config_path, variant, overrides=overrides)
    elif overrides and overrides.get("task") == "classify":
        variant = overrides.get("variant", "n")
        return train_classification_pipeline(config_path, variant, overrides=overrides)
    else:
        print("⚠️ Task não especificada nos overrides")
        return None
