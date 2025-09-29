#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

# adiciona a raiz do projeto ao sys.path quando rodado como script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# importe diretamente do arquivo para evitar depender do __init__.py
from src.train.train import train_from_config  # ← muda aqui

def resolve_config_path(passed: str | None) -> Path:
    if passed:
        return Path(passed).resolve()
    local = Path(__file__).resolve().with_name("config.json")
    if local.exists():
        return local
    root_cfg = ROOT / "config.json"
    if root_cfg.exists():
        return root_cfg
    raise FileNotFoundError("config.json não encontrado; passe --config explicitamente.")

def main():
    ap = argparse.ArgumentParser(description="Treinamento YOLO (detecção e/ou classificação) com stages e MLflow")
    ap.add_argument("--config", help="Caminho para o config.json (padrão: src/config.json)")
    ap.add_argument("--family", choices=["yolov8","yolo11"])
    ap.add_argument("--variant", choices=["n","s"])
    ap.add_argument("--task", choices=["detect","classify"])
    ap.add_argument("--dataset_key", choices=["detection","classification"])
    args = ap.parse_args()

    cfg_path = resolve_config_path(args.config)

    # Overrides via CLI (opcional)
    if any([args.family, args.variant, args.task, args.dataset_key]):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if args.family: cfg["model_family"] = args.family
        if args.variant: cfg["variant"] = args.variant
        if args.task:    cfg["task"] = args.task
        if args.dataset_key: cfg["dataset_key"] = args.dataset_key
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    train_from_config(str(cfg_path))

if __name__ == "__main__":
    main()
