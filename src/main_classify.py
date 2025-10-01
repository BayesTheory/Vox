# src/main_classify.py
import argparse
from pathlib import Path
from ultralytics import YOLO

from src.utils.utils import (
    seed_everything, setup_mlflow, start_run,
    log_training_config, log_metrics, log_artifacts,
    create_balanced_dataset
)

def ensure_cls_dir(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Diretório do dataset não existe: {root}")
    for split in ("train", "val"):
        sp = root / split
        if not sp.exists() or not any(p.is_dir() for p in sp.iterdir()):
            raise FileNotFoundError(f"Split '{split}' ausente ou sem subpastas de classe em: {sp}")

def train_entry(args: argparse.Namespace):
    data_dir = Path(args.data).resolve()
    ensure_cls_dir(data_dir)

    # Balanceamento opcional
    balanced_root = data_dir
    if args.balance_mode == "oversample":
        balanced_root = create_balanced_dataset(
            data_dir, mode="oversample", ratio=args.balance_ratio, suffix="_oversampled"
        )

    # Seed e MLflow
    seed_everything(42)
    cfg = {
        "mlflow": {"tracking_uri": args.mlflow_uri, "experiment_name": args.mlflow_exp},
        "datasets": {"classification": {"path": str(balanced_root), "yaml": ""}},
        "training": {"seed": 42},
    }
    setup_mlflow(cfg)

    # Seleção de pesos por variante
    weights = args.weights_n if args.variant == "n" else args.weights_s
    run_name = f"yolo11{args.variant}_classification_{args.suffix}"

    with start_run(run_name=run_name, tags={"family": "yolo11", "variant": args.variant, "task": "classify"}):
        log_training_config(cfg, "classification", f"yolo11{args.variant}")
        model = YOLO(str(weights))
        results = model.train(
            data=str(balanced_root),   # pasta raiz (train/val/test)
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            # Augmentations de classificação
            fliplr=args.fliplr,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            auto_augment=args.auto_augment,
            erasing=args.erasing,
            # LR
            lr0=args.lr0,
            lrf=args.lrf,
            cos_lr=args.cos_lr,
            # Execução
            name=run_name,
            project="runs",
            amp=True,
            patience=5,
            save_period=5,
            val=True,
            resume=False,
        )
        if results is not None and hasattr(results, "results_dict"):
            log_metrics(results.results_dict, prefix="final_metrics")
        # Artefatos
        best = Path(f"runs/{run_name}/weights/best.pt")
        last = Path(f"runs/{run_name}/weights/last.pt")
        arts = [p for p in (best, last) if p.exists()]
        if arts:
            log_artifacts(arts, artifact_path="artifacts")
    print(f"✅ Classificação concluída: {run_name}")

def parse_args():
    ap = argparse.ArgumentParser(description="Classificação de cores com YOLO11 (-cls.pt)")
    ap.add_argument("--data", type=str, required=True, help="Pasta raiz com train/val/test")
    ap.add_argument("--variant", type=str, default="n", choices=["n", "s"], help="Variante do YOLO11-cls")
    ap.add_argument("--weights-n", type=str, default="yolo11n-cls.pt", help="Pesos YOLO11n-cls")
    ap.add_argument("--weights-s", type=str, default="yolo11s-cls.pt", help="Pesos YOLO11s-cls")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--workers", type=int, default=16)
    # Augmentations
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--degrees", type=float, default=10.0)
    ap.add_argument("--translate", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=0.15)
    ap.add_argument("--auto_augment", type=str, default="randaugment")
    ap.add_argument("--erasing", type=float, default=0.4)
    # LR
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--lrf", type=float, default=0.1)
    ap.add_argument("--cos-lr", dest="cos_lr", action="store_true")
    # Balanceamento
    ap.add_argument("--balance-mode", type=str, default="oversample", choices=["none", "oversample"])
    ap.add_argument("--balance-ratio", type=float, default=0.6)
    # Execução e MLflow
    ap.add_argument("--suffix", type=str, default="colors_cls")
    ap.add_argument("--mlflow-uri", type=str, default="./mlruns")
    ap.add_argument("--mlflow-exp", type=str, default="YOLO_Training")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_entry(args)
