"""
main_uni.py -- Script unificado que combina:
  * CLI avançada (train-detect, train-classify, track, api) do antigo main.py
  * Pipeline de treinamento de classificação do antigo main_classify.py
  * Interface interativa em linha de comando do antigo main_interactive.py

Uso:
  python main_uni.py --help        # mostra ajuda da CLI
  python main_uni.py interactive  # inicia modo interativo (menus de texto)
  python main_uni.py              # inicia modo interativo (padrão)
"""
import warnings
import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional

# ✅ Silencia TUDO
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["ONNX_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Dependências externas (devem existir no ambiente do projeto)
from ultralytics import YOLO  # type: ignore

from src.utils.utils import (
    seed_everything,
    setup_mlflow,
    start_run,
    log_training_config,
    log_metrics,
    log_artifacts,
    create_balanced_dataset,
)
from src.cli.commands import (
    cmd_train_detect,
    cmd_train_classify,
    cmd_track,
    cmd_api,
)

logger = logging.getLogger(__name__)

###############################################################################
# 0) Função auxiliar para apagar ONNX (usado antes de tracking)
###############################################################################

def force_delete_onnx(weights_path: str) -> None:
    """✅ APAGA ONNX SEMPRE (independente de dimensão)"""
    onnx_path = Path(weights_path).with_suffix(".onnx")
    if onnx_path.exists():
        print(f"🗑️  Apagando ONNX existente: {onnx_path.name}")
        onnx_path.unlink()

###############################################################################
# 1) Lógica específica de CLASSIFICAÇÃO (antigo main_classify.py)
###############################################################################

def ensure_cls_dir(root: Path) -> None:
    """Valida se o diretório de classificação contém splits e subpastas de classe."""
    if not root.exists():
        raise FileNotFoundError(f"Diretório do dataset não existe: {root}")
    for split in ("train", "val"):
        sp = root / split
        if not sp.exists() or not any(p.is_dir() for p in sp.iterdir()):
            raise FileNotFoundError(
                f"Split '{split}' ausente ou sem subpastas de classe em: {sp}"
            )


def train_classification(args: argparse.Namespace) -> None:
    """Treinamento de classificação de cores com YOLO11-cls."""
    data_dir = Path(args.data).resolve()
    ensure_cls_dir(data_dir)

    # Balanceamento opcional
    balanced_root = data_dir
    if args.balance_mode == "oversample":
        balanced_root = create_balanced_dataset(
            data_dir,
            mode="oversample",
            ratio=args.balance_ratio,
            suffix="_oversampled",
        )

    # Semente e MLflow
    seed_everything(42)
    cfg = {
        "mlflow": {
            "tracking_uri": args.mlflow_uri,
            "experiment_name": args.mlflow_exp,
        },
        "datasets": {"classification": {"path": str(balanced_root), "yaml": ""}},
        "training": {"seed": 42},
    }
    setup_mlflow(cfg)

    # Seleção de pesos por variante
    weights = args.weights_n if args.variant == "n" else args.weights_s
    run_name = f"yolo11{args.variant}_classification_{args.suffix}"

    with start_run(
        run_name=run_name,
        tags={"family": "yolo11", "variant": args.variant, "task": "classify"},
    ):
        log_training_config(cfg, "classification", f"yolo11{args.variant}")
        model = YOLO(str(weights))
        results = model.train(
            data=str(balanced_root),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            # Augmentations
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

###############################################################################
# 2) Construtor da CLI (antigo main.py) + inclusão de 'interactive'
###############################################################################

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VehicleColorAI – CLI & Interactive")
    p.add_argument("--config", default="src/config.json")

    sub = p.add_subparsers(dest="command", required=False)

    # ---------- train-detect ----------
    d = sub.add_parser("train-detect", help="Treinar detector (YOLO11)")
    d.add_argument("--variant", choices=["n", "s"], default="n")
    d.add_argument("--part1-weights")
    d.add_argument("--part2-weights")
    d.add_argument("--force-part1", action="store_true")
    d.add_argument("--force-part2", action="store_true")
    d.set_defaults(func=cmd_train_detect)

    # ---------- train-classify ----------
    c = sub.add_parser("train-classify", help="Treinar classificador de cores")
    c.add_argument("--data", required=True)
    c.add_argument("--variant", choices=["n", "s"], default="n")
    c.add_argument("--weights-n", default="yolo11n-cls.pt")
    c.add_argument("--weights-s", default="yolo11s-cls.pt")
    c.add_argument("--epochs", type=int, default=20)
    c.add_argument("--batch", type=int, default=256)
    c.add_argument("--imgsz", type=int, default=224)
    c.add_argument("--device", default="cpu")
    c.add_argument("--workers", type=int, default=16)
    # Augs
    c.add_argument("--fliplr", type=float, default=0.5)
    c.add_argument("--degrees", type=float, default=10.0)
    c.add_argument("--translate", type=float, default=0.05)
    c.add_argument("--scale", type=float, default=0.15)
    c.add_argument("--auto_augment", default="randaugment")
    c.add_argument("--erasing", type=float, default=0.4)
    # LR
    c.add_argument("--lr0", type=float, default=0.005)
    c.add_argument("--lrf", type=float, default=0.1)
    c.add_argument("--cos-lr", dest="cos_lr", action="store_true")
    # Balance
    c.add_argument("--balance-mode", choices=["none", "oversample"], default="oversample")
    c.add_argument("--balance-ratio", type=float, default=0.6)
    # MLflow / Exec
    c.add_argument("--suffix", default="colors_cls")
    c.add_argument("--mlflow-uri", default="./mlruns")
    c.add_argument("--mlflow-exp", default="YOLO_Training")
    c.set_defaults(func=train_classification)

    # ---------- track ----------
    t = sub.add_parser("track", help="Rodar tracking em vídeo")
    t.add_argument("--video", required=True)
    t.add_argument("--det-weights", required=True)
    t.add_argument("--cls-weights", required=True)
    t.add_argument("--output", default="")
    t.set_defaults(func=cmd_track)

    # ---------- api ----------
    a = sub.add_parser("api", help="Subir API FastAPI")
    a.add_argument("--host", default="0.0.0.0")
    a.add_argument("--port", type=int, default=8000)
    a.add_argument("--reload", action="store_true")
    a.set_defaults(func=cmd_api)

    # ---------- interactive ----------
    sub.add_parser("interactive", help="Modo interativo (menus de texto)").set_defaults(func=interactive_main)

    return p

###############################################################################
# 3) Interface Interativa (antigo main_interactive.py)
###############################################################################

def prompt_choice(title, options):
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f" {i}) {opt}")
    while True:
        sel = input("Escolha [número]: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return int(sel) - 1
        print("Inválido. Tente novamente.")


def prompt_text(msg, default=None):
    s = input(f"{msg}{' [' + str(default) + ']' if default else ''}: ").strip()
    return s or (default or "")


def interactive_main(_args: Optional[argparse.Namespace] = None):
    """Fluxo de menus interativos replicando main_interactive.py."""
    print("\n=== Vehicle Color Pipeline – Modo Interativo ===\n")
    mode_idx = prompt_choice(
        "O que deseja fazer?",
        [
            "Treinar DETECÇÃO (train-detect)",
            "Treinar CLASSIFICAÇÃO de cores",
            "Rodar TRACKING em vídeo",
            "Subir API FastAPI",
            "Sair",
        ],
    )

    if mode_idx == 0:
        variant = prompt_text("Variante (n/s)", "n")
        # Encaminha para CLI train-detect
        cli_args = [
            "train-detect",
            "--variant", variant,
        ]
        p = build_parser()
        args = p.parse_args(cli_args)
        args.func(args)

    elif mode_idx == 1:
        variant = prompt_text("Variante (n/s)", "n")
        data = prompt_text("Pasta raiz do dataset (train/val/test)", "data/2/COLOR_FINAL_YOLO")
        weights = prompt_text("Pesos base (-cls.pt)", f"yolo11{variant}-cls.pt")
        # Build args list
        cli_args = [
            "train-classify",
            "--data", data,
            "--variant", variant,
            "--weights-n", weights,
            "--weights-s", weights,
            "--cos-lr",  # habilita cos_lr
        ]
        p = build_parser()
        args = p.parse_args(cli_args)
        args.func(args)

    elif mode_idx == 2:
        video = prompt_text("Vídeo (.mp4)", "input.mp4")
        det_w = prompt_text("Detector best.pt", "runs/yolo11n_detection_detect3/weights/best.pt")
        cls_w = prompt_text(
            "Classificador best.pt (ou -cls.pt)",
            "runs/yolo11n_classification_colors_n3/weights/best.pt",
        )
        out = prompt_text("Diretório de saída", "")
        
        # ✅ APAGA ONNX ANTES DE CHAMAR O TRACKING
        print("\n🔄 Verificando arquivos ONNX existentes...")
        force_delete_onnx(det_w)
        force_delete_onnx(cls_w)
        
        cli_args = [
            "track",
            "--video", video,
            "--det-weights", det_w,
            "--cls-weights", cls_w,
        ]
        if out:
            cli_args.extend(["--output", out])
        p = build_parser()
        args = p.parse_args(cli_args)
        args.func(args)

    elif mode_idx == 3:
        host = prompt_text("Host", "127.0.0.1")
        port = int(prompt_text("Porta", "8000"))
        reload = prompt_text("Reload (y/n)", "n").lower().startswith("y")
        cli_args = [
            "api",
            "--host", host,
            "--port", str(port),
        ]
        if reload:
            cli_args.append("--reload")
        p = build_parser()
        args = p.parse_args(cli_args)
        args.func(args)
    else:
        print("🚪 Saindo do modo interativo.")

###############################################################################
# 4) Função principal
###############################################################################

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Se nenhum comando foi fornecido, inicia modo interativo por padrão
    if args.command is None:
        print("ℹ️  Nenhum comando especificado. Iniciando modo interativo...\n")
        interactive_main(args)
    else:
        return args.func(args)


if __name__ == "__main__":
    main()
