# src/main_interactive.py
from pathlib import Path
import argparse

from src.main import cmd_train_detect, cmd_train_classify, cmd_track, cmd_api  # reusa funções do main atual

def prompt_choice(title, options):
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        sel = input("Escolha [número]: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(options):
            return int(sel) - 1
        print("Inválido. Tente novamente.")

def prompt_text(msg, default=None):
    s = input(f"{msg}{' ['+default+']' if default else ''}: ").strip()
    return s or (default or "")

def main():
    print("=== Vehicle Color Pipeline ===")
    mode_idx = prompt_choice("O que deseja fazer?", [
        "Treinar DETECÇÃO (Parte 2, evitar double training)",
        "Treinar CLASSIFICAÇÃO de cores",
        "Rodar TRACKING em vídeo",
        "Subir API FastAPI"
    ])

    if mode_idx == 0:
        # DETECT Part 2
        variant = prompt_text("Variante (n/s)", "n")
        best = prompt_text("Caminho do best.pt (detector)", "runs/yolo11n_detection_detect2/weights/best.pt" if variant=="n" else "runs/yolo11s_detection_detect2/weights/best.pt")
        epochs = int(prompt_text("Epochs", "18"))
        freeze = int(prompt_text("Freeze", "5"))
        batch = int(prompt_text("Batch", "384"))
        imgsz = int(prompt_text("Image size", "640"))
        device = prompt_text("Device", "0")
        workers = int(prompt_text("Workers", "16"))
        mosaic = float(prompt_text("Mosaic", "0.2"))
        lr0 = float(prompt_text("lr0", "0.005"))
        lrf = float(prompt_text("lrf", "0.01"))
        patience = int(prompt_text("Patience", "5"))
        # Monta args Namespace e chama função existente
        class A: pass
        args = A()
        args.config="src/config.json"; args.variant=variant
        args.best=best; args.epochs=epochs; args.freeze=freeze
        args.batch=batch; args.imgsz=imgsz; args.device=device
        args.workers=workers; args.mosaic=mosaic; args.lr0=lr0; args.lrf=lrf
        args.patience=patience
        from src.main import cmd_detect_part2 as run_detect_part2
        run_detect_part2(args)

    elif mode_idx == 1:
        # CLASSIFY
        variant = prompt_text("Variante (n/s)", "n")
        data = prompt_text("Pasta raiz do dataset (train/val/test)", "data/2/COLOR_FINAL_YOLO")
        weights = prompt_text("Pesos base (-cls.pt)", f"yolo11{variant}-cls.pt")
        epochs = int(prompt_text("Epochs", "20"))
        batch = int(prompt_text("Batch", "256"))
        imgsz = int(prompt_text("Image size", "224"))
        fliplr = float(prompt_text("fliplr", "0.5"))
        degrees = float(prompt_text("degrees", "10"))
        translate = float(prompt_text("translate", "0.05"))
        scale = float(prompt_text("scale", "0.15"))
        auto_augment = prompt_text("auto_augment", "randaugment")
        erasing = float(prompt_text("erasing", "0.4"))
        lr0 = float(prompt_text("lr0", "0.005"))
        lrf = float(prompt_text("lrf", "0.1"))
        cos_lr = prompt_text("cos_lr (y/n)", "y").lower().startswith("y")
        balance = prompt_text("Balancear (oversample none)", "oversample")
        ratio = float(prompt_text("Balance ratio", "0.6"))
        # Encaminha para o main de classificação (sua função train_entry/atual)
        from src.main_classify import train_entry
        class A: pass
        args = A()
        args.data=data; args.variant=variant; args.weights_n=weights; args.weights_s=weights
        args.epochs=epochs; args.batch=batch; args.imgsz=imgsz
        args.fliplr=fliplr; args.degrees=degrees; args.translate=translate; args.scale=scale
        args.auto_augment=auto_augment; args.erasing=erasing
        args.lr0=lr0; args.lrf=lrf; args.cos_lr=cos_lr
        args.balance_mode=balance; args.balance_ratio=ratio; args.balance_out=""; args.copy=False
        args.device="0"; args.workers=16; args.mlflow_uri="./mlruns"; args.mlflow_exp="YOLO_Training"; args.suffix="colors_cls"
        train_entry(args)

    elif mode_idx == 2:
        # TRACK
        video = prompt_text("Vídeo (.mp4)", "input.mp4")
        det_w = prompt_text("Detector best.pt", "runs/yolo11n_detection_detect3/weights/best.pt")
        cls_w = prompt_text("Classificador best.pt (ou -cls.pt)", "runs/yolo11n_classification_colors_n3/weights/best.pt")
        out = prompt_text("Diretório de saída", "")
        class A: pass
        args = A()
        args.video=video; args.det_weights=det_w; args.cls_weights=cls_w; args.config="src/config.json"; args.output=out
        from src.main import cmd_track as run_track
        run_track(args)

    else:
        # API
        host = prompt_text("Host", "0.0.0.0")
        port = int(prompt_text("Porta", "8000"))
        reload = prompt_text("Reload (y/n)", "n").lower().startswith("y")
        class A: pass
        args = A()
        args.host=host; args.port=port; args.reload=reload
        from src.main import cmd_api as run_api
        run_api(args)

if __name__ == "__main__":
    main()
