from pathlib import Path
from ultralytics import YOLO

DETECT_WEIGHTS = {"n": "yolov8n.pt", "s": "yolov8s.pt"}
CLASSIFY_WEIGHTS = {"n": "yolov8n-cls.pt", "s": "yolov8s-cls.pt"}

def get_model(variant: str, task: str) -> YOLO:
    variant = variant.lower()
    if task == "classify":
        return YOLO(CLASSIFY_WEIGHTS[variant])
    return YOLO(DETECT_WEIGHTS[variant])

def train_stage(model: YOLO, data_yaml: Path, base, stage, run_name: str):
    # mescla argumentos base + estágio
    params = {
        "data": str(data_yaml),
        "epochs": stage.get("epochs", base["epochs"]),
        "imgsz": stage.get("imgsz", base["imgsz"]),
        "batch": base["batch"],
        "device": base["device"],
        "patience": base["patience"],
        "save_period": base["save_period"],
        "project": "runs",
        "name": run_name,
        "freeze": stage.get("freeze", None),
        "lr0": stage.get("lr0", None),
        "lrf": stage.get("lrf", None),
        "cosine": stage.get("cosine", None),
        "mosaic": stage.get("mosaic", None),
        "hsv_h": stage.get("hsv_h", None),
        "hsv_s": stage.get("hsv_s", None),
        "hsv_v": stage.get("hsv_v", None),
        "fliplr": stage.get("fliplr", 0.5),
        "degrees": stage.get("degrees", 10),
        "translate": stage.get("translate", 0.05),
        "scale": stage.get("scale", 0.15),
    }
    # remove None (Ultralytics não aceita chaves None)
    params = {k: v for k, v in params.items() if v is not None}
    return model.train(**params)

def train(dataset_config: dict, training: dict, variant: str, run_name: str, stages: list | None = None):
    model = get_model(variant, dataset_config["task"])
    data_yaml = (Path(dataset_config["path"]) / dataset_config["yaml"]).resolve()
    if stages:
        results = None
        for stg in stages:
            results = train_stage(model, data_yaml, training, stg, run_name)
        return results
    else:
        # fallback single-shot
        return model.train(
            data=str(data_yaml),
            epochs=training["epochs"],
            imgsz=training["imgsz"],
            batch=training["batch"],
            device=training["device"],
            patience=training["patience"],
            save_period=training["save_period"],
            project="runs",
            name=run_name
        )
