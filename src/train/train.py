import importlib
import mlflow
from pathlib import Path
from src.utils import (
    setup_mlflow, log_training_config, log_metrics,
    log_model_artifacts, validate_dataset_path, load_config
)

FAMILY_MODULE = {
    "yolov8": "src.Modelos.yolov8",
    "yolo11": "src.Modelos.yolo11",
}

def train_from_config(config_path="config.json"):
    cfg = load_config(config_path)
    dataset_key = cfg["dataset_key"]
    dataset = cfg["datasets"][dataset_key]
    family = cfg["model_family"]   # "yolov8" | "yolo11"
    variant = cfg["variant"]       # "n" | "s"
    task = cfg["task"]             # "detect" | "classify"

    # sanity: task da config e do dataset devem coincidir
    assert dataset["task"] == task, f"Task do dataset ({dataset['task']}) difere de task global ({task})"

    ok, msg = validate_dataset_path(dataset["path"], dataset["yaml"])
    if not ok:
        print(f"❌ {msg}")
        return None
    print(f"✅ {msg}")

    setup_mlflow(cfg)
    run_name = f"{family}{variant}_{dataset_key}_{task}"

    mod = importlib.import_module(FAMILY_MODULE[family])

    with mlflow.start_run(run_name=run_name):
        log_training_config(cfg, dataset_key, f"{family}{variant}")
        try:
            results = mod.train(dataset, cfg["training"], variant, run_name)
            if hasattr(results, "results_dict"):
                log_metrics(results.results_dict)
            best = Path(f"runs/{run_name}/weights/best.pt")
            if best.exists():
                log_model_artifacts(best, config_path)
            print(f"✅ Treino ok: {run_name}")
            return results
        except Exception as e:
            print(f"❌ Erro no treino: {e}")
            mlflow.log_param("error", str(e))
            return None
