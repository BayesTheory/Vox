import importlib
import mlflow
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.utils import (
    setup_mlflow, log_training_config, log_metrics, log_artifacts,
    validate_dataset_path, load_config, seed_everything, select_device,
    start_run, timer, log_stage, save_manifest
)

# Opção A: renomeie arquivos para minúsculo: src/Modelos/yolo11.py e src/Modelos/yolov8.py
FAMILY_MODULE = {
    "yolov8": "src.Modelos.yolov8",
    "yolo11": "src.Modelos.yolo11",
}

# Opção B (se mantiver CamelCase): use este mapeamento
# FAMILY_MODULE = {
#     "yolov8": "src.Modelos.Yolov8",
#     "yolo11": "src.Modelos.Yolov11",
# }

def _resolve_task_key(task: str) -> str:
    return task if task in ("detect","classify") else ("detect" if task == "detection" else "classify")

def train_from_config(config_path: str = "src/config.json"):
    cfg = load_config(config_path)
    dataset_key: str = cfg["dataset_key"]
    dataset: Dict[str, Any] = cfg["datasets"][dataset_key]
    family: str = cfg["model_family"]
    variant: str = cfg["variant"]
    task: str = cfg["task"]
    task_key = _resolve_task_key(task)

    assert _resolve_task_key(dataset["task"]) == task_key, \
        f"Task do dataset ({dataset['task']}) difere da task global ({task})"

    ok, msg = validate_dataset_path(dataset["path"], dataset["yaml"])
    if not ok:
        print(f"❌ {msg}")
        return None
    print(f"✅ {msg}")

    seed = cfg.get("training", {}).get("seed", 42)
    seed_everything(seed)
    device = select_device(cfg["training"].get("device", "cpu"))

    setup_mlflow(cfg)
    run_name = f"{family}{variant}_{dataset_key}_{task_key}"

    save_manifest(dataset["path"], dataset["yaml"], Path("runs")/run_name/"dataset_manifest.json")

    try:
        mod = importlib.import_module(FAMILY_MODULE[family])
    except ModuleNotFoundError as e:
        print(f"❌ Não foi possível importar módulo da família '{family}': {e}")
        print("   Verifique o mapeamento FAMILY_MODULE e os nomes dos arquivos em src/Modelos/.")
        return None

    stages: Optional[List[Dict[str, Any]]] = (cfg.get("stages", {}) or {}).get(task_key, None)

    with start_run(run_name=run_name, tags={"family": family, "variant": variant, "task": task_key}):
        log_training_config(cfg, dataset_key, f"{family}{variant}")
        try:
            if stages:
                results = None
                for i, stg in enumerate(stages, start=1):
                    log_stage(i, stg)
                    with timer(f"stage_{i}"):
                        results = mod.train(dataset, cfg["training"], variant, run_name, stages=[stg])
                        if hasattr(results, "results_dict"):
                            log_metrics({f"stage_{i}_{k}": v for k, v in results.results_dict.items()})
            else:
                results = mod.train(dataset, cfg["training"], variant, run_name)

            if results is not None and hasattr(results, "results_dict"):
                log_metrics(results.results_dict)

            best = Path(f"runs/{run_name}/weights/best.pt")
            last = Path(f"runs/{run_name}/weights/last.pt")
            arts = [p for p in [best, last, Path(f"runs/{run_name}/dataset_manifest.json")] if p.exists()]
            if arts:
                log_artifacts(arts, artifact_path="artifacts")

            print(f"✅ Treino concluído: {run_name}")
            return results
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            mlflow.log_param("error", str(e))
            return None
