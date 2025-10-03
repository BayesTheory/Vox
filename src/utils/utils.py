# src/utils/utils.py
import json
import os
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from contextlib import contextmanager
from collections import defaultdict

# Imports essenciais para reproducibilidade
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# MLflow imports
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Extensões de imagem suportadas
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def seed_everything(seed: int = 42) -> None:
    """
    Fixa todas as fontes de aleatoriedade para reprodutibilidade completa.
    Baseado nas melhores práticas do PyTorch e OpenCV.
    """
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch (se disponível)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        
        # CuDNN settings para reprodutibilidade
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Evita nondeterminism em alguns ops do PyTorch
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.info(f"🎲 Seeds fixadas: {seed}")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carrega arquivo de configuração JSON com validação flexível.
    
    ATUALIZADO: Suporta estrutura hierárquica (project.model_family).
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config não encontrado: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validação flexível: aceita model_family no raiz OU em project.model_family
        if "model_family" not in config:
            # Tenta buscar em project.model_family
            if "project" in config and "model_family" in config["project"]:
                # Move para o nível raiz para compatibilidade com código legado
                config["model_family"] = config["project"]["model_family"]
                logger.info("✅ model_family encontrado em project.model_family")
            else:
                logger.warning("⚠️  model_family não encontrado no config, usando 'yolo11' como padrão")
                config["model_family"] = "yolo11"
        
        # Validação de estruturas essenciais (com defaults)
        if "datasets" not in config:
            logger.warning("⚠️  Seção 'datasets' ausente no config, criando vazia")
            config["datasets"] = {}
        
        if "training" not in config:
            logger.warning("⚠️  Seção 'training' ausente no config, usando defaults")
            config["training"] = {"seed": 42}
        
        if "mlflow" not in config:
            logger.warning("⚠️  Seção 'mlflow' ausente no config, usando defaults")
            config["mlflow"] = {
                "tracking_uri": "./mlruns",
                "experiment_name": "YOLO_Training"
            }
        
        logger.info(f"✅ Config carregado: {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido em {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar config {config_path}: {e}")

def validate_dataset_path(dataset_root: str, yaml_file: str) -> Tuple[bool, str]:
    """
    Valida estrutura de dataset YOLO (detecção) ou classificação.
    Compatível com paths absolutos e relativos.
    """
    root_path = Path(dataset_root).resolve()
    yaml_path = root_path / yaml_file
    
    # Verifica existência básica
    if not root_path.exists():
        return False, f"Diretório do dataset não existe: {root_path}"
    
    if not yaml_path.exists():
        return False, f"YAML não encontrado: {yaml_path}"
    
    try:
        # Tenta carregar e validar YAML
        import yaml as pyyaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = pyyaml.safe_load(f)
        
        if not isinstance(data, dict):
            return False, f"YAML inválido (não é dict): {yaml_path}"
        
        # Detecção de tipo: detect vs classify
        has_train_val = 'train' in data and 'val' in data
        has_nc_names = 'nc' in data and 'names' in data
        
        if has_train_val and has_nc_names:
            # Formato YOLO detect
            train_path = root_path / data['train'] if not Path(data['train']).is_absolute() else Path(data['train'])
            val_path = root_path / data['val'] if not Path(data['val']).is_absolute() else Path(data['val'])
            
            if not train_path.exists():
                return False, f"Diretório de treino não encontrado: {train_path}"
            if not val_path.exists():
                return False, f"Diretório de validação não encontrado: {val_path}"
            
            # Conta imagens em train e val (amostragem)
            train_images = list(train_path.glob('**/*'))
            train_images = [p for p in train_images if p.suffix.lower() in IMAGE_EXTENSIONS]
            val_images = list(val_path.glob('**/*'))
            val_images = [p for p in val_images if p.suffix.lower() in IMAGE_EXTENSIONS]
            
            if len(train_images) == 0:
                return False, f"Nenhuma imagem encontrada em {train_path}"
            if len(val_images) == 0:
                return False, f"Nenhuma imagem encontrada em {val_path}"
            
            return True, f"Dataset DETECT OK: {root_path} | YAML: {yaml_path} | Train: {len(train_images)} | Val: {len(val_images)}"
        
        elif (root_path / "train").exists() and (root_path / "val").exists():
            # Formato classify (pastas por classe)
            train_dir = root_path / "train"
            val_dir = root_path / "val"
            
            # Conta classes e imagens
            train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
            val_classes = [d for d in val_dir.iterdir() if d.is_dir()]
            
            if len(train_classes) == 0:
                return False, f"Nenhuma classe encontrada em {train_dir}"
            if len(val_classes) == 0:
                return False, f"Nenhuma classe encontrada em {val_dir}"
            
            # Conta imagens por classe (amostragem)
            train_total = sum(len([p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]) 
                            for cls_dir in train_classes)
            val_total = sum(len([p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]) 
                          for cls_dir in val_classes)
            
            if train_total == 0:
                return False, f"Nenhuma imagem encontrada nas classes de treino"
            if val_total == 0:
                return False, f"Nenhuma imagem encontrada nas classes de validação"
            
            return True, f"Dataset CLASSIFY OK: {root_path} | Classes: {len(train_classes)} | Train: {train_total} | Val: {val_total}"
        
        else:
            return False, f"Formato de dataset não reconhecido em {yaml_path}"
            
    except ImportError:
        logger.warning("PyYAML não disponível, pulando validação detalhada do YAML")
        return True, f"Dataset path existe: {root_path} | YAML: {yaml_path} (validação limitada)"
    except Exception as e:
        return False, f"Erro na validação: {e}"

def create_balanced_dataset(
    source_root: Union[str, Path], 
    mode: str = "oversample", 
    ratio: float = 0.6,
    suffix: str = "_balanced"
) -> Path:
    """
    Cria dataset balanceado para classificação com cleanup automático.
    Preparado para CI/CD com logging detalhado.
    """
    source_path = Path(source_root).resolve()
    out_root = source_path.parent / f"{source_path.name}{suffix}"
    
    logger.info(f"🔄 Criando dataset balanceado: {source_path} → {out_root}")
    
    # Cleanup de versões anteriores
    if out_root.exists():
        logger.info(f"🗑️ Removendo dataset balanceado anterior: {out_root}")
        shutil.rmtree(out_root)
    
    train_src = source_path / "train"
    if not train_src.exists():
        raise RuntimeError(f"Diretório train/ não encontrado em {source_path}")
    
    # Analisa distribuição atual
    class_counts = {}
    class_images = {}
    
    for cls_dir in train_src.iterdir():
        if not cls_dir.is_dir():
            continue
        
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        class_counts[cls_dir.name] = len(images)
        class_images[cls_dir.name] = images
    
    if not class_counts:
        raise RuntimeError(f"Nenhuma classe encontrada em {train_src}")
    
    # Calcula targets
    max_count = max(class_counts.values())
    target_count = max(1, int(max_count * ratio))
    
    logger.info(f"📊 Distribuição original: {dict(class_counts)}")
    logger.info(f"🎯 Target count: {target_count} (ratio={ratio}, max={max_count})")
    
    # Cria estrutura balanceada
    out_train = out_root / "train"
    out_train.mkdir(parents=True, exist_ok=True)
    
    balanced_counts = {}
    
    for cls_name, current_count in class_counts.items():
        src_cls = train_src / cls_name
        dst_cls = out_train / cls_name
        dst_cls.mkdir(exist_ok=True)
        
        images = class_images[cls_name]
        
        # Copia todas as imagens originais
        for img in images:
            try:
                shutil.copy2(img, dst_cls / img.name)
            except Exception as e:
                logger.warning(f"Erro copiando {img}: {e}")
                continue
        
        final_count = current_count
        
        # Aplicar balanceamento conforme modo
        if mode == "oversample" and current_count < target_count:
            needed = target_count - current_count
            logger.info(f"  📈 {cls_name}: {current_count} → {target_count} (+{needed})")
            
            for i in range(needed):
                src_img = images[i % len(images)]
                dst_name = f"{src_img.stem}_dup{i:04d}{src_img.suffix}"
                try:
                    shutil.copy2(src_img, dst_cls / dst_name)
                except Exception as e:
                    logger.warning(f"Erro duplicando {src_img}: {e}")
                    continue
            
            final_count = target_count
        
        elif mode == "undersample" and current_count > target_count:
            # Remove imagens extras (mantém as primeiras target_count)
            to_remove = list(dst_cls.iterdir())[target_count:]
            for img in to_remove:
                img.unlink()
            
            final_count = target_count
            logger.info(f"  📉 {cls_name}: {current_count} → {target_count} (-{current_count - target_count})")
        
        else:
            logger.info(f"  ⚪ {cls_name}: {current_count} (sem alteração)")
        
        balanced_counts[cls_name] = final_count
    
    # Copia val e test sem modificar
    for split in ["val", "test"]:
        src_split = source_path / split
        if src_split.exists():
            dst_split = out_root / split
            logger.info(f"📁 Copiando {split}/ ...")
            try:
                shutil.copytree(src_split, dst_split)
            except Exception as e:
                logger.warning(f"Erro copiando {split}/: {e}")
    
    # Copia YAML se existir
    for yaml_file in ["data.yaml", "dataset.yaml"]:
        yaml_src = source_path / yaml_file
        if yaml_src.exists():
            shutil.copy2(yaml_src, out_root / yaml_file)
            break
    
    logger.info(f"✅ Dataset balanceado criado: {balanced_counts}")
    return out_root

# ================================
# MLflow Integration
# ================================

def setup_mlflow(config: Dict[str, Any]) -> None:
    """
    Configura MLflow com base na configuração fornecida.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow não disponível, pulando configuração de tracking")
        return
    
    mlflow_cfg = config.get("mlflow", {})
    
    # Configura tracking URI
    tracking_uri = mlflow_cfg.get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Configura experimento
    experiment_name = mlflow_cfg.get("experiment_name", "Default")
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        logger.info(f"🔬 MLflow configurado: {experiment_name} @ {tracking_uri}")
    except Exception as e:
        logger.warning(f"Erro configurando MLflow: {e}")

@contextmanager
def start_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Context manager para runs do MLflow com tratamento de erros.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow não disponível, executando sem tracking")
        yield None
        return
    
    try:
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            logger.info(f"🏃 MLflow run iniciado: {run.info.run_id}")
            yield run
    except Exception as e:
        logger.error(f"Erro no MLflow run: {e}")
        yield None

def log_training_config(config: Dict[str, Any], task: str, model_name: str) -> None:
    """
    Loga configuração de treinamento no MLflow.
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    
    try:
        # Parâmetros básicos
        mlflow.log_param("task", task)
        mlflow.log_param("model", model_name)
        mlflow.log_param("seed", config["training"].get("seed", 42))
        
        # Dataset info
        if task in config["datasets"]:
            dataset_cfg = config["datasets"][task]
            mlflow.log_param("dataset_name", dataset_cfg.get("name", "unknown"))
            mlflow.log_param("dataset_nc", dataset_cfg.get("nc", 0))
        
        # Training config
        training_cfg = config["training"]
        for key, value in training_cfg.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(f"training_{key}", value)
        
        # Model family
        mlflow.log_param("model_family", config.get("model_family", "unknown"))
        
        logger.info("📝 Config logada no MLflow")
        
    except Exception as e:
        logger.warning(f"Erro logando config no MLflow: {e}")

def log_metrics(metrics: Dict[str, Union[int, float]], prefix: str = "") -> None:
    """
    Loga métricas no MLflow com prefixo opcional.
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    
    try:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_name = f"{prefix}_{key}" if prefix else key
                mlflow.log_metric(metric_name, value)
        
        logger.info(f"📊 Métricas logadas: {len(metrics)} items")
        
    except Exception as e:
        logger.warning(f"Erro logando métricas no MLflow: {e}")

def log_artifacts(artifact_paths: List[Union[str, Path]], artifact_path: str = "") -> None:
    """
    Loga artefatos no MLflow (modelos, plots, etc).
    """
    if not MLFLOW_AVAILABLE or not mlflow.active_run():
        return
    
    logged_count = 0
    for path in artifact_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Artefato não encontrado: {path}")
            continue
        
        try:
            if path_obj.is_file():
                mlflow.log_artifact(str(path_obj), artifact_path)
                logged_count += 1
            elif path_obj.is_dir():
                mlflow.log_artifacts(str(path_obj), artifact_path)
                logged_count += 1
        except Exception as e:
            logger.warning(f"Erro logando artefato {path}: {e}")
    
    if logged_count > 0:
        logger.info(f"📦 Artefatos logados: {logged_count} items")

# ================================
# Utilidades Gerais
# ================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Garante que um diretório existe, criando se necessário.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def count_images_in_dir(directory: Union[str, Path]) -> int:
    """
    Conta imagens válidas em um diretório (recursivo).
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return 0
    
    count = 0
    for file_path in dir_path.rglob("*"):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
    
    return count

def get_dataset_stats(dataset_root: Union[str, Path], task: str = "auto") -> Dict[str, Any]:
    """
    Coleta estatísticas detalhadas de um dataset.
    """
    root_path = Path(dataset_root).resolve()
    stats = {
        "root": str(root_path),
        "task": task,
        "splits": {},
        "total_images": 0,
        "classes": []
    }
    
    # Auto-detecta task se necessário
    if task == "auto":
        if (root_path / "train").exists() and any((root_path / "train").iterdir()):
            # Verifica se train tem subpastas (classify) ou imagens+labels (detect)
            train_contents = list((root_path / "train").iterdir())
            if all(item.is_dir() for item in train_contents[:5]):  # Amostra
                stats["task"] = "classify"
            else:
                stats["task"] = "detect"
        else:
            stats["task"] = "unknown"
    
    # Coleta stats por split
    for split in ["train", "val", "test"]:
        split_dir = root_path / split
        if not split_dir.exists():
            continue
        
        split_stats = {
            "exists": True,
            "images": 0,
            "classes": {}
        }
        
        if stats["task"] == "classify":
            # Conta por classe
            for cls_dir in split_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                
                cls_images = count_images_in_dir(cls_dir)
                split_stats["classes"][cls_dir.name] = cls_images
                split_stats["images"] += cls_images
                
                if cls_dir.name not in stats["classes"]:
                    stats["classes"].append(cls_dir.name)
        
        else:
            # Conta total para detect
            split_stats["images"] = count_images_in_dir(split_dir)
        
        stats["splits"][split] = split_stats
        stats["total_images"] += split_stats["images"]
    
    return stats

def format_dataset_summary(stats: Dict[str, Any]) -> str:
    """
    Formata estatísticas de dataset para exibição.
    """
    lines = [
        f"📊 Dataset: {stats['root']}",
        f"🎯 Tipo: {stats['task']}",
        f"📸 Total de imagens: {stats['total_images']:,}"
    ]
    
    if stats["task"] == "classify" and stats["classes"]:
        lines.append(f"🏷️ Classes: {len(stats['classes'])} ({', '.join(stats['classes'][:5])}{'...' if len(stats['classes']) > 5 else ''})")
    
    for split_name, split_data in stats["splits"].items():
        if not split_data["exists"]:
            continue
        
        line = f"  📁 {split_name}: {split_data['images']:,} imagens"
        
        if stats["task"] == "classify" and split_data["classes"]:
            # Mostra distribuição de classes
            class_counts = split_data["classes"]
            if class_counts:
                avg_per_class = split_data["images"] / len(class_counts)
                min_count = min(class_counts.values())
                max_count = max(class_counts.values())
                line += f" | Classes: {len(class_counts)} | Min/Avg/Max: {min_count}/{avg_per_class:.1f}/{max_count}"
        
        lines.append(line)
    
    return "\n".join(lines)

# ================================
# Validação e Debug
# ================================

def validate_model_weights(weights_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Valida se um arquivo de pesos é válido.
    """
    path_obj = Path(weights_path)
    
    if not path_obj.exists():
        return False, f"Arquivo não encontrado: {weights_path}"
    
    if not path_obj.suffix == '.pt':
        return False, f"Extensão inválida: {path_obj.suffix} (esperado .pt)"
    
    # Verifica tamanho mínimo (evita arquivos corrompidos/vazios)
    size_mb = path_obj.stat().st_size / (1024 * 1024)
    if size_mb < 0.1:
        return False, f"Arquivo muito pequeno: {size_mb:.2f}MB (possível corrupção)"
    
    # Tenta carregar com torch se disponível
    if TORCH_AVAILABLE:
        try:
            checkpoint = torch.load(path_obj, map_location='cpu')
            if not isinstance(checkpoint, dict):
                return False, "Checkpoint não é um dict válido"
            
            # Verifica chaves essenciais
            if 'model' not in checkpoint:
                return False, "Chave 'model' não encontrada no checkpoint"
            
            return True, f"Pesos válidos: {weights_path} ({size_mb:.1f}MB)"
            
        except Exception as e:
            return False, f"Erro carregando checkpoint: {e}"
    
    else:
        # Validação básica sem torch
        return True, f"Arquivo existe: {weights_path} ({size_mb:.1f}MB) - validação limitada"

def debug_environment() -> Dict[str, Any]:
    """
    Coleta informações de debug do ambiente.
    """
    info = {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "platform": os.sys.platform,
        "cwd": str(Path.cwd()),
        "torch_available": TORCH_AVAILABLE,
        "mlflow_available": MLFLOW_AVAILABLE
    }
    
    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_current_device"] = torch.cuda.current_device()
    
    if MLFLOW_AVAILABLE:
        info["mlflow_version"] = mlflow.__version__
        try:
            info["mlflow_tracking_uri"] = mlflow.get_tracking_uri()
        except:
            info["mlflow_tracking_uri"] = "not_set"
    
    return info

# ================================
# Aliases e Retrocompatibilidade
# ================================

def validate_dataset_structure(*args, **kwargs):
    """Alias para validate_dataset_path (retrocompatibilidade)"""
    return validate_dataset_path(*args, **kwargs)

def set_seeds(seed: int = 42):
    """Alias para seed_everything (retrocompatibilidade)"""
    return seed_everything(seed)

# ================================
# Inicialização do Módulo
# ================================

# Log de inicialização
logger.info(f"🔧 Utils carregado - Torch: {TORCH_AVAILABLE} | MLflow: {MLFLOW_AVAILABLE}")

# Verificação de dependências críticas
if not TORCH_AVAILABLE:
    logger.warning("⚠️ PyTorch não disponível - funções de ML limitadas")

if not MLFLOW_AVAILABLE:
    logger.warning("⚠️ MLflow não disponível - sem tracking de experimentos")

# Configurações padrão para ambientes especiais
if os.getenv("CI") == "true":
    logger.info("🤖 Ambiente CI detectado - configurações otimizadas para automação")
    # Configurações específicas para CI/CD podem ser adicionadas aqui

# Versionamento do módulo utils
__version__ = "1.0.0"
__all__ = [
    'seed_everything', 'load_config', 'validate_dataset_path', 'create_balanced_dataset',
    'setup_mlflow', 'start_run', 'log_training_config', 'log_metrics', 'log_artifacts',
    'ensure_dir', 'count_images_in_dir', 'get_dataset_stats', 'format_dataset_summary',
    'validate_model_weights', 'debug_environment'
]

if __name__ == "__main__":
    # Teste básico quando executado diretamente
    print("🧪 Teste básico do utils.py")
    print(f"Versão: {__version__}")
    
    # Debug do ambiente
    env_info = debug_environment()
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    # Teste de seed
    print("\n🎲 Teste de reprodutibilidade:")
    seed_everything(42)
    print(f"  Random: {random.random():.6f}")
    print(f"  NumPy: {np.random.random():.6f}")
    
    if TORCH_AVAILABLE:
        print(f"  Torch: {torch.rand(1).item():.6f}")
    
    print("✅ Utils funcionando corretamente!")
