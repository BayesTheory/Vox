# main_uni.py - VERSÃO PRODUÇÃO COM TREINAMENTO DESATIVADO

import warnings
import argparse
import sys
import os
import logging
import time
from pathlib import Path
from typing import Optional
import json

# ✅ Silencia TUDO
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["ONNX_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore")

def force_delete_onnx(weights_path: str) -> None:
    """✅ APAGA ONNX SEMPRE"""
    onnx_path = Path(weights_path).with_suffix(".onnx")
    if onnx_path.exists():
        print(f"🗑️ Apagando ONNX existente: {onnx_path.name}")
        onnx_path.unlink()

def find_config_file():
    """Encontra arquivo de configuração disponível"""
    possible_configs = [
        "config.json",
        "src/config.json", 
        Path.cwd() / "config.json",
        Path.cwd() / "src" / "config.json"
    ]
    
    for config_path in possible_configs:
        if Path(config_path).exists():
            print(f"✅ Usando config: {config_path}")
            return str(config_path)
    
    print("⚠️ Config não encontrado, criando config padrão...")
    return create_default_config()

def create_default_config(config_path="config.json"):
    """Cria configuração padrão baseada no hardware"""
    
    # Detectar hardware disponível
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name()
            print(f"🎮 GPU detectada: {gpu_name}")
        else:
            print("💻 Usando CPU")
    except ImportError:
        has_cuda = False
        print("💻 PyTorch não encontrado, assumindo CPU")
    
    try:
        import psutil
        cpu_cores = psutil.cpu_count(logical=False)
        optimal_threads = min(cpu_cores, 12)
    except ImportError:
        optimal_threads = 8
    
    # ✅ USAR SEU CONFIG COMO BASE - FOCO EM PRODUÇÃO
    config = {
        "tracking": {
            "tracker": "bytetrack.yaml",
            "inference": {
                "device": "cpu",
                "half_precision": False
            },
            "detection": {
                "imgsz_cpu": 320,
                "conf_threshold": 0.45,
                "iou_threshold": 0.6,
                "max_det": 25
            },
            "classification_model": {
                "imgsz": 128,
                "batch_size": 10,
                "center_crop_margin": 0.15,
                "min_crop_size": 10
            },
            "classification": {
                "min_confidence": 0.35
            },
            "sampling": {
                "classify_every": 5
            },
            "performance": {
                "frame_stride": 2,
                "detection_interval": 3,
                "num_threads_cpu": optimal_threads,
                "use_onnx": False,
                "force_pytorch": True,
                "force_classification_pytorch": True,
                "enable_warmup": True,
                "warmup_iterations": 3
            },
            "visualization": {
                "draw_boxes": True,
                "draw_track_id": True,
                "draw_labels": True,
                "draw_confidence": True,
                "box_thickness": 2,
                "font_scale": 0.6,
                "font_thickness": 2
            },
            "output": {
                "save_video": True,
                "save_json": True,
                "save_csv": True,
                "video_codec": "mp4v",
                "timeline_mode": "duplicate"
            }
        },
        "api": {
            "server": {
                "host": "127.0.0.1",
                "port": 8000,
                "workers": 1,
                "reload": False,
                "debug": False
            },
            "upload": {
                "max_file_size": 500,
                "allowed_extensions": [".mp4", ".avi", ".mov", ".mkv"],
                "upload_dir": "uploads",
                "cleanup_after_hours": 24
            },
            "processing": {
                "queue_size": 10,
                "concurrent_jobs": 2,
                "timeout_minutes": 30
            },
            "security": {
                "rate_limit": 10,
                "rate_window_minutes": 1
            }
        },
        "system": {
            "production_mode": True,
            "training_disabled": True,
            "cicd_pipeline": "future_implementation"
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config de produção criado: {config_path}")
    return config_path

def validate_config(config_path):
    """Valida se o config tem as seções necessárias"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_sections = ["tracking"]
        missing = [section for section in required_sections if section not in config]
        
        if missing:
            print(f"⚠️ Seções ausentes no config: {missing}")
            return False
        
        print(f"✅ Config válido: {config_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Config JSON inválido: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro lendo config: {e}")
        return False

def cmd_track_direct(video_path, det_weights, cls_weights, output_dir=""):
    """Tracking direto otimizado - PRODUÇÃO"""
    try:
        config_path = find_config_file()
        
        if not validate_config(config_path):
            print("❌ Config inválido. Abortando...")
            return None
        
        # Limpeza ONNX
        print("🔄 Limpando arquivos ONNX...")
        force_delete_onnx(det_weights)
        force_delete_onnx(cls_weights)
        
        # Validação de arquivos
        validations = [
            (video_path, "Vídeo", "📹"),
            (det_weights, "Detector", "🤖"),
            (cls_weights, "Classificador", "🎨")
        ]
        
        for file_path, name, icon in validations:
            if not Path(file_path).exists():
                print(f"❌ {name} não encontrado: {file_path}")
                return None
            else:
                size_mb = Path(file_path).stat().st_size / (1024**2)
                print(f"✅ {icon} {name}: {file_path} ({size_mb:.1f}MB)")
        
        print(f"⚙️ Config: {config_path}")
        
        # Diagnóstico
        print("\n🔍 === DIAGNÓSTICO PRE-PROCESSAMENTO ===")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("💻 Dispositivo: CPU")
                
            print(f"🔥 PyTorch: {torch.__version__}")
        except ImportError:
            print("⚠️ PyTorch não encontrado")
        
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            ram_total = psutil.virtual_memory().total / (1024**3)
            print(f"💻 CPU: {cpu_count} cores @ {cpu_freq.current:.0f}MHz")
            print(f"💾 RAM: {ram_total:.1f}GB")
        except ImportError:
            print("⚠️ psutil não encontrado")
        
        try:
            from ultralytics import YOLO
            print("🤖 Ultralytics: ✅ Disponível")
        except ImportError:
            print("❌ Ultralytics não encontrado! pip install ultralytics")
            return None
        
        # Executa tracking
        print("\n🚀 Iniciando tracking otimizado...")
        from src.tracking.track import process_video_tracking
        
        start_total = time.time()
        
        result = process_video_tracking(
            video_path=video_path,
            det_weights=det_weights,
            cls_weights=cls_weights,
            config_path=config_path,
            out_dir=output_dir if output_dir else None,
        )
        
        total_time = time.time() - start_total
        
        if result:
            print(f"\n🎉 === TRACKING CONCLUÍDO ===")
            print(f"⏱️ Tempo total: {total_time:.1f}s")
            print(f"📊 Tracks detectados: {result.get('total_tracks', 0)}")
            
            # Performance metrics
            if 'performance' in result:
                perf = result['performance']
                print(f"⚡ FPS médio: {perf.get('average_fps', 0):.1f}")
                print(f"🚀 FPS equivalente: {perf.get('fps_raw_equiv', 0):.1f}")
            
            # Distribuição de cores
            if 'color_distribution' in result and result['color_distribution']:
                print("🎨 Distribuição de cores:")
                total_vehicles = sum(result['color_distribution'].values())
                for color, count in result['color_distribution'].items():
                    percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
                    print(f"   {color}: {count} veículos ({percentage:.1f}%)")
            
            # Arquivos gerados
            if 'output_files' in result and result['output_files']:
                print("📁 Arquivos gerados:")
                for file_type, file_path in result['output_files'].items():
                    if file_path and Path(file_path).exists():
                        size_mb = Path(file_path).stat().st_size / (1024**2)
                        print(f"   {file_type}: {Path(file_path).name} ({size_mb:.1f}MB)")
        else:
            print("\n❌ Processamento falhou.")
        
        return result
        
    except Exception as e:
        print(f"❌ Erro no tracking: {e}")
        import traceback
        traceback.print_exc()
        return None

def cmd_api_integrated(host="127.0.0.1", port=8000, reload=False):
    """✅ API INTEGRADA COM MAIN_API.PY COMPLETA"""
    try:
        print("🚀 === INICIANDO API INTEGRADA ===")
        
        # ✅ VERIFICAR SE MAIN_API.PY EXISTE
        api_paths = [
            "src/api/main_api.py",
            "main_api.py",
            "src/main_api.py"
        ]
        
        api_file = None
        for api_path in api_paths:
            if Path(api_path).exists():
                api_file = api_path
                print(f"✅ API encontrada: {api_path}")
                break
        
        if not api_file:
            print("⚠️ main_api.py não encontrado, criando API básica...")
            return cmd_api_basic(host, port)
        
        # ✅ TENTAR IMPORTAR E EXECUTAR API COMPLETA
        try:
            import uvicorn
            print("✅ Uvicorn disponível")
            
            # Verificar dependências da API
            try:
                import fastapi
                from fastapi import FastAPI, UploadFile, File, HTTPException
                from fastapi.responses import JSONResponse, FileResponse
                print("✅ FastAPI disponível")
            except ImportError as e:
                print(f"❌ FastAPI não disponível: {e}")
                print("💡 Instale: pip install fastapi uvicorn python-multipart")
                return None
            
            # ✅ IMPORTAR E EXECUTAR API COMPLETA
            if api_file == "src/api/main_api.py":
                from src.api.main_api import app
                print("✅ API completa carregada de src/api/main_api.py")
            elif api_file == "main_api.py":
                from main_api import app
                print("✅ API completa carregada de main_api.py")
            elif api_file == "src/main_api.py":
                from src.main_api import app
                print("✅ API completa carregada de src/main_api.py")
            
            print(f"🌐 Servidor: http://{host}:{port}")
            print(f"📖 Documentação: http://{host}:{port}/docs")
            print(f"🔧 Swagger UI: http://{host}:{port}/redoc")
            print("💡 Pressione Ctrl+C para parar")
            
            if reload:
                print("🔄 Modo desenvolvimento (auto-reload)")
                if api_file == "src/api/main_api.py":
                    uvicorn.run("src.api.main_api:app", host=host, port=port, reload=True, workers=1)
                elif api_file == "main_api.py":
                    uvicorn.run("main_api:app", host=host, port=port, reload=True, workers=1)
                elif api_file == "src/main_api.py":
                    uvicorn.run("src.main_api:app", host=host, port=port, reload=True, workers=1)
            else:
                uvicorn.run(app, host=host, port=port, reload=False, workers=1)
            
            return {"success": True}
            
        except ImportError as e:
            print(f"❌ Erro importando API: {e}")
            print("📋 Falling back to basic API...")
            return cmd_api_basic(host, port)
            
    except KeyboardInterrupt:
        print("\n👋 API interrompida pelo usuário")
        return {"success": True, "message": "Interrompido"}
    except Exception as e:
        print(f"❌ Erro na API integrada: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def cmd_api_basic(host="127.0.0.1", port=8000):
    """API básica como fallback"""
    try:
        import uvicorn
        from fastapi import FastAPI, UploadFile, File, HTTPException
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        
        print("🔧 Criando API básica (fallback)...")
        
        app = FastAPI(
            title="Vox Color Detection API - Production",
            description="API de produção para detecção e classificação de cores de veículos",
            version="2.0.0"
        )
        
        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        def root():
            return {
                "message": "Vox API - Production Mode",
                "status": "ok",
                "version": "2.0.0",
                "mode": "production",
                "training": "disabled"
            }
        
        @app.get("/health")
        def health():
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    gpu_name = torch.cuda.get_device_name()
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_info = f"{gpu_name} ({gpu_memory:.1f}GB)"
                else:
                    gpu_info = "N/A"
            except:
                cuda_available = False
                gpu_info = "N/A"
            
            return {
                "status": "healthy",
                "mode": "production",
                "cuda_available": cuda_available,
                "gpu_info": gpu_info,
                "config_exists": Path("config.json").exists(),
                "tracking_available": Path("src/tracking/track.py").exists(),
                "training_disabled": True
            }
        
        @app.post("/upload")
        async def upload_video(file: UploadFile = File(...)):
            if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                raise HTTPException(status_code=400, detail="Formato de vídeo não suportado")
            
            return {
                "message": "Upload recebido (API de produção)",
                "filename": file.filename,
                "size": file.size,
                "note": "Para processamento completo, use a API principal"
            }
        
        @app.post("/track")
        async def track_basic(file: UploadFile = File(...)):
            return {
                "message": "Endpoint básico - use a API principal para processamento completo",
                "filename": file.filename,
                "status": "received"
            }
        
        print(f"🌐 API de produção: http://{host}:{port}")
        print(f"📖 Documentação: http://{host}:{port}/docs")
        print("💡 Pressione Ctrl+C para parar")
        
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except ImportError as e:
        print(f"❌ Dependências da API não encontradas: {e}")
        print("💡 Instale: pip install fastapi uvicorn python-multipart")
    except Exception as e:
        print(f"❌ Erro na API básica: {e}")

# 🚫 ============================================================================
# TREINAMENTO DESATIVADO - IMPLEMENTAÇÃO FUTURA VIA CI/CD PIPELINE
# ============================================================================

def cmd_train_detect_disabled(config_path, variant="n"):
    """🚫 TREINAMENTO DE DETECÇÃO - DESATIVADO"""
    print("\n" + "="*70)
    print("🚫 FUNCIONALIDADE DE TREINAMENTO TEMPORARIAMENTE DESATIVADA")
    print("="*70)
    print("📋 Status: Desabilitado para versão de produção")
    print("🔄 Implementação: Planejada para pipeline CI/CD")
    print("🎯 Objetivo: Automatização via containers e cloud")
    print("📅 Previsão: Próxima versão (v3.0)")
    print("")
    print("💡 PRÓXIMOS PASSOS:")
    print("   1. Configuração de ambiente de treinamento isolado")
    print("   2. Pipeline automatizado com MLflow")
    print("   3. Deployment automático de modelos")
    print("   4. Monitoramento de performance")
    print("   5. Versionamento automático de modelos")
    print("")
    print("🔧 Para ativar temporariamente:")
    print("   - Modifique a flag 'training_disabled' no config")
    print("   - Ou implemente pipeline CI/CD personalizado")
    print("="*70)
    
    return {
        "success": False, 
        "status": "disabled",
        "message": "Treinamento desativado - implementação futura via CI/CD",
        "next_steps": [
            "Setup CI/CD pipeline",
            "Container training environment",
            "MLflow integration",
            "Automated model deployment"
        ]
    }

def cmd_train_classify_disabled(config_path, variant="n", data_path=""):
    """🚫 TREINAMENTO DE CLASSIFICAÇÃO - DESATIVADO"""
    print("\n" + "="*70)
    print("🚫 FUNCIONALIDADE DE TREINAMENTO TEMPORARIAMENTE DESATIVADA")
    print("="*70)
    print("📋 Status: Desabilitado para versão de produção")
    print("🔄 Implementação: Planejada para pipeline CI/CD")
    print("🎯 Objetivo: Automatização via containers e cloud")
    print("📅 Previsão: Próxima versão (v3.0)")
    print("")
    print("💡 BENEFÍCIOS DO PIPELINE CI/CD:")
    print("   ✅ Treinamento automatizado em cloud")
    print("   ✅ Versionamento automático de modelos")
    print("   ✅ Testes automatizados de qualidade")
    print("   ✅ Deploy sem interrupção de serviço")
    print("   ✅ Rollback automático se performance cair")
    print("   ✅ Monitoramento contínuo de drift")
    print("")
    print("🏗️ ARQUITETURA PLANEJADA:")
    print("   🐳 Docker containers para treinamento")
    print("   ☁️ Cloud GPU instances sob demanda")
    print("   📊 MLflow para tracking de experimentos")
    print("   🚀 Kubernetes para orquestração")
    print("   📈 Prometheus para monitoramento")
    print("="*70)
    
    return {
        "success": False,
        "status": "disabled", 
        "message": "Treinamento desativado - implementação futura via CI/CD",
        "planned_features": [
            "Automated cloud training",
            "Model versioning",
            "Quality gates",
            "Zero-downtime deployment",
            "Performance monitoring"
        ]
    }

def prompt_choice(title, options):
    """Menu de escolha otimizado"""
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f" {i}) {opt}")
    
    while True:
        try:
            choice = input("Escolha [número]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return int(choice) - 1
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Saindo...")
            sys.exit(0)
        except:
            pass
        print("❌ Inválido. Tente novamente.")

def prompt_text(msg, default=None):
    """Prompt de texto otimizado"""
    prompt = f"{msg}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    
    try:
        result = input(prompt).strip()
        return result if result else (default or "")
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Saindo...")
        sys.exit(0)

def show_system_info():
    """Mostra informações completas do sistema"""
    print("\n💻 === INFORMAÇÕES DO SISTEMA - MODO PRODUÇÃO ===")
    
    # Python
    print(f"🐍 Python: {sys.version.split()[0]} ({sys.platform})")
    print(f"📂 Diretório atual: {Path.cwd()}")
    
    # PyTorch
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🎮 CUDA disponível: {'Sim' if torch.cuda.is_available() else 'Não'}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    except ImportError:
        print("🔥 PyTorch: ❌ Não instalado")
    
    # Ultralytics
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"🤖 Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("🤖 Ultralytics: ❌ Não instalado")
    
    # OpenCV
    try:
        import cv2
        print(f"📷 OpenCV: {cv2.__version__}")
    except ImportError:
        print("📷 OpenCV: ❌ Não instalado")
    
    # FastAPI
    try:
        import fastapi
        print(f"🚀 FastAPI: {fastapi.__version__}")
    except ImportError:
        print("🚀 FastAPI: ❌ Não instalado")
    
    # Sistema
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        ram = psutil.virtual_memory()
        print(f"💻 CPU: {cpu_count} cores @ {cpu_freq.current:.0f}MHz")
        print(f"💾 RAM: {ram.total/(1024**3):.1f}GB total")
    except ImportError:
        print("💻 System info: psutil não disponível")
    
    # Config e arquivos
    config_path = find_config_file()
    print(f"⚙️ Config: {'✅' if Path(config_path).exists() else '❌'} {config_path}")
    
    api_files = ["src/api/main_api.py", "main_api.py", "src/main_api.py"]
    api_found = any(Path(f).exists() for f in api_files)
    print(f"🚀 API completa: {'✅' if api_found else '❌'}")
    
    track_file = Path("src/tracking/track.py")
    print(f"🎯 Track module: {'✅' if track_file.exists() else '❌'}")
    
    # Status de produção
    print(f"\n🏭 === STATUS DE PRODUÇÃO ===")
    print(f"🔧 Modo: Produção (Production Mode)")
    print(f"🚫 Treinamento: Desativado")
    print(f"📈 Tracking: Ativo")
    print(f"🌐 API: Ativa")
    print(f"🔄 CI/CD: Planejado para v3.0")

def interactive_main():
    """Interface interativa - MODO PRODUÇÃO"""
    print("\n🚗 === Vox - MODO PRODUÇÃO ===")
    print("🏭 Versão de produção com foco em inferência")
    print("🚫 Treinamento desativado - Pipeline CI/CD em desenvolvimento")
    
    config_path = find_config_file()
    
    while True:
        choice = prompt_choice(
            "🎯 O que deseja fazer?",
            [
                "🎥 Processar vídeo (Tracking + Classificação)",
                "🚀 Iniciar API (Integrada com main_api.py)",
                "🚫 Treinar modelo de detecção (DESATIVADO)",
                "🚫 Treinar modelo de classificação (DESATIVADO)",
                "ℹ️ Informações do sistema",
                "🧪 Testar dependências",
                "❌ Sair"
            ]
        )
        
        if choice == 0:  # Tracking
            print("\n📹 === PROCESSAMENTO DE VÍDEO ===")
            
            video = prompt_text("Caminho do vídeo", "input.mp4")
            det_weights = prompt_text("Detector (.pt)", "runs/yolo11n_detection_detect3/weights/best.pt")
            cls_weights = prompt_text("Classificador (.pt)", "runs/yolo11n_classification_colors_n3/weights/best.pt")
            output_dir = prompt_text("Diretório de saída (opcional)", "")
            
            print(f"\n🔄 Processando em modo produção...")
            result = cmd_track_direct(video, det_weights, cls_weights, output_dir)
            
            if result:
                input("\n✅ Processamento concluído! Pressione Enter para continuar...")
            else:
                input("\n❌ Erro no processamento. Pressione Enter para continuar...")
        
        elif choice == 1:  # API Integrada
            print("\n🚀 === API MODO PRODUÇÃO ===")
            
            host = prompt_text("Host", "127.0.0.1")
            port_str = prompt_text("Porta", "8000")
            reload_str = prompt_text("Auto-reload? (y/N)", "n")
            
            try:
                port = int(port_str)
                reload = reload_str.lower().startswith('y')
                
                print(f"\n🌐 Iniciando API de produção em http://{host}:{port}")
                cmd_api_integrated(host, port, reload)
            except ValueError:
                print("❌ Porta inválida")
                input("Pressione Enter para continuar...")
        
        elif choice == 2:  # Treinar detecção - DESATIVADO
            print("\n🚫 === TREINAMENTO DE DETECÇÃO - DESATIVADO ===")
            variant = prompt_text("Variante do modelo (n/s/m/l/x)", "n")
            cmd_train_detect_disabled(config_path, variant)
            input("\nPressione Enter para continuar...")
        
        elif choice == 3:  # Treinar classificação - DESATIVADO
            print("\n🚫 === TREINAMENTO DE CLASSIFICAÇÃO - DESATIVADO ===")
            variant = prompt_text("Variante do modelo (n/s/m/l/x)", "s")
            cmd_train_classify_disabled(config_path, variant)
            input("\nPressione Enter para continuar...")
        
        elif choice == 4:  # Info do sistema
            show_system_info()
            input("\nPressione Enter para continuar...")
        
        elif choice == 5:  # Teste de dependências
            print("\n🧪 === TESTANDO DEPENDÊNCIAS - MODO PRODUÇÃO ===")
            
            # Dependências essenciais para produção
            production_dependencies = [
                ("torch", "PyTorch"),
                ("ultralytics", "Ultralytics YOLO"),
                ("cv2", "OpenCV"),
                ("PIL", "Pillow"),
                ("fastapi", "FastAPI"),
                ("uvicorn", "Uvicorn"),
                ("psutil", "psutil")
            ]
            
            all_ok = True
            for module_name, display_name in production_dependencies:
                try:
                    __import__(module_name)
                    print(f"✅ {display_name}: OK")
                except ImportError:
                    print(f"❌ {display_name}: FALTANDO")
                    all_ok = False
            
            print("\n🔍 Dependências opcionais para treinamento:")
            training_dependencies = [
                ("mlflow", "MLflow (CI/CD)"),
                ("wandb", "Weights & Biases (CI/CD)"),
                ("tensorboard", "TensorBoard (CI/CD)")
            ]
            
            for module_name, display_name in training_dependencies:
                try:
                    __import__(module_name)
                    print(f"🔄 {display_name}: Disponível (não usado em produção)")
                except ImportError:
                    print(f"🚫 {display_name}: Não instalado (será usado em CI/CD)")
            
            if all_ok:
                print("\n🎉 Todas as dependências de produção estão instaladas!")
                print("🏭 Sistema pronto para uso em produção!")
            else:
                print("\n⚠️ Instale dependências de produção:")
                print("pip install torch ultralytics opencv-python pillow fastapi uvicorn psutil python-multipart")
            
            input("\nPressione Enter para continuar...")
        
        else:  # Sair
            print("\n👋 Saindo do modo produção... Até mais!")
            break

def build_parser():
    """Parser com comandos de produção"""
    parser = argparse.ArgumentParser(
        description="Vox - Sistema de Produção v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🏭 MODO PRODUÇÃO - Exemplos de uso:
  python main_uni.py                                    # Modo interativo
  python main_uni.py track --video input.mp4 --det-weights detector.pt --cls-weights classifier.pt
  python main_uni.py api --host 0.0.0.0 --port 8000
  
🚫 TREINAMENTO DESATIVADO:
  - Funcionalidades de treinamento estão desabilitadas
  - Implementação planejada via pipeline CI/CD
  - Para desenvolvimento, use versão separada

💡 Para ativar treinamento:
  - Configure pipeline CI/CD personalizado
  - Ou modifique flags no config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=False)
    
    # Track command (ATIVO)
    track_parser = subparsers.add_parser("track", help="Processar vídeo (PRODUÇÃO)")
    track_parser.add_argument("--video", required=True, help="Caminho do vídeo")
    track_parser.add_argument("--det-weights", required=True, help="Pesos do detector (.pt)")
    track_parser.add_argument("--cls-weights", required=True, help="Pesos do classificador (.pt)")
    track_parser.add_argument("--output", default="", help="Diretório de saída")
    
    # API command (ATIVO)
    api_parser = subparsers.add_parser("api", help="Iniciar API (PRODUÇÃO)")
    api_parser.add_argument("--host", default="127.0.0.1", help="Host da API")
    api_parser.add_argument("--port", type=int, default=8000, help="Porta da API")
    api_parser.add_argument("--reload", action="store_true", help="Auto-reload")
    
    # Train commands (DESATIVADOS)
    train_detect_parser = subparsers.add_parser("train-detect", help="Treinar detecção (DESATIVADO)")
    train_detect_parser.add_argument("--variant", default="n", choices=['n', 's', 'm', 'l', 'x'], help="Variante do modelo")
    
    train_classify_parser = subparsers.add_parser("train-classify", help="Treinar classificação (DESATIVADO)")
    train_classify_parser.add_argument("--variant", default="s", choices=['n', 's', 'm', 'l', 'x'], help="Variante do modelo")
    
    # Other commands
    subparsers.add_parser("interactive", help="Modo interativo (PRODUÇÃO)")
    subparsers.add_parser("info", help="Informações do sistema")
    
    return parser

def main():
    """Função principal - MODO PRODUÇÃO"""
    print("🏭 Vox - MODO PRODUÇÃO")
    print("🚫 Treinamento desativado | 🎯 Foco em inferência")
    
    parser = build_parser()
    args = parser.parse_args()
    
    # Se nenhum comando, inicia interativo
    if args.command is None:
        print("ℹ️ Iniciando modo interativo de produção...")
        interactive_main()
        return
    
    # Comandos CLI
    config_path = find_config_file()
    
    if args.command == "track":
        result = cmd_track_direct(args.video, args.det_weights, args.cls_weights, args.output)
        return result
    
    elif args.command == "api":
        cmd_api_integrated(args.host, args.port, args.reload)
        return
    
    elif args.command == "train-detect":
        result = cmd_train_detect_disabled(config_path, args.variant)
        return result
    
    elif args.command == "train-classify":
        result = cmd_train_classify_disabled(config_path, args.variant)
        return result
    
    elif args.command == "interactive":
        interactive_main()
        return
    
    elif args.command == "info":
        show_system_info()
        return
    
    else:
        print(f"❌ Comando desconhecido: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário!")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()