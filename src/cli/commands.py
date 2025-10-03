# src/cli/commands.py - COMANDOS ULTRA-OTIMIZADOS

import uvicorn
from pathlib import Path

def cmd_train_detect(args):
    """Comando de treinamento de detecção otimizado"""
    try:
        from src.train.train import train_detection_pipeline
        
        print("🤖 === INICIANDO TREINAMENTO DE DETECÇÃO ===")
        
        result = train_detection_pipeline(
            config_path=args.config,
            variant=args.variant,
            part1_weights=getattr(args, "part1_weights", None),
            part2_weights=getattr(args, "part2_weights", None),
            force_part1=getattr(args, "force_part1", False),
            force_part2=getattr(args, "force_part2", False),
        )
        
        if result.get("success"):
            print(f"🎉 Treinamento de detecção concluído!")
            print(f"📦 Pesos finais: {result.get('final_weights', 'N/A')}")
        else:
            print(f"❌ Treinamento falhou: {result.get('error', 'Erro desconhecido')}")
            
        return result
        
    except Exception as e:
        print(f"❌ Erro no comando train-detect: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def cmd_train_classify(args):
    """✅ COMANDO DE CLASSIFICAÇÃO CORRIGIDO E OTIMIZADO"""
    try:
        from src.train.train import train_classification_pipeline
        
        print("🎨 === INICIANDO TREINAMENTO DE CLASSIFICAÇÃO ===")
        print(f"   Dataset: {args.data}")
        print(f"   Variante: {args.variant}")
        
        # ✅ VALIDAÇÃO DO DATASET
        dataset_path = Path(args.data)
        if not dataset_path.exists():
            print(f"❌ Dataset não encontrado: {dataset_path}")
            return {"success": False, "error": "Dataset não encontrado"}
        
        # Verificar estrutura de classificação
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            print(f"❌ Estrutura inválida. Esperado: {dataset_path}/train/ e {dataset_path}/val/")
            return {"success": False, "error": "Estrutura de dataset inválida"}
        
        # Contar classes
        train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        val_classes = [d.name for d in val_dir.iterdir() if d.is_dir()]
        
        if not train_classes or not val_classes:
            print("❌ Nenhuma classe encontrada nas pastas train/ ou val/")
            return {"success": False, "error": "Nenhuma classe encontrada"}
        
        print(f"📊 Classes detectadas: {len(train_classes)} ({', '.join(train_classes[:5])}{'...' if len(train_classes) > 5 else ''})")
        
        # ✅ EXECUÇÃO OTIMIZADA
        result = train_classification_pipeline(
            config_path=args.config,
            variant=args.variant,
            base_weights=None,  # Usar pesos padrão do YOLO
        )
        
        if result.get("success"):
            print(f"🎉 Treinamento de classificação concluído!")
            print(f"📦 Pesos finais: {result.get('final_weights', 'N/A')}")
            
            # Mostrar métricas se disponíveis
            if "results" in result and hasattr(result["results"], "results_dict"):
                metrics = result["results"].results_dict
                if "metrics/accuracy_top1" in metrics:
                    print(f"🎯 Acurácia Top-1: {metrics['metrics/accuracy_top1']:.3f}")
                if "metrics/accuracy_top5" in metrics:
                    print(f"🎯 Acurácia Top-5: {metrics['metrics/accuracy_top5']:.3f}")
        else:
            print(f"❌ Treinamento falhou: {result.get('error', 'Erro desconhecido')}")
            
        return result
        
    except Exception as e:
        print(f"❌ Erro no comando train-classify: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def cmd_track(args):
    """Comando de tracking ultra-otimizado"""
    try:
        from src.tracking.track import process_video_tracking
        
        print("🎥 === INICIANDO TRACKING ULTRA-OTIMIZADO ===")
        
        # Validações
        video_path = Path(args.video)
        det_weights = Path(args.det_weights) 
        cls_weights = Path(args.cls_weights)
        
        validations = [
            (video_path, "Vídeo", "📹"),
            (det_weights, "Detector", "🤖"),  
            (cls_weights, "Classificador", "🎨")
        ]
        
        for file_path, name, icon in validations:
            if not file_path.exists():
                print(f"❌ {name} não encontrado: {file_path}")
                return {"success": False, "error": f"{name} não encontrado"}
            else:
                size_mb = file_path.stat().st_size / (1024**2)
                print(f"✅ {icon} {name}: {file_path.name} ({size_mb:.1f}MB)")
        
        # ✅ LIMPEZA ONNX AUTOMÁTICA
        def force_delete_onnx(weights_path):
            onnx_path = weights_path.with_suffix(".onnx")
            if onnx_path.exists():
                print(f"🗑️ Removendo ONNX: {onnx_path.name}")
                onnx_path.unlink()
        
        force_delete_onnx(det_weights)
        force_delete_onnx(cls_weights)
        
        result = process_video_tracking(
            video_path=str(video_path),
            det_weights=str(det_weights),
            cls_weights=str(cls_weights),
            config_path=args.config,
            out_dir=args.output if args.output else None,
        )
        
        if result:
            print("\n🎉 === TRACKING CONCLUÍDO ===")
            print(f"📊 Total de tracks: {result.get('total_tracks', 0)}")
            
            # Performance
            if 'performance' in result:
                perf = result['performance']
                print(f"⚡ FPS médio: {perf.get('average_fps', 0):.1f}")
                print(f"🚀 FPS equivalente: {perf.get('fps_raw_equiv', 0):.1f}")
                print(f"⏱️ Tempo total: {perf.get('total_time_formatted', 'N/A')}")
            
            # Cores detectadas
            if 'color_distribution' in result and result['color_distribution']:
                print("🎨 Distribuição de cores:")
                for color, count in result['color_distribution'].items():
                    print(f"   {color}: {count} veículos")
            
            # Arquivos gerados
            if 'output_files' in result:
                print("📁 Arquivos gerados:")
                for file_type, file_path in result['output_files'].items():
                    if file_path and Path(file_path).exists():
                        size_mb = Path(file_path).stat().st_size / (1024**2)
                        print(f"   {file_type}: {Path(file_path).name} ({size_mb:.1f}MB)")
        else:
            print("❌ Tracking falhou")
            
        return result
        
    except Exception as e:
        print(f"❌ Erro no tracking: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def cmd_api(args):
    """Comando de API ultra-otimizado"""
    try:
        print("🚀 === INICIANDO API ULTRA-OTIMIZADA ===")
        
        # Verificar dependências
        try:
            import fastapi
            import uvicorn
            print(f"✅ FastAPI {fastapi.__version__} disponível")
        except ImportError as e:
            print(f"❌ Dependências faltando: {e}")
            print("💡 Instale com: pip install fastapi uvicorn")
            return {"success": False, "error": "Dependências faltando"}
        
        # Importar aplicação
        try:
            from src.api.main_api import app
            print("✅ Aplicação carregada")
        except ImportError:
            print("⚠️ API customizada não encontrada, usando API básica")
            
            # Criar API básica
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
            
            app = FastAPI(
                title="Vehicle Color API - Básica",
                description="API básica para tracking de veículos",
                version="1.0.0"
            )
            
            @app.get("/")
            def root():
                return {"message": "API básica funcionando", "status": "ok"}
            
            @app.get("/health")  
            def health():
                return {"status": "healthy", "mode": "basic"}
        
        app_import = "src.api.main_api:app" if args.reload else None
        
        print(f"🌐 Servidor: http://{args.host}:{args.port}")
        print(f"📖 Docs: http://{args.host}:{args.port}/docs")
        print("💡 Pressione Ctrl+C para parar")
        
        if args.reload:
            print("🔄 Modo desenvolvimento (auto-reload)")
            uvicorn.run(app_import, host=args.host, port=args.port, reload=True, workers=1)
        else:
            uvicorn.run(app, host=args.host, port=args.port, reload=False, workers=1)
            
        return {"success": True}
        
    except KeyboardInterrupt:
        print("\n👋 API interrompida pelo usuário")
        return {"success": True, "message": "Interrompido"}
    except Exception as e:
        print(f"❌ Erro na API: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}