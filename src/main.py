#!/usr/bin/env python3
"""
Sistema de Treinamento YOLO
- Detecção de veículos (UA-DETRAC)
- Classificação de cores (VeRi + VCoR)
"""

import argparse
from pathlib import Path
from src.train import train_model, train_all_available
from src.utils import load_config, validate_dataset_path

def check_datasets():
    """Verifica status dos datasets processados"""
    config = load_config()
    
    print("🔍 Verificando datasets processados...")
    print("=" * 50)
    
    all_ready = True
    
    for key, dataset_config in config["datasets"].items():
        path = dataset_config["path"]
        yaml_file = dataset_config["yaml"]
        task = dataset_config["task"]
        
        is_valid, message = validate_dataset_path(path, yaml_file)
        
        status = "✅ PRONTO" if is_valid else "❌ FALTANDO"
        print(f"{key.upper()} ({task}): {status}")
        print(f"  Caminho: {path}")
        print(f"  YAML: {yaml_file}")
        
        if not is_valid:
            print(f"  Erro: {message}")
            all_ready = False
        
        print()
    
    if all_ready:
        print("🎯 Todos os datasets estão prontos para treinamento!")
    else:
        print("⚠️ Execute os scripts de pré-processamento primeiro")
    
    return all_ready

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Treinamento YOLO para Detecção + Classificação"
    )
    
    parser.add_argument(
        "--check", action="store_true",
        help="Verificar status dos datasets"
    )
    
    parser.add_argument(
        "--train", choices=["detection", "classification", "all"],
        help="Treinar modelo específico ou todos"
    )
    
    parser.add_argument(
        "--model", default="yolo11n",
        help="Modelo a usar (yolo11n, yolo11s, etc.)"
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_datasets()
        return
    
    if args.train:
        if args.train == "all":
            print("🚀 Iniciando treinamento de todos os modelos disponíveis...")
            results = train_all_available()
            
            print("\n📊 Resumo dos treinamentos:")
            for name, result in results.items():
                status = "✅" if result else "❌"
                print(f"  {status} {name}")
        
        elif args.train == "detection":
            print(f"🚗 Treinando detecção com {args.model}...")
            train_model("detection", args.model)
        
        elif args.train == "classification":
            print(f"🎨 Treinando classificação com {args.model}...")
            train_model("classification", args.model)
    
    else:
        # Modo interativo básico
        print("🎯 Sistema de Treinamento YOLO")
        print("=" * 40)
        
        if check_datasets():
            print("\n🚀 Datasets prontos! Use:")
            print("  python main.py --train all")
            print("  python main.py --train detection --model yolo11n")
            print("  python main.py --train classification --model yolo11n")
        else:
            print("\n⚠️ Execute primeiro os pré-processamentos necessários")

if __name__ == "__main__":
    main()
