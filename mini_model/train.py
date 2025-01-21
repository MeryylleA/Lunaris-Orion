"""
Script principal para treinamento do modelo Mini 16x16.
"""

import os
import argparse
from pathlib import Path
import torch
from training.trainer import train_distributed, MultiGPUTrainer
from training.trainer import GPUManager

def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina modelo Mini 16x16 para geração de pixel art"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho para arquivo de configuração"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Diretório com dados de treinamento"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Diretório para salvar checkpoints e logs"
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Se deve usar treinamento distribuído"
    )
    
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Número de GPUs para treinamento distribuído. Se None, usa todas disponíveis"
    )
    
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="IDs específicos das GPUs a serem usadas (ex: '0,1,2')"
    )
    
    return parser.parse_args()

def main():
    # Parse argumentos
    args = parse_args()
    
    # Verificar GPUs disponíveis
    if not torch.cuda.is_available():
        print("CUDA não disponível, usando CPU")
        args.distributed = False
        args.num_gpus = 0
        gpu_ids = None
    else:
        # Listar GPUs disponíveis
        available_gpus = GPUManager.get_gpu_info()
        print("\nGPUs Disponíveis:")
        for gpu in available_gpus:
            print(f"GPU {gpu['index']}: {gpu['name']}")
            print(f"  Memória Total: {gpu['total_memory'] / 1024:.1f} GB")
            print(f"  Memória Livre: {gpu['free_memory'] / 1024:.1f} GB")
            print(f"  Capacidade: {gpu['compute_capability']}")
            print(f"  Núcleos: {gpu['multi_processor_count']}\n")
        
        # Processar IDs de GPU
        if args.gpu_ids:
            gpu_ids = [int(i) for i in args.gpu_ids.split(",")]
            args.num_gpus = len(gpu_ids)
        else:
            gpu_ids = None
            if args.num_gpus is None:
                args.num_gpus = len(available_gpus)
        
        # Validar número de GPUs
        max_gpus = len(available_gpus)
        if args.num_gpus > max_gpus:
            print(f"Ajustando num_gpus de {args.num_gpus} para {max_gpus}")
            args.num_gpus = max_gpus
    
    # Criar diretórios
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Otimizar configurações de GPU
    if torch.cuda.is_available():
        GPUManager.optimize_gpu_settings()
        print("\nConfigurações de GPU otimizadas:")
        print(f"- TF32 Habilitado: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"- cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"- Memória Reservada: 95%")
    
    # Iniciar treinamento
    if args.distributed and args.num_gpus > 1:
        print(f"\nIniciando treinamento distribuído com {args.num_gpus} GPUs")
        if gpu_ids:
            print(f"Usando GPUs: {gpu_ids}")
        else:
            print("Selecionando melhores GPUs automaticamente")
            
        train_distributed(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            world_size=args.num_gpus,
            gpu_ids=gpu_ids
        )
    else:
        print("\nIniciando treinamento single-GPU")
        if gpu_ids:
            selected_gpu = gpu_ids[0]
            print(f"Usando GPU {selected_gpu}")
        else:
            selected_gpu = GPUManager.select_best_gpus(1)[0]
            print(f"Selecionada melhor GPU: {selected_gpu}")
            
        trainer = MultiGPUTrainer(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            gpu_ids=[selected_gpu]
        )
        trainer.train()

if __name__ == "__main__":
    main() 