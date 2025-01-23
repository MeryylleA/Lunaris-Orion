"""
Script principal para treinamento do modelo Mini 16x16.
"""

import os
import argparse
from pathlib import Path
import torch
from datetime import datetime
from .training.trainer import MultiGPUTrainer, GPUManager
import yaml

def create_experiment_folders(base_dir: str) -> Path:
    """Cria estrutura de pastas para o experimento.
    
    Args:
        base_dir: Diretório base para o experimento
        
    Returns:
        Path do diretório do experimento
    """
    # Criar nome do experimento com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"experiment_{timestamp}"
    
    # Criar estrutura de pastas
    exp_dir = Path(base_dir) / exp_name
    
    folders = {
        "checkpoints": exp_dir / "checkpoints",
        "logs": {
            "tensorboard": exp_dir / "logs" / "tensorboard",
            "wandb": exp_dir / "logs" / "wandb"
        },
        "backups": exp_dir / "backups",
        "outputs": exp_dir / "outputs"
    }
    
    # Criar todas as pastas
    for folder in [exp_dir] + [p for p in folders.values() if isinstance(p, Path)] + [p for p in folders["logs"].values()]:
        folder.mkdir(parents=True, exist_ok=True)
        
    print(f"\nEstrutura de pastas criada em: {exp_dir}")
    print("├── checkpoints/")
    print("├── logs/")
    print("│   ├── tensorboard/")
    print("│   └── wandb/")
    print("├── backups/")
    print("└── outputs/")
    
    return exp_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Treina modelo Mini 16x16")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Diretório com dados de treinamento"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Ativar modo de desenvolvimento (usa CPU e configurações reduzidas)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configurar caminhos
    root_dir = Path(__file__).parent.parent
    config_path = root_dir / "mini_model" / "configs" / "base_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}\n"
            "Verifique se o arquivo base_config.yaml existe no diretório configs."
        )
        
    exp_dir = create_experiment_folders(root_dir / "experiments")
    
    # Configurar ambiente
    if args.dev:
        print("\nModo desenvolvimento ativado - usando CPU com otimizações")
        world_size = 1
        gpu_ids = None
        
        # Configurações otimizadas para CPU
        n_cores = os.cpu_count()
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(min(n_cores, 4))
        
        # Otimizações de CPU
        torch.backends.mkldnn.enabled = True
        torch.backends.openmp.enabled = True
        torch.set_float32_matmul_precision('medium')
        
        # Desativar recursos problemáticos em CPU
        os.environ["USE_FLASH_ATTENTION"] = "0"
        os.environ["USE_SPARSE_ATTENTION"] = "0"
        os.environ["USE_CHECKPOINT"] = "0"
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"
        os.environ["TORCH_USE_CUDA_DSA"] = "0"
        os.environ["TORCH_USE_RTLD_GLOBAL"] = "0"  # Evitar problemas com DLL em Windows
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
        os.environ["MKL_NUM_THREADS"] = str(n_cores)
        os.environ["PYTORCH_JIT"] = "0"  # Desativar JIT
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # Otimizar alocação de memória
        
        # Desativar otimizações que podem causar problemas
        torch._dynamo.config.suppress_errors = True
        torch.jit.enable = False
        torch._dynamo.config.dynamic_shapes = False
        torch._dynamo.config.cache_size_limit = 0
        
        # Reduzir batch size e dimensões do modelo para CPU
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Ajustar configurações para CPU
        config["model"].update({
            "embedding_dim": 256,  # Reduzir dimensões
            "ff_dim": 1024,
            "num_heads": 4,
            "num_layers": 6,  # Reduzir número de camadas
            "use_flash_attention": False,
            "use_sparse_attention": False,
            "use_checkpoint": False
        })
        
        config["training"].update({
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "mixed_precision": False,  # Desativar mixed precision em CPU
            "num_workers": 0,  # Reduzir workers para debug
            "learning_rate": 1e-4  # Ajustar learning rate
        })
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        gpu_ids = GPUManager.select_best_gpus(world_size)
            
        if gpu_ids:
            print("\nGPUs disponíveis:")
            for i, gpu_id in enumerate(gpu_ids):
                props = torch.cuda.get_device_properties(gpu_id)
                print(f"GPU {i}: {props.name} ({props.total_memory / 1024:.1f} GB)")
        else:
            print("\nNenhuma GPU disponível - usando CPU")
            world_size = 1
    
    # Iniciar treinamento
    if world_size > 1 and not args.dev:
        print(f"\nIniciando treinamento distribuído com {world_size} processos")
        
        # Configurar variáveis de ambiente
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
        # Iniciar processos
        torch.multiprocessing.spawn(
            _train_worker,
            args=(config_path, args.data_dir, str(exp_dir), world_size, gpu_ids, args.dev),
            nprocs=world_size,
            join=True
        )
    else:
        print("\nIniciando treinamento single-GPU")
        trainer = MultiGPUTrainer(
            config_path=config_path,
            data_dir=args.data_dir,
            output_dir=str(exp_dir),
            gpu_ids=gpu_ids if not args.dev else None,
            dev_mode=args.dev
        )
        trainer.train()

def _train_worker(
    local_rank: int,
    config_path: str,
    data_dir: str,
    output_dir: str,
    world_size: int,
    gpu_ids: list,
    dev_mode: bool
):
    """Worker para treinamento distribuído."""
    trainer = MultiGPUTrainer(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        local_rank=local_rank,
        world_size=world_size,
        gpu_ids=gpu_ids,
        dev_mode=dev_mode
    )
    trainer.train()

if __name__ == "__main__":
    main() 