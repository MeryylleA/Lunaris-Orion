import torch
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import os
import json

logger = logging.getLogger(__name__)

class ModelSetup:
    def __init__(self):
        self.model_id = "stabilityai/stable-diffusion-3.5-large"
        self.root_dir = Path(__file__).parent.parent
        self.models_dir = self.root_dir / "models"
        self.dataset_dir = self.root_dir / "dataset"
        self.cache_dir = self.root_dir / "cache"
        
        # Detecta GPUs disponíveis
        self.available_gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.available_gpus.append({
                    "index": i,
                    "name": props.name,
                    "memory": props.total_memory / 1e9,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "sms": props.multi_processor_count
                })
        
        # Configurações otimizadas por GPU
        self.model_config = self._get_optimal_config()
    
    def _get_optimal_config(self) -> dict:
        """Determina a configuração ótima baseada nas GPUs disponíveis."""
        config = {
            "use_safetensors": True,
            "device_map": "auto"
        }
        
        if not self.available_gpus:
            config["torch_dtype"] = torch.float32
            return config
        
        # Identifica as GPUs mais poderosas
        gpu_types = {
            "H100": {"priority": 1, "dtype": torch.bfloat16, "batch_size": 32},
            "A100": {"priority": 2, "dtype": torch.float16, "batch_size": 24},
            "L40": {"priority": 3, "dtype": torch.float16, "batch_size": 16},
            "L4": {"priority": 4, "dtype": torch.float16, "batch_size": 12},
            "V100": {"priority": 5, "dtype": torch.float16, "batch_size": 8}
        }
        
        best_priority = float('inf')
        selected_config = None
        
        for gpu in self.available_gpus:
            for gpu_type, specs in gpu_types.items():
                if gpu_type in gpu["name"] and specs["priority"] < best_priority:
                    best_priority = specs["priority"]
                    selected_config = specs
        
        if selected_config:
            config.update({
                "torch_dtype": selected_config["dtype"],
                "batch_size": selected_config["batch_size"],
                "variant": "fp16" if selected_config["dtype"] == torch.float16 else None
            })
        else:
            # Configuração padrão para outras GPUs
            config.update({
                "torch_dtype": torch.float16,
                "batch_size": 8,
                "variant": "fp16"
            })
        
        # Configurações adicionais para multi-GPU
        if len(self.available_gpus) > 1:
            total_memory = sum(gpu["memory"] for gpu in self.available_gpus)
            config.update({
                "device_map": "balanced",
                "max_memory": {
                    f"cuda:{i}": f"{int(gpu['memory']*0.95)}GiB"
                    for i, gpu in enumerate(self.available_gpus)
                }
            })
        
        return config
    
    def get_training_config(self) -> dict:
        """Retorna configurações otimizadas para treinamento."""
        config = self._get_optimal_config()
        
        # Ajustes específicos para treinamento
        if self.available_gpus:
            total_memory = sum(gpu["memory"] for gpu in self.available_gpus)
            
            # Configurações baseadas na memória total disponível
            if total_memory >= 160:  # H100 + A100 ou configuração similar
                config.update({
                    "gradient_accumulation_steps": 1,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": False
                })
            elif total_memory >= 80:  # Múltiplas V100 ou similar
                config.update({
                    "gradient_accumulation_steps": 2,
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True
                })
            else:
                config.update({
                    "gradient_accumulation_steps": 4,
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True
                })
        
        return config
    
    def get_inference_config(self) -> dict:
        """Retorna configurações otimizadas para inferência."""
        config = self._get_optimal_config()
        
        if self.available_gpus:
            # Habilita otimizações específicas para inferência
            config.update({
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "torch_compile": True,
                "compile_mode": "reduce-overhead"
            })
            
            # Configurações de quantização para GPUs com menos memória
            if any(gpu["memory"] < 24 for gpu in self.available_gpus):
                config.update({
                    "load_in_8bit": True,
                    "quantization_config": {
                        "llm_int8_threshold": 6.0,
                        "llm_int8_has_fp16_weight": True
                    }
                })
        
        return config
    
    def setup_dataset(self):
        """Baixa e prepara o dataset DiffusionDB."""
        logger.info("Iniciando download do dataset...")
        
        dataset_url = "https://huggingface.co/datasets/poloclub/diffusiondb"
        metadata_file = self.dataset_dir / "metadata.json"
        
        try:
            # Cria diretórios
            self.dataset_dir.mkdir(exist_ok=True)
            (self.dataset_dir / "images").mkdir(exist_ok=True)
            
            # Download dos metadados
            if not metadata_file.exists():
                logger.info("Baixando metadados...")
                # Implementar download real dos metadados
            
            # Filtra imagens de pixel art
            logger.info("Filtrando imagens de pixel art...")
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            pixel_art_prompts = [
                item for item in metadata
                if any(kw in item["prompt"].lower() 
                      for kw in ["pixel art", "pixelated", "8-bit", "16-bit"])
            ]
            
            # Download das imagens
            logger.info(f"Baixando {len(pixel_art_prompts)} imagens...")
            for item in tqdm(pixel_art_prompts):
                # Implementar download real das imagens
                pass
            
            logger.info("Dataset preparado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao preparar dataset: {str(e)}")
            raise
    
    def setup_model(self):
        """Configura o modelo base e otimizações."""
        logger.info("Configurando modelo...")
        
        try:
            # Baixa modelo
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                **self.model_config,
                cache_dir=str(self.cache_dir)
            )
            
            # Otimizações para V100S
            pipeline.enable_attention_slicing("max")
            pipeline.enable_vae_slicing()
            if torch.cuda.is_available():
                pipeline.enable_xformers_memory_efficient_attention()
            
            # Salva configurações
            pipeline.save_pretrained(self.models_dir / "base_model")
            
            logger.info("Modelo configurado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao configurar modelo: {str(e)}")
            raise 