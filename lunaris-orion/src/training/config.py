from dataclasses import dataclass
from typing import Optional, List
import torch
from pathlib import Path

@dataclass
class TrainingConfig:
    # Diretórios e caminhos
    output_dir: str = "models/lunaris-pixel-art"
    dataset_name: str = "poloclub/diffusiondb"
    dataset_subset: str = "2m"
    model_name: str = "stabilityai/stable-diffusion-3.5-large"
    
    # Parâmetros de treinamento
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = 10000
    
    # Parâmetros do modelo
    resolution: int = 1024
    center_crop: bool = True
    random_flip: bool = True
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    
    # Otimizações para NVIDIA L40S
    use_8bit_adam: bool = True
    enable_cuda_graph: bool = True
    torch_compile: bool = True
    
    # Logging e checkpoints
    checkpointing_steps: int = 1000
    validation_steps: int = 200
    validation_prompts: List[str] = None
    
    def __post_init__(self):
        self.validation_prompts = [
            "pixel art of a medieval castle on a hill, 16-bit style",
            "pixel art character sprite of a warrior with sword and shield",
            "isometric pixel art of a cozy village with small houses",
            "pixel art landscape with mountains and pine trees at sunset",
            "retro gaming style pixel art dungeon with torches and treasure"
        ]
        
        # Configurações específicas para L40S
        if torch.cuda.get_device_properties(0).name == "NVIDIA L40S":
            self.train_batch_size = 4
            self.gradient_accumulation_steps = 1
            self.mixed_precision = "bf16"
            self.enable_cuda_graph = True
            
    def save(self, path: Path):
        """Salva a configuração em um arquivo JSON"""
        import json
        
        config_dict = {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.__dict__.items()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
            
    @classmethod
    def load(cls, path: Path):
        """Carrega a configuração de um arquivo JSON"""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        return cls(**config_dict) 