import torch
from torch import nn
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from contextlib import nullcontext
import gc
import os

logger = logging.getLogger(__name__)

class PixelDiffusionModel:
    """Modelo otimizado para geração de pixel art usando múltiplas GPUs de alta performance."""
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        enable_memory_efficient: bool = True,
        multi_gpu: bool = True,
        batch_size: int = 16,
        max_retries: int = 3
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        self.max_retries = max_retries
        
        # Configuração de logging detalhado
        logger.info(f"Inicializando modelo em {device} com {torch_dtype}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"Memória Total GPU {i}: {props.total_memory / 1e9:.2f}GB")
                logger.info(f"Compute Capability: {props.major}.{props.minor}")
                logger.info(f"SMs: {props.multi_processor_count}")
        
        # Limpeza inicial de memória
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Configurações otimizadas para GPUs de alta performance
            if enable_memory_efficient and torch.cuda.is_available():
                # Configurações específicas por GPU
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    
                    # Configurações para H100
                    if "H100" in props.name:
                        self.torch_dtype = torch.bfloat16
                        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '8'
                        
                    # Configurações para A100
                    elif "A100" in props.name:
                        self.torch_dtype = torch.float16
                        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '4'
                        
                    # Configurações para L4/L40S
                    elif any(gpu in props.name for gpu in ["L4", "L40"]):
                        self.torch_dtype = torch.float16
                        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '2'
                        
                    # Configurações para V100/V100S
                    elif "V100" in props.name:
                        self.torch_dtype = torch.float16
                        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
                
                # Otimizações gerais
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Configurações CUDA
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
                
            # Configuração do BitsAndBytes para quantização 8-bit
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True
            )
            
            # Carrega o pipeline base com retry
            for attempt in range(self.max_retries):
                try:
                    self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                        model_id,
                        torch_dtype=self.torch_dtype,
                        variant="fp16",
                        use_safetensors=True,
                        safety_checker=None,
                        requires_safety_checker=False,
                        quantization_config=quantization_config
                    )
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"Tentativa {attempt + 1} falhou: {str(e)}. Tentando novamente...")
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Configuração Multi-GPU otimizada
            if self.multi_gpu:
                try:
                    gpu_count = torch.cuda.device_count()
                    if gpu_count >= 4:  # Para 4 ou mais GPUs
                        self.pipeline.text_encoder = self.pipeline.text_encoder.to("cuda:0")
                        self.pipeline.vae = self.pipeline.vae.to("cuda:1")
                        self.pipeline.unet = nn.DataParallel(
                            self.pipeline.unet, 
                            device_ids=list(range(2, gpu_count))
                        )
                    elif gpu_count == 3:  # Para 3 GPUs
                        self.pipeline.text_encoder = self.pipeline.text_encoder.to("cuda:0")
                        self.pipeline.vae = self.pipeline.vae.to("cuda:1")
                        self.pipeline.unet = self.pipeline.unet.to("cuda:2")
                    else:  # Para 2 GPUs
                        self.pipeline.text_encoder = self.pipeline.text_encoder.to("cuda:0")
                        self.pipeline.unet = self.pipeline.unet.to("cuda:1")
                        self.pipeline.vae = self.pipeline.vae.to("cuda:0")
                    
                    logger.info("Modelo distribuído entre múltiplas GPUs com sucesso")
                except Exception as e:
                    logger.error(f"Erro na distribuição multi-GPU: {str(e)}")
                    logger.info("Revertendo para modo single-GPU")
                    self.multi_gpu = False
                    self.pipeline = self.pipeline.to(device)
            else:
                self.pipeline = self.pipeline.to(device)
            
            # Otimizações de memória
            if enable_memory_efficient:
                self.pipeline.enable_attention_slicing(slice_size="max")
                self.pipeline.enable_vae_slicing()
                if not self.multi_gpu:
                    self.pipeline.enable_model_cpu_offload()
            
            logger.info("Modelo inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro fatal na inicialização do modelo: {str(e)}")
            raise
    
    def _preprocess_prompt(self, prompt: str) -> str:
        """Adiciona condicionamento específico para pixel art ao prompt."""
        pixel_art_conditioning = [
            "pixel art style",
            "pixelated",
            "clear pixels",
            "retro game aesthetic",
            "limited color palette",
            "sharp pixel edges",
            "high quality pixel art"
        ]
        return f"{', '.join(pixel_art_conditioning)}, {prompt}"
    
    def _postprocess_image(
        self,
        image: torch.Tensor,
        target_size: Tuple[int, int],
        palette_size: Optional[int] = None
    ) -> torch.Tensor:
        """Pós-processa a imagem para garantir qualidade de pixel art."""
        try:
            # Validação de entrada
            if not isinstance(image, torch.Tensor):
                raise ValueError("Input deve ser um tensor PyTorch")
            
            # Garante que a imagem está no formato correto
            if image.dim() != 4:
                image = image.unsqueeze(0) if image.dim() == 3 else image
            
            # Redimensiona com validação de tamanho
            if target_size[0] > 0 and target_size[1] > 0:
                image = nn.functional.interpolate(
                    image,
                    size=target_size,
                    mode='nearest'
                )
            else:
                raise ValueError("Dimensões de target_size devem ser positivas")
            
            # Quantização de cores com validação
            if palette_size:
                if palette_size < 2:
                    raise ValueError("palette_size deve ser pelo menos 2")
                image = torch.round(image * (palette_size - 1)) / (palette_size - 1)
            
            # Garante que os valores estão no intervalo [0, 1]
            image = torch.clamp(image, 0, 1)
            
            return image
            
        except Exception as e:
            logger.error(f"Erro no pós-processamento da imagem: {str(e)}")
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        negative_prompt: Optional[str] = None,
        resolution: Tuple[int, int] = (32, 32),
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        palette_size: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Gera múltiplas imagens em batch para melhor utilização da GPU."""
        try:
            # Validação de entrada
            if not prompts:
                raise ValueError("Lista de prompts não pode estar vazia")
            
            if len(prompts) > self.batch_size:
                logger.warning(f"Número de prompts ({len(prompts)}) excede batch_size ({self.batch_size})")
                prompts = prompts[:self.batch_size]
            
            # Configuração de seed
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            # Processa prompts
            processed_prompts = [self._preprocess_prompt(prompt) for prompt in prompts]
            
            # Gerenciamento de memória antes da geração
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Contexto de autocast para mixed precision
            autocast = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext
            
            # Gera imagens com retry
            for attempt in range(self.max_retries):
                try:
                    with autocast():
                        result = self.pipeline(
                            prompt=processed_prompts,
                            negative_prompt=[negative_prompt] * len(prompts) if negative_prompt else None,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            max_sequence_length=max_sequence_length,
                            num_images_per_prompt=1,
                            **kwargs
                        )
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e) and attempt < self.max_retries - 1:
                        logger.warning(f"OOM na tentativa {attempt + 1}, limpando memória e tentando novamente...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise
            
            # Processa resultados
            results = []
            for i, (image, prompt) in enumerate(zip(result.images, prompts)):
                try:
                    if not isinstance(image, torch.Tensor):
                        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
                        image = image.to(self.device) / 255.0
                    
                    final_image = self._postprocess_image(
                        image,
                        target_size=resolution,
                        palette_size=palette_size
                    )
                    
                    results.append({
                        "image": final_image,
                        "seed": seed + i if seed is not None else torch.initial_seed(),
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "parameters": {
                            "resolution": resolution,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "palette_size": palette_size,
                            "max_sequence_length": max_sequence_length
                        }
                    })
                except Exception as e:
                    logger.error(f"Erro no processamento da imagem {i}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na geração em batch: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        resolution: Tuple[int, int] = (32, 32),
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        palette_size: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Gera uma única imagem de pixel art."""
        try:
            results = self.generate_batch(
                prompts=[prompt],
                negative_prompt=negative_prompt,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                palette_size=palette_size,
                seed=seed,
                **kwargs
            )
            return results[0]
        except Exception as e:
            logger.error(f"Erro na geração única: {str(e)}")
            raise
    
    def to(self, device: str) -> 'PixelDiffusionModel':
        """Move o modelo para o dispositivo especificado."""
        if self.multi_gpu:
            logger.warning("Modelo está em modo multi-GPU, ignorando mudança de device")
            return self
        
        try:
            self.device = device
            self.pipeline = self.pipeline.to(device)
            return self
        except Exception as e:
            logger.error(f"Erro ao mover modelo para {device}: {str(e)}")
            raise
    
    @staticmethod
    def list_available_styles() -> Dict[str, str]:
        """Retorna os estilos de pixel art disponíveis."""
        return {
            "retro": "estilo retro 8-bit",
            "cyberpunk": "estilo cyberpunk neon",
            "fantasy": "estilo fantasy medieval",
            "minimal": "estilo minimalista",
            "anime": "estilo anime pixel art",
            "modern": "estilo moderno clean"
        }
    
    def __del__(self):
        """Limpeza de recursos ao destruir a instância."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.error(f"Erro na limpeza de recursos: {str(e)}") 