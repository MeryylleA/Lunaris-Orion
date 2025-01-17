import os
import logging
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import accelerate
import diffusers
import datasets
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from .config import TrainingConfig

logger = get_logger(__name__, log_level="INFO")

def create_logging_dir(config):
    """Cria diretório para logs e checkpoints"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_dataset(config):
    """Carrega e prepara o dataset"""
    dataset = datasets.load_dataset(
        config.dataset_name,
        config.dataset_subset,
        cache_dir="cache"
    )
    
    # Filtra apenas imagens de pixel art
    def is_pixel_art(example):
        return any(tag in example["prompt"].lower() for tag in ["pixel", "8-bit", "16-bit", "retro game"])
    
    dataset = dataset.filter(is_pixel_art)
    
    # Função de pré-processamento
    def preprocess_images(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        images = [image.resize((config.resolution, config.resolution)) for image in images]
        pixel_values = [torch.from_numpy(image).float() / 127.5 - 1.0 for image in images]
        return {"pixel_values": pixel_values, "prompt": examples["prompt"]}
    
    processed_dataset = dataset.map(
        preprocess_images,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count()
    )
    
    return processed_dataset

def create_dataloaders(dataset, config, accelerator):
    """Cria os dataloaders para treinamento"""
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=config.train_batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    return train_dataloader

def train_one_epoch(
    config,
    epoch,
    train_dataloader,
    accelerator,
    unet,
    text_encoder,
    vae,
    optimizer,
    lr_scheduler,
    weight_dtype,
    global_step
):
    """Executa um epoch de treinamento"""
    progress_bar = tqdm(
        total=len(train_dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch}"
    )
    
    unet.train()
    
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            # Converte imagens para latents
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # Adiciona ruído aos latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],))
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Codifica o texto
            encoder_hidden_states = text_encoder(batch["prompt"])[0]
            
            # Predição do ruído
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Calcula a loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            # Backpropagation
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            
            if global_step % config.validation_steps == 0:
                logger.info(f"Step {global_step}: loss = {loss.detach().item():.4f}")
                
            if global_step % config.checkpointing_steps == 0:
                save_checkpoint(
                    config, accelerator, unet, text_encoder,
                    global_step, epoch, output_dir
                )
                
    return global_step

def save_checkpoint(config, accelerator, unet, text_encoder, global_step, epoch, output_dir):
    """Salva um checkpoint do modelo"""
    if accelerator.is_main_process:
        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva os modelos
        accelerator.save_state(checkpoint_dir)
        
        # Salva uma amostra de validação
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model_name,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            torch_dtype=torch.float16
        )
        
        for i, prompt in enumerate(config.validation_prompts):
            image = pipeline(prompt).images[0]
            image.save(checkpoint_dir / f"validation_{i}.png")
            
        logger.info(f"Saved checkpoint: {checkpoint_dir}")

def main():
    # Carrega configuração
    config = TrainingConfig()
    
    # Inicializa acelerador
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_project_config
    )
    
    # Prepara diretório de output
    output_dir = create_logging_dir(config)
    
    # Carrega tokenizer e modelos
    tokenizer = CLIPTokenizer.from_pretrained(config.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.model_name, subfolder="unet")
    
    # Configura otimizações
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        
    if config.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()
        
    if config.torch_compile:
        unet = torch.compile(unet)
    
    # Prepara dataset
    dataset = load_dataset(config)
    train_dataloader = create_dataloaders(dataset, config, accelerator)
    
    # Configura otimizador
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Para usar 8-bit Adam, instale bitsandbytes: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW
        
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )
    
    # Prepara scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_train_steps
    )
    
    # Prepara para treinamento distribuído
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Treina o modelo
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Iniciando treinamento *****")
    logger.info(f"  Tamanho do dataset = {len(dataset)}")
    logger.info(f"  Tamanho do batch = {total_batch_size}")
    logger.info(f"  Número de epochs = {config.num_train_epochs}")
    logger.info(f"  Passos de gradient accumulation = {config.gradient_accumulation_steps}")
    
    global_step = 0
    for epoch in range(config.num_train_epochs):
        global_step = train_one_epoch(
            config=config,
            epoch=epoch,
            train_dataloader=train_dataloader,
            accelerator=accelerator,
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            weight_dtype=accelerator.unwrap_model(unet).dtype,
            global_step=global_step
        )
        
    # Salva o modelo final
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model_name,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder)
        )
        pipeline.save_pretrained(output_dir / "modelo-final")
        
    accelerator.end_training()

if __name__ == "__main__":
    main() 