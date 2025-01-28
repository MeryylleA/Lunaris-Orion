"""
Módulo para gerenciar o download e processamento de prompts do dataset DiffusionDB.
"""

from typing import List, Dict, Optional
import os
from pathlib import Path
import json
import numpy as np
from datasets import load_dataset
import logging
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(self, config: Dict):
        """
        Inicializa o gerenciador de prompts.
        
        Args:
            config: Dicionário com configurações
        """
        self.config = config
        self.cache_dir = Path(config.get('prompt_cache_dir', 'data/prompt_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = None
        
    def download_and_prepare(self, subset: str = '2k_random_1k') -> bool:
        """
        Baixa e prepara os prompts do DiffusionDB.
        
        Args:
            subset: Nome do subset do dataset a ser usado
            
        Returns:
            bool: True se o download e preparação foram bem sucedidos
        """
        cache_file = self.cache_dir / f'prompts_{subset}.json'
        
        # Verifica se já existe cache
        if cache_file.exists():
            try:
                logger.info(f"Carregando prompts do cache: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.prompts = json.load(f)
                return True
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")
                # Se houver erro, tenta baixar novamente
                
        try:
            # Baixa o dataset
            logger.info(f"Baixando dataset DiffusionDB (subset: {subset})")
            dataset = load_dataset('jainr3/diffusiondb-pixelart', subset)
            
            if not dataset:
                logger.error("Falha ao carregar o dataset")
                return False
                
            # Extrai e processa os prompts
            logger.info("Processando prompts...")
            processed_prompts = []
            
            for item in tqdm(dataset['train'], desc="Processando prompts"):
                if 'text' in item and item['text']:
                    # Remove caracteres especiais e normaliza
                    prompt = self._clean_prompt(item['text'])
                    if prompt:  # Só adiciona se não estiver vazio
                        processed_prompts.append(prompt)
            
            if not processed_prompts:
                logger.error("Nenhum prompt válido encontrado")
                return False
                
            # Salva no cache
            logger.info(f"Salvando {len(processed_prompts)} prompts no cache")
            self.prompts = processed_prompts
            
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_prompts, f, ensure_ascii=False, indent=2)
                logger.info("Cache salvo com sucesso")
                return True
            except Exception as e:
                logger.error(f"Erro ao salvar cache: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Erro durante download/processamento: {e}")
            return False
    
    def _clean_prompt(self, prompt: str) -> Optional[str]:
        """
        Limpa e normaliza um prompt.
        
        Args:
            prompt: Texto do prompt
            
        Returns:
            str: Prompt limpo ou None se inválido
        """
        try:
            # Remove espaços extras
            prompt = ' '.join(prompt.split())
            
            # Remove caracteres especiais problemáticos
            prompt = prompt.replace('\n', ' ').replace('\r', ' ')
            
            # Verifica tamanho mínimo
            if len(prompt) < 3:
                return None
                
            return prompt
        except Exception as e:
            logger.warning(f"Erro ao limpar prompt: {e}")
            return None
    
    def get_random_prompts(self, n: int = 1) -> List[str]:
        """
        Retorna prompts aleatórios do conjunto.
        
        Args:
            n: Número de prompts a retornar
            
        Returns:
            List[str]: Lista de prompts
        """
        if not self.prompts:
            logger.warning("Nenhum prompt carregado")
            return []
            
        try:
            return list(np.random.choice(self.prompts, size=min(n, len(self.prompts)), replace=False))
        except Exception as e:
            logger.error(f"Erro ao selecionar prompts aleatórios: {e}")
            return []
    
    def get_prompt_count(self) -> int:
        """
        Retorna o número total de prompts disponíveis.
        
        Returns:
            int: Número de prompts
        """
        return len(self.prompts) if self.prompts else 0 