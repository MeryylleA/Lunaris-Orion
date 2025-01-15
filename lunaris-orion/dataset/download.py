import os
import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import json
from typing import Dict, List, Optional
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiffusionDBDownloader:
    """Classe para baixar e preparar o dataset DiffusionDB."""
    
    def __init__(
        self,
        output_dir: str,
        num_workers: int = 4,
        subset: str = "2M",  # ou "Large"
        filter_pixel_art: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.subset = subset
        self.filter_pixel_art = filter_pixel_art
        
        # Cria diretórios necessários
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # Lock para download concorrente
        self._download_lock = threading.Lock()
    
    def _download_metadata(self) -> pd.DataFrame:
        """Baixa e carrega os metadados do dataset."""
        logger.info("Baixando metadados do dataset...")
        
        # Carrega o dataset usando a biblioteca datasets
        dataset = load_dataset("poloclub/diffusiondb")
        metadata_df = pd.DataFrame(dataset["train"])
        
        if self.filter_pixel_art:
            # Filtra prompts relacionados a pixel art
            pixel_art_keywords = [
                "pixel art", "pixelated", "8-bit", "16-bit", "retro game",
                "sprite", "pixel", "pixelized", "pixel-art"
            ]
            
            filter_condition = metadata_df["prompt"].str.lower().apply(
                lambda x: any(keyword in x.lower() for keyword in pixel_art_keywords)
            )
            
            metadata_df = metadata_df[filter_condition]
            logger.info(f"Filtradas {len(metadata_df)} imagens relacionadas a pixel art")
        
        # Salva metadados processados
        metadata_df.to_parquet(self.output_dir / "metadata_processed.parquet")
        
        return metadata_df
    
    def _download_image(self, image_info: Dict) -> Optional[str]:
        """Baixa uma única imagem do dataset."""
        try:
            image_name = image_info["image_name"]
            part_id = image_info["part_id"]
            
            # Constrói o URL da imagem
            if self.subset == "2M":
                url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-{part_id:06d}/{image_name}"
            else:
                url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/diffusiondb-large-part-{1 if part_id <= 10000 else 2}/part-{part_id:06d}/{image_name}"
            
            # Baixa a imagem
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                output_path = self.images_dir / image_name
                with open(output_path, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
                return str(output_path)
            else:
                logger.warning(f"Erro ao baixar {image_name}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao processar {image_name}: {str(e)}")
            return None
    
    def download_dataset(self) -> None:
        """Baixa todo o dataset."""
        # Baixa e processa metadados
        metadata_df = self._download_metadata()
        total_images = len(metadata_df)
        
        logger.info(f"Iniciando download de {total_images} imagens...")
        
        # Prepara lista de imagens para download
        image_infos = [
            {"image_name": row["image_name"], "part_id": row["part_id"]}
            for _, row in metadata_df.iterrows()
        ]
        
        # Download paralelo das imagens
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(self._download_image, image_infos),
                total=len(image_infos),
                desc="Baixando imagens"
            ))
        
        # Conta resultados
        successful = len([r for r in results if r is not None])
        failed = len(results) - successful
        
        logger.info(f"Download concluído: {successful} sucesso, {failed} falhas")
        
        # Salva relatório
        report = {
            "total_images": total_images,
            "successful_downloads": successful,
            "failed_downloads": failed,
            "filtered_pixel_art": self.filter_pixel_art
        }
        
        with open(self.output_dir / "download_report.json", "w") as f:
            json.dump(report, f, indent=2)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download do dataset DiffusionDB")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Diretório para salvar o dataset")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Número de workers para download paralelo")
    parser.add_argument("--subset", type=str, choices=["2M", "Large"], default="2M",
                       help="Qual subset do DiffusionDB usar")
    parser.add_argument("--no_filter", action="store_true",
                       help="Desativa filtro de pixel art")
    
    args = parser.parse_args()
    
    downloader = DiffusionDBDownloader(
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        subset=args.subset,
        filter_pixel_art=not args.no_filter
    )
    
    downloader.download_dataset() 