import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from typing import Optional, Tuple, Dict, List
import json
from tqdm import tqdm
import cv2

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiffusionDBPreprocessor:
    """Classe para pré-processar o dataset DiffusionDB para treinamento."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_sizes: List[Tuple[int, int]] = [(32, 32), (64, 64), (128, 128)],
        palette_sizes: List[int] = [8, 16, 32],
        edge_enhance: bool = True,
        symmetry_augmentation: bool = True
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_sizes = target_sizes
        self.palette_sizes = palette_sizes
        self.edge_enhance = edge_enhance
        self.symmetry_augmentation = symmetry_augmentation
        
        # Cria diretórios de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for size in target_sizes:
            (self.output_dir / f"{size[0]}x{size[1]}").mkdir(exist_ok=True)
        
        # Carrega metadados
        self.metadata_df = pd.read_parquet(self.input_dir / "metadata_processed.parquet")
        
        # Pipeline de transformações base
        self.base_transform = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _quantize_colors(self, image: np.ndarray, palette_size: int) -> np.ndarray:
        """Quantiza as cores da imagem para criar uma paleta limitada."""
        # Converte para espaço de cores LAB para melhor quantização
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Reshape para 2D array de pixels
        pixels = lab_image.reshape(-1, 3)
        
        # Quantização usando K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels.astype(np.float32),
            palette_size,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Reconstrói a imagem
        quantized = centers[labels.flatten()].reshape(image.shape)
        
        # Converte de volta para RGB
        return cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Realça as bordas da imagem para criar pixels mais nítidos."""
        if not self.edge_enhance:
            return image
        
        # Kernel para realce de bordas
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        # Aplica o filtro separadamente em cada canal
        enhanced = np.zeros_like(image)
        for i in range(3):
            enhanced[:,:,i] = cv2.filter2D(image[:,:,i], -1, kernel)
        
        return np.clip(enhanced, 0, 1)
    
    def _create_symmetry(self, image: np.ndarray) -> List[np.ndarray]:
        """Cria variações simétricas da imagem."""
        if not self.symmetry_augmentation:
            return [image]
        
        flipped_h = np.fliplr(image)
        flipped_v = np.flipud(image)
        flipped_both = np.flipud(flipped_h)
        
        return [image, flipped_h, flipped_v, flipped_both]
    
    def _process_single_image(
        self,
        image_path: str,
        prompt: str,
        target_size: Tuple[int, int]
    ) -> Dict[str, List[Dict]]:
        """Processa uma única imagem para um tamanho alvo específico."""
        try:
            # Carrega a imagem
            image = Image.open(image_path).convert('RGB')
            image = np.array(image) / 255.0
            
            results = []
            
            # Para cada tamanho de paleta
            for palette_size in self.palette_sizes:
                # Redimensiona
                resized = cv2.resize(
                    image,
                    target_size,
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Quantiza cores
                quantized = self._quantize_colors(resized, palette_size)
                
                # Realça bordas
                enhanced = self._enhance_edges(quantized)
                
                # Cria variações simétricas
                variations = self._create_symmetry(enhanced)
                
                # Salva cada variação
                for i, var_image in enumerate(variations):
                    output_name = f"{Path(image_path).stem}_p{palette_size}_var{i}.png"
                    output_path = self.output_dir / f"{target_size[0]}x{target_size[1]}" / output_name
                    
                    # Salva a imagem
                    Image.fromarray((var_image * 255).astype(np.uint8)).save(output_path)
                    
                    # Adiciona aos resultados
                    results.append({
                        "file_path": str(output_path),
                        "prompt": prompt,
                        "target_size": target_size,
                        "palette_size": palette_size,
                        "variation": i,
                        "original_path": image_path
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Erro ao processar {image_path}: {str(e)}")
            return []
    
    def process_dataset(self) -> None:
        """Processa todo o dataset."""
        logger.info("Iniciando processamento do dataset...")
        
        all_results = []
        
        # Para cada tamanho alvo
        for target_size in self.target_sizes:
            logger.info(f"Processando imagens para tamanho {target_size}...")
            
            # Processa cada imagem
            for _, row in tqdm(self.metadata_df.iterrows(), total=len(self.metadata_df)):
                image_path = self.input_dir / "images" / row["image_name"]
                if not image_path.exists():
                    continue
                
                results = self._process_single_image(
                    str(image_path),
                    row["prompt"],
                    target_size
                )
                
                all_results.extend(results)
        
        # Salva metadados do processamento
        metadata = pd.DataFrame(all_results)
        metadata.to_parquet(self.output_dir / "processed_metadata.parquet")
        
        # Salva relatório
        report = {
            "total_original_images": len(self.metadata_df),
            "total_processed_variations": len(all_results),
            "target_sizes": [f"{size[0]}x{size[1]}" for size in self.target_sizes],
            "palette_sizes": self.palette_sizes,
            "edge_enhance": self.edge_enhance,
            "symmetry_augmentation": self.symmetry_augmentation
        }
        
        with open(self.output_dir / "processing_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("Processamento concluído!")

class ProcessedDiffusionDBDataset(Dataset):
    """Dataset para carregar as imagens processadas do DiffusionDB."""
    
    def __init__(
        self,
        processed_dir: str,
        target_size: Optional[Tuple[int, int]] = None,
        transform: Optional[A.Compose] = None
    ):
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        
        # Carrega metadados
        self.metadata = pd.read_parquet(self.processed_dir / "processed_metadata.parquet")
        
        # Filtra por tamanho se especificado
        if target_size:
            size_str = f"{target_size[0]}x{target_size[1]}"
            self.metadata = self.metadata[
                self.metadata["file_path"].str.contains(f"/{size_str}/")
            ]
        
        logger.info(f"Dataset carregado com {len(self.metadata)} imagens")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        
        # Carrega a imagem
        image = Image.open(row["file_path"]).convert('RGB')
        image = np.array(image)
        
        # Aplica transformações se definidas
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # Converte para tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            "image": image,
            "prompt": row["prompt"],
            "target_size": row["target_size"],
            "palette_size": row["palette_size"]
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pré-processador do DiffusionDB")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Diretório com o dataset baixado")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Diretório para salvar resultados")
    parser.add_argument("--target_sizes", type=str, default="32x32,64x64,128x128",
                       help="Tamanhos alvo (formato: WxH,WxH,...)")
    parser.add_argument("--palette_sizes", type=str, default="8,16,32",
                       help="Tamanhos de paleta (formato: N,N,...)")
    parser.add_argument("--no_edge_enhance", action="store_true",
                       help="Desativa realce de bordas")
    parser.add_argument("--no_symmetry", action="store_true",
                       help="Desativa augmentação por simetria")
    
    args = parser.parse_args()
    
    # Processa argumentos de tamanho
    target_sizes = [
        tuple(map(int, size.split("x")))
        for size in args.target_sizes.split(",")
    ]
    
    palette_sizes = list(map(int, args.palette_sizes.split(",")))
    
    # Inicializa e executa o preprocessador
    preprocessor = DiffusionDBPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sizes=target_sizes,
        palette_sizes=palette_sizes,
        edge_enhance=not args.no_edge_enhance,
        symmetry_augmentation=not args.no_symmetry
    )
    
    preprocessor.process_dataset() 