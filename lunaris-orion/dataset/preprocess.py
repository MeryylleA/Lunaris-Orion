import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import json
import logging
from typing import Optional, Tuple, Dict, List
import cv2
from tqdm import tqdm
import albumentations as A

logger = logging.getLogger(__name__)

class PixelArtPreprocessor:
    """Classe para pré-processar imagens para o formato de pixel art."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (32, 32),
        palette_size: Optional[int] = None,
        edge_enhance: bool = True,
        symmetry_augmentation: bool = True
    ):
        self.target_size = target_size
        self.palette_size = palette_size
        self.edge_enhance = edge_enhance
        self.symmetry_augmentation = symmetry_augmentation
        
        # Pipeline de transformações
        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _quantize_colors(self, image: np.ndarray) -> np.ndarray:
        """Quantiza as cores da imagem para criar uma paleta limitada."""
        if self.palette_size:
            image = np.round(image * (self.palette_size - 1)) / (self.palette_size - 1)
        return image
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Realça as bordas da imagem para criar pixels mais nítidos."""
        if not self.edge_enhance:
            return image
        
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(image, -1, kernel)
        return np.clip(enhanced, 0, 1)
    
    def _create_symmetry(self, image: np.ndarray) -> List[np.ndarray]:
        """Cria variações simétricas da imagem."""
        if not self.symmetry_augmentation:
            return [image]
        
        flipped_h = np.fliplr(image)
        flipped_v = np.flipud(image)
        flipped_both = np.flipud(flipped_h)
        
        return [image, flipped_h, flipped_v, flipped_both]
    
    def process_image(self, image_path: str) -> Dict[str, np.ndarray]:
        """Processa uma única imagem."""
        try:
            # Carrega a imagem
            image = Image.open(image_path).convert('RGB')
            image = np.array(image) / 255.0
            
            # Aplica transformações básicas
            transformed = self.transform(image=image)['image']
            
            # Quantiza cores
            quantized = self._quantize_colors(transformed)
            
            # Realça bordas
            enhanced = self._enhance_edges(quantized)
            
            # Cria variações simétricas
            variations = self._create_symmetry(enhanced)
            
            return {
                'original': transformed,
                'processed': enhanced,
                'variations': variations
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar {image_path}: {str(e)}")
            return None

class PixelArtDataset(Dataset):
    """Dataset para treinamento com imagens de pixel art."""
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (32, 32)
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Carrega metadados se disponíveis
        self.metadata = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Lista todas as imagens
        self.image_paths = list(self.data_dir.glob('**/*.png')) + \
                          list(self.data_dir.glob('**/*.jpg'))
        
        logger.info(f"Encontradas {len(self.image_paths)} imagens em {data_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = str(self.image_paths[idx])
        
        # Carrega a imagem
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Aplica transformações se definidas
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Converte para tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Carrega metadados se disponíveis
        metadata = self.metadata.get(str(self.image_paths[idx].name), {})
        
        return {
            'image': image,
            'metadata': metadata
        }

def process_dataset(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (32, 32),
    palette_size: Optional[int] = None,
    edge_enhance: bool = True,
    symmetry_augmentation: bool = True
) -> None:
    """Processa todo o dataset de imagens."""
    
    # Cria diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Inicializa o preprocessador
    preprocessor = PixelArtPreprocessor(
        target_size=target_size,
        palette_size=palette_size,
        edge_enhance=edge_enhance,
        symmetry_augmentation=symmetry_augmentation
    )
    
    # Lista todas as imagens
    input_path = Path(input_dir)
    image_paths = list(input_path.glob('**/*.png')) + \
                 list(input_path.glob('**/*.jpg'))
    
    logger.info(f"Iniciando processamento de {len(image_paths)} imagens")
    
    # Processa cada imagem
    metadata = {}
    for img_path in tqdm(image_paths, desc="Processando imagens"):
        try:
            result = preprocessor.process_image(str(img_path))
            if result is None:
                continue
            
            # Salva a imagem processada
            output_name = output_path / f"{img_path.stem}_processed.png"
            Image.fromarray((result['processed'] * 255).astype(np.uint8)).save(output_name)
            
            # Salva variações se geradas
            for i, variation in enumerate(result['variations']):
                var_name = output_path / f"{img_path.stem}_var{i}.png"
                Image.fromarray((variation * 255).astype(np.uint8)).save(var_name)
            
            # Adiciona aos metadados
            metadata[img_path.name] = {
                'original_path': str(img_path),
                'processed_path': str(output_name),
                'variations': [str(output_path / f"{img_path.stem}_var{i}.png")
                             for i in range(len(result['variations']))]
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar {img_path}: {str(e)}")
            continue
    
    # Salva metadados
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Processamento concluído. Resultados salvos em {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pré-processador de dataset para pixel art")
    parser.add_argument("--input_dir", type=str, required=True, help="Diretório com imagens originais")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório para salvar resultados")
    parser.add_argument("--target_size", type=int, nargs=2, default=[32, 32], help="Tamanho alvo (altura, largura)")
    parser.add_argument("--palette_size", type=int, default=None, help="Tamanho da paleta de cores")
    parser.add_argument("--no_edge_enhance", action="store_true", help="Desativa realce de bordas")
    parser.add_argument("--no_symmetry", action="store_true", help="Desativa augmentação por simetria")
    
    args = parser.parse_args()
    
    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Processa o dataset
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        palette_size=args.palette_size,
        edge_enhance=not args.no_edge_enhance,
        symmetry_augmentation=not args.no_symmetry
    ) 