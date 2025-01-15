#!/usr/bin/env python3
"""Script para empacotar a extensão Lunaris Orion."""

import os
import zipfile
import shutil
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_aseprite_extension_dir():
    """Retorna o diretório de extensões do Aseprite"""
    if os.name == 'nt':  # Windows
        return Path(os.getenv('APPDATA')) / 'Aseprite' / 'extensions'
    else:  # Linux/Mac
        return Path.home() / '.config' / 'aseprite' / 'extensions'

def build_extension(version: str = "2.0.0", install: bool = False):
    """Empacota a extensão no formato .aseprite-extension"""
    
    # Diretórios
    root_dir = Path(__file__).parent
    plugin_dir = root_dir / "aseprite_plugin"
    build_dir = root_dir / "build"
    
    # Cria diretório build
    build_dir.mkdir(exist_ok=True)
    
    # Nome do arquivo de saída
    output_file = build_dir / f"lunaris-orion-v{version}.aseprite-extension"
    
    try:
        # Remove arquivo anterior
        if output_file.exists():
            output_file.unlink()
        
        # Lista de arquivos
        files_to_pack = ["plugin.lua", "package.json"]
        
        # Verifica arquivos
        for file in files_to_pack:
            file_path = plugin_dir / file
            if not file_path.exists():
                raise FileNotFoundError(f"Arquivo {file} não encontrado em {plugin_dir}")
        
        # Cria arquivo zip
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in files_to_pack:
                file_path = plugin_dir / file
                zf.write(file_path, file_path.name)
        
        logger.info(f"Extensão empacotada: {output_file}")
        
        # Instala a extensão
        if install:
            extension_dir = get_aseprite_extension_dir()
            install_dir = extension_dir / f"lunaris-orion-v{version}"
            
            # Remove instalação anterior
            if install_dir.exists():
                shutil.rmtree(install_dir)
            
            # Cria diretório e extrai arquivos
            install_dir.mkdir(parents=True)
            with zipfile.ZipFile(output_file, 'r') as zf:
                zf.extractall(install_dir)
            
            logger.info(f"Extensão instalada em: {install_dir}")
            
            print("\nInstalação concluída!")
            print("1. Feche COMPLETAMENTE o Aseprite")
            print("2. Abra o Aseprite novamente")
            print("3. A extensão deve aparecer em File > Scripts > Lunaris Orion")
        
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default="2.0.0")
    parser.add_argument('--install', action='store_true')
    
    args = parser.parse_args()
    build_extension(args.version, args.install) 