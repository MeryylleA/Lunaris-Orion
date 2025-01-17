import os
import sys
import subprocess
import logging
import shutil
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Setup')

class SystemSetup:
    def __init__(self, dev_mode: bool = False):
        self.root_dir = Path(__file__).parent
        self.env_file = self.root_dir / '.env'
        self.dev_mode = dev_mode
        self.requirements = {
            'base': 'requirements.txt',
            'training': 'requirements-training.txt' if not dev_mode else None
        }
        
    def check_python_version(self) -> bool:
        """Verifica se a versão do Python é compatível"""
        logger.info("Verificando versão do Python...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("Python 3.8 ou superior é necessário")
            return False
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} encontrado")
        return True
    
    def check_gpu(self) -> Dict:
        """Verifica disponibilidade e especificações da GPU"""
        if self.dev_mode:
            logger.info("Modo dev: pulando verificação de GPU")
            return {'available': True, 'dev_mode': True}
            
        logger.info("Verificando GPU...")
        if not torch.cuda.is_available():
            logger.warning("CUDA não disponível! GPU é recomendada para produção")
            return {'available': False}
            
        gpu_info = {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'count': torch.cuda.device_count(),
            'memory': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
        
        logger.info(f"GPU detectada: {gpu_info['name']}")
        logger.info(f"Memória total: {gpu_info['memory']:.1f}GB")
        return gpu_info
    
    def create_directories(self) -> bool:
        """Cria estrutura de diretórios necessária"""
        logger.info("Criando estrutura de diretórios...")
        directories = [
            'logs',
            'src/database',
            'src/commands',
            'src/webhooks',
            'src/config'
        ]
        
        if not self.dev_mode:
            directories.extend([
                'models',
                'models/lunaris-pixel-art',
                'cache',
                'data',
                'src/training'
            ])
        
        try:
            for dir_path in directories:
                path = self.root_dir / dir_path
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Diretório criado: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao criar diretórios: {e}")
            return False
    
    def install_requirements(self) -> bool:
        """Instala todas as dependências necessárias"""
        logger.info("Instalando dependências...")
        
        for req_type, req_file in self.requirements.items():
            if req_file is None:
                continue
                
            logger.info(f"Instalando dependências {req_type}...")
            req_path = self.root_dir / req_file
            
            if not req_path.exists():
                logger.error(f"Arquivo {req_file} não encontrado")
                return False
                
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install',
                    '-r', str(req_path)
                ])
                logger.info(f"Dependências {req_type} instaladas com sucesso")
            except subprocess.CalledProcessError as e:
                logger.error(f"Erro ao instalar dependências {req_type}: {e}")
                return False
                
        return True
    
    def setup_env(self) -> bool:
        """Configura variáveis de ambiente"""
        logger.info("Configurando variáveis de ambiente...")
        
        if self.env_file.exists():
            logger.info(".env já existe, pulando configuração")
            return True
            
        try:
            env_template = {
                'DISCORD_TOKEN': 'seu_token_aqui',
                'DISCORD_CLIENT_ID': 'seu_client_id_aqui',
                'DISCORD_BOT_URL': 'http://localhost:3000',
                'DB_HOST': 'localhost',
                'DB_PORT': '5432',
                'DB_NAME': 'lunaris',
                'DB_USER': 'lunaris',
                'DB_PASSWORD': 'sua_senha_aqui',
                'STRIPE_SECRET_KEY': 'sk_test_...',
                'STRIPE_WEBHOOK_SECRET': 'whsec_...',
                'STRIPE_PREMIUM_PRODUCT_ID': 'prod_...',
                'STRIPE_PREMIUM_PRICE_ID': 'price_...'
            }
            
            if not self.dev_mode:
                env_template.update({
                    'GENERATION_API_URL': 'http://seu-servidor:8000',
                    'GENERATION_API_ADMIN_KEY': 'sua_chave_admin_aqui'
                })
            
            with open(self.env_file, 'w') as f:
                for key, value in env_template.items():
                    f.write(f"{key}={value}\n")
                    
            logger.info(".env criado com valores padrão")
            return True
        except Exception as e:
            logger.error(f"Erro ao criar .env: {e}")
            return False
    
    def setup_database(self) -> bool:
        """Configura o banco de dados PostgreSQL"""
        logger.info("Configurando banco de dados remoto...")
        
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            # Tenta conectar ao banco remoto
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'lunaris'),
                user=os.getenv('DB_USER', 'lunaris'),
                password=os.getenv('DB_PASSWORD', 'sua_senha_aqui')
            )
            
            logger.info("Conexão com banco remoto estabelecida")
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            
            # Executa migrações
            migrations_dir = self.root_dir / 'src/database/migrations'
            if migrations_dir.exists():
                for migration in sorted(migrations_dir.glob('*.sql')):
                    logger.info(f"Executando migração: {migration.name}")
                    try:
                        cur.execute(migration.read_text())
                        logger.info(f"Migração {migration.name} executada com sucesso")
                    except Exception as e:
                        logger.warning(f"Erro ao executar migração {migration.name}: {e}")
                        # Continua para próxima migração mesmo se houver erro
                        continue
            
            cur.close()
            conn.close()
            logger.info("Configuração do banco concluída")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao configurar banco de dados: {e}")
            return False
    
    def download_model(self) -> bool:
        """Baixa o modelo base Stable Diffusion 3.5"""
        if self.dev_mode:
            logger.info("Modo dev: pulando download do modelo")
            return True
            
        logger.info("Baixando modelo base...")
        try:
            from diffusers import StableDiffusionPipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            pipeline.save_pretrained(self.root_dir / "models/base")
            logger.info("Modelo base baixado com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao baixar modelo: {e}")
            return False
    
    def setup_all(self) -> bool:
        """Executa todo o processo de setup"""
        logger.info(f"Iniciando setup {'(modo dev)' if self.dev_mode else '(modo produção)'}...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Verifica requisitos
        if not self.check_python_version():
            return False
            
        gpu_info = self.check_gpu()
        if not self.dev_mode and not gpu_info['available']:
            return False
            
        # Cria estrutura
        if not self.create_directories():
            return False
            
        # Instala dependências
        if not self.install_requirements():
            return False
            
        # Configura ambiente
        if not self.setup_env():
            return False
            
        # Configura banco
        if not self.setup_database():
            return False
            
        # Baixa modelo (apenas em produção)
        if not self.download_model():
            return False
            
        # Salva informações do setup
        setup_info = {
            'timestamp': timestamp,
            'mode': 'development' if self.dev_mode else 'production',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'gpu_info': gpu_info,
            'directories_created': True,
            'requirements_installed': True,
            'env_configured': True,
            'database_configured': True,
            'model_downloaded': not self.dev_mode
        }
        
        with open(self.root_dir / 'setup_info.json', 'w') as f:
            json.dump(setup_info, f, indent=2)
            
        logger.info(f"Setup concluído com sucesso em modo {'desenvolvimento' if self.dev_mode else 'produção'}!")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup do sistema Lunaris Orion')
    parser.add_argument('--dev', action='store_true', help='Executa setup em modo desenvolvimento')
    args = parser.parse_args()
    
    setup = SystemSetup(dev_mode=args.dev)
    if not setup.setup_all():
        logger.error("Setup falhou!")
        sys.exit(1)
    sys.exit(0) 