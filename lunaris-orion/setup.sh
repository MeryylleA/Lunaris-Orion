#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Iniciando setup do Lunaris Orion...${NC}"

# Verifica Ubuntu 22.04
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$VERSION_ID" != "22.04" ] || [ "$ID" != "ubuntu" ]; then
        echo -e "${RED}Este script requer Ubuntu 22.04${NC}"
        echo -e "${YELLOW}Sistema atual: $PRETTY_NAME${NC}"
        exit 1
    fi
else
    echo -e "${RED}Não foi possível determinar a versão do sistema operacional${NC}"
    exit 1
fi

# Verifica CUDA 12.4
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
if [[ "$CUDA_VERSION" != "12.4"* ]]; then
    echo -e "${RED}CUDA 12.4 é necessário. Versão atual: $CUDA_VERSION${NC}"
    exit 1
fi

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 não encontrado. Por favor, instale o Python 3.${NC}"
    exit 1
fi

# Verifica pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip3 não encontrado. Por favor, instale o pip3.${NC}"
    exit 1
fi

# Verifica GPUs NVIDIA
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA drivers não encontrados. GPUs NVIDIA são necessárias.${NC}"
    exit 1
fi

# Verifica número de GPUs
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ "$GPU_COUNT" -ne 2 ]; then
    echo -e "${RED}Este script requer exatamente 2 GPUs NVIDIA. Encontradas: $GPU_COUNT${NC}"
    exit 1
fi

# Verifica memória do sistema
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -lt 75 ]; then  # 75GB mínimo (considerando margem)
    echo -e "${RED}Mínimo de 80GB de RAM necessário. Disponível: ${TOTAL_MEM}GB${NC}"
    exit 1
fi

# Verifica CPUs
CPU_CORES=$(nproc)
if [ "$CPU_CORES" -lt 24 ]; then  # 24 cores mínimo (considerando margem)
    echo -e "${RED}Mínimo de 26 vCores necessário. Disponível: ${CPU_CORES}${NC}"
    exit 1
fi

# Verifica e instala dependências do sistema
echo -e "${YELLOW}Verificando dependências do sistema...${NC}"
DEPS=(
    "build-essential"
    "python3-dev"
    "python3-pip"
    "python3-venv"
    "postgresql"
    "postgresql-contrib"
    "redis-server"
    "nginx"
    "git"
)

for dep in "${DEPS[@]}"; do
    if ! dpkg -l | grep -q "^ii  $dep "; then
        echo -e "${YELLOW}Instalando $dep...${NC}"
        sudo apt-get install -y "$dep"
    fi
done

# Cria ambiente virtual se não existir
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Criando ambiente virtual...${NC}"
    python3 -m venv venv
fi

# Ativa ambiente virtual
source venv/bin/activate

# Atualiza pip e instala dependências básicas
pip install --upgrade pip
pip install python-dotenv psycopg2-binary

# Verifica se .env existe e oferece configuração
if [ ! -f ".env" ] || [ "$1" == "--configure" ]; then
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}Arquivo .env não encontrado. Iniciando configuração...${NC}"
    else
        echo -e "${YELLOW}Reconfigurando variáveis de ambiente...${NC}"
    fi
    
    python setup.py --configure
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Erro na configuração das variáveis de ambiente.${NC}"
        exit 1
    fi
    
    if [ "$1" == "--configure" ]; then
        echo -e "${GREEN}Configuração concluída com sucesso!${NC}"
        exit 0
    fi
fi

# Configura limites do sistema
echo -e "${YELLOW}Configurando limites do sistema...${NC}"
if ! grep -q "* soft nofile 65535" /etc/security/limits.conf; then
    echo "* soft nofile 65535" | sudo tee -a /etc/security/limits.conf
    echo "* hard nofile 65535" | sudo tee -a /etc/security/limits.conf
fi

# Otimiza configurações do sistema
echo -e "${YELLOW}Otimizando configurações do sistema...${NC}"
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.dirty_ratio=60
sudo sysctl -w vm.dirty_background_ratio=2

# Executa setup.py
echo -e "${GREEN}Executando setup principal...${NC}"

# Verifica argumentos
ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-dataset)
            ARGS="$ARGS --skip-dataset"
            shift
            ;;
        --skip-training)
            ARGS="$ARGS --skip-training"
            shift
            ;;
        --only-services)
            ARGS="$ARGS --only-services"
            shift
            ;;
        --configure)
            # Já tratado acima
            shift
            ;;
        *)
            echo -e "${RED}Argumento desconhecido: $1${NC}"
            exit 1
            ;;
    esac
done

python setup.py $ARGS

# Verifica se ocorreu algum erro
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Setup concluído com sucesso!${NC}"
    echo -e "${GREEN}Comandos disponíveis:${NC}"
    echo -e "${YELLOW}./setup.sh --configure${NC}     - Configura variáveis de ambiente"
    echo -e "${YELLOW}./setup.sh --only-services${NC} - Inicia apenas os serviços"
    echo -e "${YELLOW}./setup.sh --skip-dataset${NC}  - Pula download do dataset"
    echo -e "${YELLOW}./setup.sh --skip-training${NC} - Pula fine-tuning"
else
    echo -e "${RED}Erro durante o setup. Verifique os logs em setup.log${NC}"
    exit 1
fi 