#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Iniciando Lunaris Orion Bot em modo desenvolvimento...${NC}"

# Verificar arquivo .env
if [ ! -f .env ]; then
    echo -e "${YELLOW}Arquivo .env não encontrado. Criando a partir do exemplo...${NC}"
    cp .env.example .env
    echo "Por favor, configure as variáveis no arquivo .env antes de continuar."
    exit 1
fi

# Verificar dependências
echo -e "${GREEN}Verificando dependências...${NC}"
npm install

# Criar diretório de logs se não existir
mkdir -p logs

# Iniciar bot com nodemon e monitoramento de logs
echo -e "${GREEN}Iniciando bot...${NC}"
npx nodemon src/bot.js | tee -a logs/combined.log &

# Aguardar um momento para o bot iniciar
sleep 2

# Mostrar logs em tempo real
echo -e "${GREEN}Monitorando logs...${NC}"
tail -f logs/combined.log
