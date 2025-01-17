#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Iniciando Lunaris Orion Discord Bot...${NC}"

# Verificar arquivo .env
if [ ! -f .env ]; then
    echo -e "${YELLOW}Arquivo .env não encontrado. Criando a partir do exemplo...${NC}"
    cp .env.example .env
    echo -e "${RED}Por favor, configure as variáveis no arquivo .env antes de continuar.${NC}"
    exit 1
fi

# Verificar dependências
echo -e "${GREEN}Verificando dependências...${NC}"
npm install

# Criar diretório de logs se não existir
mkdir -p logs

# Limpar logs antigos se necessário
if [ -f "logs/bot.log" ]; then
    echo -e "${YELLOW}Arquivando logs antigos...${NC}"
    mv logs/bot.log "logs/bot_$(date +%Y%m%d_%H%M%S).log"
fi

# Iniciar bot com nodemon
echo -e "${GREEN}Iniciando bot...${NC}"
npm run dev | tee logs/bot.log
