# Lunaris Orion

Sistema avançado de geração de pixel art baseado em IA, utilizando o modelo Stable Diffusion 3.5.

## Visão Geral

O Lunaris Orion é um sistema especializado em geração de pixel art que utiliza como base o modelo stabilityai/stable-diffusion-3.5-large com fine-tuning específico para pixel art. O sistema inclui:

- API para geração de imagens
- Extensão para Aseprite
- Bot do Discord
- Sistema de gerenciamento de usuários e assinaturas

## Requisitos do Sistema

- Python 3.10+
- CUDA 11.8+ (para GPU NVIDIA)
- 16GB+ RAM
- GPU com 12GB+ VRAM (recomendado)

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/lunaris-orion.git
cd lunaris-orion
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

## Estrutura do Projeto

```
lunaris-orion/
├── api/                # API REST
├── models/            # Modelos e pipelines
├── training/          # Scripts de treinamento
├── dataset/           # Processamento do dataset
├── aseprite_plugin/   # Extensão Aseprite
├── discord_bot/       # Bot do Discord
└── utils/             # Utilitários gerais
```

## Uso

### API

A API principal está disponível em `http://localhost:8000` após iniciar o servidor:

```bash
uvicorn api.main:app --reload
```

### Extensão Aseprite

1. Instale a extensão através do menu Extensions > Add Extension
2. Configure sua chave API nas configurações da extensão
3. Acesse as funcionalidades através do menu Sprite > Lunaris Orion

### Bot Discord

Configure o token do bot no arquivo .env e execute:

```bash
python -m discord_bot.main
```

## Monitoramento

O sistema inclui métricas Prometheus expostas em `/metrics` e logs detalhados em `/logs`.

## Licença

[Sua Licença] - Veja LICENSE.md para mais detalhes. 