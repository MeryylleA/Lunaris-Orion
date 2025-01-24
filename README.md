# Lunar Core

Um modelo de rede neural baseado em JAX e Flax para geração de pixel art 16x16.

## Descrição

O Lunar Core é um modelo encoder-decoder com blocos residuais otimizado para geração de pixel art em resolução 16x16. O modelo utiliza uma arquitetura VAE (Variational Autoencoder) com as seguintes características:

- Encoder-Decoder com blocos residuais
- Dimensão latente configurável
- Suporte a data augmentation específico para pixel art
- Treinamento otimizado com JAX e Flax
- Logging com Weights & Biases
- Compatibilidade com GPUs NVIDIA e CPUs

## Requisitos

- Python 3.8+
- JAX 0.5.0
- Flax 0.10.2
- Outras dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/lunar-core.git
cd lunar-core
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

```
lunar_core/
├── config/
│   └── config.yaml        # Configurações gerais do projeto
├── data/
│   ├── dataset_loader.py  # Carregamento do dataset
│   └── augmentation.py    # Data augmentation
├── models/
│   └── lunar_core.py      # Definição do modelo
├── training/
│   └── train.py          # Script de treinamento
├── inference/
│   └── inference.py      # Script de inferência
├── logs/                 # Logs de treinamento
├── outputs/              # Imagens geradas
└── README.md
```

## Uso

### Treinamento

1. Configure os parâmetros no arquivo `config/config.yaml`
2. Execute o treinamento:
```bash
python training/train.py
```

### Inferência

Para gerar novas imagens usando um modelo treinado:
```bash
python inference/inference.py
```

## Dataset

O modelo espera um dataset com a seguinte estrutura:
- `sprites.npy`: Array numpy com as imagens 16x16
- `labels.csv`: Arquivo CSV com as descrições das labels
- `sprites_labels.npy`: Array numpy associando sprites às labels

## Configuração

O arquivo `config.yaml` permite configurar:
- Parâmetros do modelo (dimensão latente, filtros, etc.)
- Hiperparâmetros de treinamento
- Configurações de hardware
- Opções de logging

## Monitoramento

O treinamento pode ser monitorado através do Weights & Biases, que registra:
- Métricas de treinamento e validação
- Amostras geradas durante o treinamento
- Configurações do experimento

## Contribuição

Contribuições são bem-vindas! Por favor, siga estas etapas:
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes. 
