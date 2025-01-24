# Lunar Core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![JAX](https://img.shields.io/badge/JAX-0.5.0-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.10.2-green.svg)](https://github.com/google/flax)

Um modelo de rede neural baseado em JAX e Flax para geração de pixel art 16x16.

## Descrição

O **Lunar Core** é um modelo encoder-decoder com blocos residuais, otimizado para a geração de pixel art em resolução 16x16. Utilizando uma arquitetura VAE (Variational Autoencoder), o modelo possui as seguintes características:

- **Encoder-Decoder com blocos residuais**: Estrutura que facilita o aprendizado de características complexas.
- **Dimensão latente configurável**: Permite ajustar a complexidade do espaço latente.
- **Suporte a data augmentation específico para pixel art**: Melhora a robustez do modelo.
- **Treinamento otimizado com JAX e Flax**: Aproveita o poder de computação de GPUs NVIDIA e CPUs.
- **Logging com Weights & Biases**: Facilita o monitoramento e a análise do treinamento.
- **Compatibilidade com GPUs NVIDIA e CPUs**: Flexibilidade para diferentes ambientes de execução.

## Requisitos

- **Python 3.8+**
- **JAX 0.5.0**
- **Flax 0.10.2**
- Outras dependências listadas em [`requirements.txt`](requirements.txt)

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

1. Configure os parâmetros no arquivo [`config/config.yaml`](config/config.yaml).
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

- `sprites.npy`: Array NumPy com as imagens 16x16.
- `labels.csv`: Arquivo CSV com as descrições das labels.
- `sprites_labels.npy`: Array NumPy associando sprites às labels.

## Configuração

O arquivo [`config.yaml`](config/config.yaml) permite configurar:

- **Parâmetros do modelo**: Dimensão latente, filtros, etc.
- **Hiperparâmetros de treinamento**: Taxa de aprendizado, número de épocas, etc.
- **Configurações de hardware**: Uso de GPU ou CPU.
- **Opções de logging**: Configurações para o Weights & Biases.

## Monitoramento

O treinamento pode ser monitorado através do [Weights & Biases](https://wandb.ai/site), que registra:

- **Métricas de treinamento e validação**: Acompanhe o desempenho do modelo.
- **Amostras geradas durante o treinamento**: Visualize o progresso do modelo.
- **Configurações do experimento**: Documente as condições de execução.

## Contribuição

Contribuições são bem-vindas! Por favor, siga estas etapas:

1. **Fork** o projeto.
2. **Crie uma branch** para sua feature.
3. **Commit** suas mudanças.
4. **Push** para a branch.
5. Abra um **Pull Request**.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
```
