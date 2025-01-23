# 🎨 Mini Model - Gerador de Pixel Art 16x16

<div align="center">

![Mini Model Logo](https://raw.githubusercontent.com/meryyllebr/mini-model/main/docs/images/logo.png)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## 📖 Sobre o Projeto

Mini Model é um gerador de pixel art 16x16 baseado em deep learning, otimizado para criar sprites de alta qualidade com controle de estilo. O modelo utiliza uma arquitetura transformer moderna com várias otimizações para performance tanto em GPU quanto CPU.

<div align="center">
<img src="https://raw.githubusercontent.com/meryyllebr/mini-model/main/docs/images/samples.png" width="600px"/>
</div>

### ✨ Características Principais

- 🖼️ Geração de sprites 16x16 de alta qualidade
- 🎯 Controle preciso de estilo através de labels
- ⚡ Otimizado para performance em GPU e CPU
- 🔄 Suporte a treinamento distribuído
- 📊 Monitoramento detalhado com Weights & Biases
- 🎨 Paleta de cores otimizada para pixel art

## 🚀 Começando

### Pré-requisitos

- Python 3.11+
- PyTorch 2.4+
- CUDA (opcional, para treinamento em GPU)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/mini-model.git
cd mini-model
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Treinamento

Para treinar o modelo:

```bash
python -m mini_model.train --data_dir caminho/para/dataset
```

Para treinamento em modo desenvolvimento (CPU):

```bash
python -m mini_model.train --data_dir caminho/para/dataset --dev
```

### Configuração

O modelo pode ser configurado através do arquivo `mini_model/configs/base_config.yaml`:

```yaml
model:
  image_size: 16
  embedding_dim: 1024
  num_heads: 16
  num_layers: 12
  ...

training:
  batch_size: 64
  learning_rate: 2e-4
  num_epochs: 5
  ...
```

## 📊 Monitoramento

O Mini Model utiliza Weights & Biases para monitoramento detalhado do treinamento:

<div align="center">
<img src="https://raw.githubusercontent.com/meryyllebr/mini-model/main/docs/images/wandb.png" width="800px"/>
</div>

### Métricas Monitoradas

- 📈 Loss de treinamento e validação
- 🎯 Qualidade das imagens geradas
- 💾 Uso de memória e GPU
- ⚡ Throughput de treinamento
- 🎨 Consistência de estilo

## 🏗️ Arquitetura

O modelo utiliza uma arquitetura transformer moderna com várias otimizações:

<div align="center">
<img src="https://raw.githubusercontent.com/meryyllebr/mini-model/main/docs/images/architecture.png" width="700px"/>
</div>

### Componentes Principais

- 🔄 Transformer com atenção otimizada
- 🎯 ALiBi para melhor generalização
- 🚀 Flash Attention para performance
- 🎨 Quantizador de cores adaptativo
- 📊 Loss multi-objetivo

## 📁 Estrutura do Projeto

```
mini_model/
├── configs/          # Arquivos de configuração
├── core/            # Implementação principal do modelo
├── data/            # Processamento de dados
├── training/        # Lógica de treinamento
├── utils/           # Utilitários
└── tests/           # Testes unitários
```

## 📊 Resultados

Exemplos de sprites gerados pelo modelo:

<div align="center">
<img src="https://raw.githubusercontent.com/meryyllebr/mini-model/main/docs/images/results.png" width="800px"/>
</div>

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, leia nosso guia de contribuição antes de submeter um PR.

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ✨ Agradecimentos

- [Weights & Biases](https://wandb.ai/) pelo suporte ao monitoramento
- [PyTorch](https://pytorch.org/) pela excelente framework
- Toda a comunidade de pixel art pelo feedback e suporte

## 📧 Contato

Para questões e sugestões, por favor abra uma issue ou entre em contato através do email: seu-email@exemplo.com

---
<div align="center">
Feito com ❤️ pela equipe Mini Model
</div> 
