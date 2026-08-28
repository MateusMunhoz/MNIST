# MNIST Digit Classifier (PyTorch)

Rede neural totalmente conectada (MLP) para reconhecimento de dígitos manuscritos do dataset **MNIST**, implementada em **PyTorch**.

##  Sobre o projeto

Este projeto treina uma rede neural simples (Multi-Layer Perceptron) para classificar imagens de dígitos de 0 a 9 do dataset MNIST, um dos datasets mais clássicos para introdução a redes neurais e visão computacional.

##  Arquitetura do modelo

A rede é composta por:

- **Flatten**: transforma a imagem 28x28 em um vetor de 784 posições
- **Linear(784 → 128)** + **ReLU**
- **Linear(128 → 10)**: camada de saída com um logit para cada dígito (0-9)

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

##  Configuração de treino

| Parâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Taxa de aprendizado | 1e-3 |
| Função de perda | CrossEntropyLoss |
| Batch size | 64 |
| Épocas | 5 |

##  Requisitos

- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib

Instale as dependências com:

```bash
pip install torch torchvision numpy matplotlib
```

## 🚀 Como executar

```bash
python main.py
```

O dataset MNIST é baixado automaticamente na primeira execução e salvo na pasta `./data`.

##  Estrutura do código

- `dataset_load()`: carrega e prepara os DataLoaders de treino e teste
- `MLP`: classe do modelo (arquitetura da rede)
- `check_Gpu()`: verifica disponibilidade de GPU e inicializa modelo, critério e otimizador
- `training_data()`: loop de treinamento do modelo
- `accuracy()`: avalia a acurácia do modelo no conjunto de teste



