# Perceptron Simples

## O que é
Implementação do Perceptron Simples para classificação binária,
treinado para detectar falhas em equipamentos industriais com base
na leitura de 5 sensores binários.

## Como funciona
O Perceptron é um modelo de rede neural com uma única camada.
Ele recebe as entradas, multiplica cada uma pelo seu respectivo
peso, soma o bias e aplica a função de ativação sinal (sign).
Durante o treinamento, os pesos são ajustados sempre que o modelo
erra uma classificação, seguindo a regra:

    w = w + η × erro × x

O processo se repete por um número definido de épocas até o modelo
convergir.

## Pré-requisitos
- Python 3.x
- VS Code com a extensão Python instalada
- Bibliotecas: numpy, pandas

## Instalação das dependências
Abra o terminal no VS Code e execute:

    pip install numpy pandas

## Como executar
1. Abra a pasta do projeto no VS Code
2. Abra o arquivo perceptron.py
3. Execute o script pelo terminal:

    python perceptron.py

O dataset dataset.csv já está incluso na pasta do projeto
e será carregado automaticamente.

## Parâmetros
| Parâmetro | Valor padrão | Descrição                       |
|-----------|--------------|---------------------------------|
| lr        | 0.1          | Taxa de aprendizado             |
| epochs    | 100          | Número de épocas de treinamento |

## Saída esperada
```
========================================
   PERCEPTRON SIMPLES
========================================
  Acurácia         : 100.00%
  Tempo treinamento: 0.0176s
  Amostras         : 32
  Features         : 5
  Taxa aprendizado : 0.1
  Épocas           : 100
========================================
```

## Estrutura do dataset
| Coluna             | Tipo    | Descrição              |
|--------------------|---------|------------------------|
| sensor_temperatura | binário | Estado do sensor (0/1) |
| sensor_pressao     | binário | Estado do sensor (0/1) |
| sensor_vibracao    | binário | Estado do sensor (0/1) |
| sensor_corrente    | binário | Estado do sensor (0/1) |
| sensor_tensao      | binário | Estado do sensor (0/1) |
| falha              | binário | 0 = normal, 1 = falha  |