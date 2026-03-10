# Rede Neural RAM (WiSARD)

## O que é
Implementação da Rede Neural RAM no modelo WiSARD (Wilkie,
Stonham and Aleksander's Recognition Device) para classificação
binária, treinada para detectar falhas em equipamentos industriais
com base na leitura de 5 sensores binários.

## Como funciona
A RAM é uma rede neural sem pesos (Weightless Neural Network).
Em vez de ajustar parâmetros, ela memoriza os padrões de entrada
durante o treinamento. As entradas binárias são divididas em grupos
de bits chamados tuplas. Para cada grupo, o endereço formado pelos
bits é registrado na memória da classe correspondente.

Na classificação, o modelo consulta as memórias de cada classe e
conta quantas respondem positivamente ao endereço apresentado.
Esse total é chamado de score. A classe com o maior score é
escolhida como a predição final.

## Pré-requisitos
- Python 3.x
- VS Code com a extensão Python instalada
- Bibliotecas: numpy, pandas

## Instalação das dependências
Abra o terminal no VS Code e execute:

    pip install numpy pandas

## Como executar
1. Abra a pasta do projeto no VS Code
2. Abra o arquivo ram.py
3. Execute o script pelo terminal:

    python ram.py

O dataset dataset.csv já está incluso na pasta do projeto
e será carregado automaticamente.

## Parâmetros
| Parâmetro  | Valor padrão | Descrição                        |
|------------|--------------|----------------------------------|
| tuple_size | 2            | Quantidade de bits por grupo/RAM |

## Saída esperada
```
========================================
   REDE NEURAL RAM (WiSARD)
========================================
  Acurácia         : 50.00%
  Tempo treinamento: 0.0002s
  Amostras         : 32
  Bits de entrada  : 5
  Tuple size       : 2
  Nº de grupos/RAMs: 3
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

## Observação
O valor de tuple_size influencia diretamente a acurácia do modelo.
Com tuple_size = 2 e apenas 5 bits de entrada, o número de
endereços possíveis por RAM é reduzido, aumentando a chance de
colisão entre padrões de classes diferentes. Para este dataset,
tuple_size = 5 resultaria em endereços únicos por amostra,
eliminando as colisões.