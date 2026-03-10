import pandas as pd
import numpy as np
import time

# Upload e leitura do dataset
df = pd.read_csv("dataset.csv")

# Separa features e rótulo
X = df.drop("falha", axis=1).values
y = df["falha"].values


class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr          # taxa de aprendizado
        self.epochs = epochs  # número de épocas de treino
        self.w = None         # pesos das features
        self.b = 0.0          # bias

    def fit(self, X, y):
        # Inicializa pesos zerados
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        # Converte rótulos para -1 e 1 (padrão do Perceptron)
        labels = np.where(y == 0, -1, 1)

        for _ in range(self.epochs):
            for xi, yi in zip(X, labels):
                pred = np.sign(np.dot(xi, self.w) + self.b)
                if pred == 0:
                    pred = -1
                # Atualiza pesos apenas quando erra
                if pred != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi

    def predict(self, X):
        # Retorna 0 ou 1 com base no sinal da combinação linear
        raw = np.dot(X, self.w) + self.b
        return np.where(raw >= 0, 1, 0)


# Treino com todos os dados (dataset pequeno, sem divisão treino/teste)
modelo = Perceptron(lr=0.1, epochs=100)

inicio = time.perf_counter()
modelo.fit(X, y)
tempo_treino = time.perf_counter() - inicio

# Avaliação
y_pred = modelo.predict(X)
acuracia = (y_pred == y).mean() * 100

print("=" * 40)
print("   PERCEPTRON SIMPLES")
print("=" * 40)
print(f"  Acurácia         : {acuracia:.2f}%")
print(f"  Tempo treinamento: {tempo_treino:.4f}s")
print(f"  Amostras         : {len(X)}")
print(f"  Features         : {X.shape[1]}")
print(f"  Taxa aprendizado : {modelo.lr}")
print(f"  Épocas           : {modelo.epochs}")
print("=" * 40)