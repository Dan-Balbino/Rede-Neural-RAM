import pandas as pd
import numpy as np
import time

# Upload e leitura do dataset
df = pd.read_csv("dataset.csv")

# Separa features e rótulo (entradas já são binárias)
X = df.drop("falha", axis=1).values
y = df["falha"].values


class WiSARD:
    def __init__(self, tuple_size=2):
        self.tuple_size = tuple_size  # quantidade de bits por grupo
        self.grupos = []              # mapeamento de bits para grupos
        self.rams = {}                # memórias por classe
        self.classes_ = []

    def _build_grupos(self, n_bits):
        # Embaralha os índices dos bits e divide em grupos de tuple_size
        idx = list(range(n_bits))
        np.random.shuffle(idx)
        grupos = [idx[i:i + self.tuple_size]
                  for i in range(0, n_bits, self.tuple_size)]
        # Completa o último grupo se o número de bits não for divisível
        while len(grupos[-1]) < self.tuple_size:
            grupos[-1].append(np.random.randint(0, n_bits))
        return grupos

    def fit(self, X, y):
        self.classes_ = list(np.unique(y))
        self.grupos = self._build_grupos(X.shape[1])
        # Inicializa uma RAM (dicionário) por grupo para cada classe
        self.rams = {c: [{} for _ in self.grupos] for c in self.classes_}

        for xi, yi in zip(X, y):
            for g_idx, bits in enumerate(self.grupos):
                # O endereço é a combinação de bits do grupo
                endereco = tuple(xi[b] for b in bits)
                # Marca o endereço como visto durante o treino
                self.rams[yi][g_idx][endereco] = 1

    def predict(self, X):
        preds = []
        for xi in X:
            scores = {}
            for cls in self.classes_:
                score = 0
                for g_idx, bits in enumerate(self.grupos):
                    endereco = tuple(xi[b] for b in bits)
                    # Soma 1 se o endereço foi visto no treino, 0 caso contrário
                    score += self.rams[cls][g_idx].get(endereco, 0)
                scores[cls] = score
            # Classe com maior número de RAMs ativadas vence
            preds.append(max(scores, key=scores.get))
        return np.array(preds)


# Treino com todos os dados (dataset pequeno, sem divisão treino/teste)
np.random.seed(42)
modelo = WiSARD(tuple_size=2)

inicio = time.perf_counter()
modelo.fit(X, y)
tempo_treino = time.perf_counter() - inicio

# Avaliação
y_pred = modelo.predict(X)
acuracia = (y_pred == y).mean() * 100

print("=" * 40)
print("   REDE NEURAL RAM (WiSARD)")
print("=" * 40)
print(f"  Acurácia         : {acuracia:.2f}%")
print(f"  Tempo treinamento: {tempo_treino:.4f}s")
print(f"  Amostras         : {len(X)}")
print(f"  Bits de entrada  : {X.shape[1]}")
print(f"  Tuple size       : {modelo.tuple_size}")
print(f"  Nº de grupos/RAMs: {len(modelo.grupos)}")
print("=" * 40)