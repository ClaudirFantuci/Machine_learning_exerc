import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import classificador_bayes
import classificador_perceptron
import classificador_mlp
import classificador_svm
import classificador_decision_tree
import graficos

print("=" * 60)
print("     BANK MARKETING - LIMPEZA E PRÉ-PROCESSAMENTO")
print("=" * 60)

# =============================================================================
# 1. CARREGAMENTO
# =============================================================================
df = pd.read_csv("bank-additional-full.csv", sep=";")

print(f"\n[1] Dataset carregado")
print(f"    Linhas  : {df.shape[0]}")
print(f"    Colunas : {df.shape[1]}")

# =============================================================================
# 2. LIMPEZA — remove linhas com "unknown" (missing disfarçado)
# =============================================================================
linhas_antes = df.shape[0]

df.replace("unknown", np.nan, inplace=True)
df.dropna(inplace=True)

linhas_depois    = df.shape[0]
linhas_removidas = linhas_antes - linhas_depois

print(f"\n[2] Limpeza de valores ausentes ('unknown' → NaN → dropna)")
print(f"    Linhas antes   : {linhas_antes}")
print(f"    Linhas depois  : {linhas_depois}")
print(f"    Removidas      : {linhas_removidas} ({linhas_removidas/linhas_antes*100:.2f}%)")

# =============================================================================
# 3. REMOVE COLUNA DURATION (data leakage)
# =============================================================================
df.drop(columns=["duration"], inplace=True)
print(f"\n[3] Coluna 'duration' removida. Colunas restantes: {df.shape[1]}")

# =============================================================================
# 4. LABEL ENCODING — categóricas → numéricas
#    Usando include=["object", "string"] para compatibilidade com Pandas 3+
# =============================================================================
colunas_categoricas = df.select_dtypes(include=["object", "string"]).columns.tolist()
print(f"\n[4] Encoding de colunas categóricas ({len(colunas_categoricas)}): {colunas_categoricas}")

le = LabelEncoder()
for col in colunas_categoricas:
    df[col] = le.fit_transform(df[col])

print("    Encoding concluído. Todos os tipos agora são numéricos.")

# =============================================================================
# 5. SALVA O DATASET PROCESSADO
# =============================================================================
df.to_csv("bank_processado.csv", index=False)
print(f"\n[5] bank_processado.csv salvo ({df.shape[0]} linhas, {df.shape[1]} colunas)")
print("    Shuffle e split serão feitos em cada classificador")
print("    com as seeds fixas [42, 7, 123].")
print("=" * 60)

# =============================================================================
# 6. EXECUTA OS CLASSIFICADORES
# =============================================================================
classificador_bayes.executar()
classificador_perceptron.executar()
classificador_mlp.executar()
classificador_svm.executar()
classificador_decision_tree.executar()
graficos.gerar()