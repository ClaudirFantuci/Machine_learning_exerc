import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEEDS        = [42, 38, 123]
SPLIT_TREINO = 0.90

def executar():
    resultados = []

    print("\n" + "=" * 60)
    print("    SVM — 3 RODADAS")
    print("=" * 60)

    for i, seed in enumerate(SEEDS, start=1):
        df = pd.read_csv("bank_processado.csv")
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        X = df.drop(columns=["y"])
        y = df["y"]
        corte    = int(len(df) * SPLIT_TREINO)
        X_treino = X.iloc[:corte]
        y_treino = y.iloc[:corte]
        X_teste  = X.iloc[corte:]
        y_teste  = y.iloc[corte:]

        modelo = SVC(kernel="rbf", random_state=0)
        modelo.fit(X_treino, y_treino)
        predicoes = modelo.predict(X_teste)

        acc = accuracy_score(y_teste, predicoes)
        cm  = confusion_matrix(y_teste, predicoes)
        rep = classification_report(y_teste, predicoes, output_dict=True)

        resultados.append({
            "rodada"     : i,
            "seed"       : seed,
            "acuracia"   : acc,
            "precision_1": rep["1"]["precision"],
            "recall_1"   : rep["1"]["recall"],
            "f1_1"       : rep["1"]["f1-score"],
        })

        print(f"\n  Rodada {i} (seed={seed})")
        print(f"  Acurácia : {acc * 100:.2f}%")
        print(f"  Matriz de Confusão:")
        print(f"                   Previsto 0   Previsto 1")
        print(f"    Real 0 (não) :   {cm[0][0]:>6}       {cm[0][1]:>6}")
        print(f"    Real 1 (sim) :   {cm[1][0]:>6}       {cm[1][1]:>6}")
        print(f"  Classe 1 — Precision: {rep['1']['precision']:.4f}  "
              f"Recall: {rep['1']['recall']:.4f}  F1: {rep['1']['f1-score']:.4f}")

    df_res = pd.DataFrame(resultados)
    df_res.to_csv("resultado_svm.csv", index=False)

    print(f"\n  MÉDIA DAS 3 RODADAS")
    print(f"  Acurácia  : {df_res['acuracia'].mean()*100:.2f}%  (± {df_res['acuracia'].std()*100:.2f}%)")
    print(f"  Precision : {df_res['precision_1'].mean():.4f}  (± {df_res['precision_1'].std():.4f})")
    print(f"  Recall    : {df_res['recall_1'].mean():.4f}  (± {df_res['recall_1'].std():.4f})")
    print(f"  F1-score  : {df_res['f1_1'].mean():.4f}  (± {df_res['f1_1'].std():.4f})")
    print(f"  Resultados salvos em: resultado_svm.csv")