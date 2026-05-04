import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def gerar():
    # =============================================================================
    # 1. CARREGA OS RESULTADOS DE CADA CLASSIFICADOR
    # =============================================================================
    classificadores = {
        "Naive Bayes"   : "resultado_bayes.csv",
        "Perceptron"    : "resultado_perceptron.csv",
        "MLP"           : "resultado_mlp.csv",
        "SVM"           : "resultado_svm.csv",
        "Decision Tree" : "resultado_decision_tree.csv",
    }

    medias  = {}
    desvios = {}

    for nome, arquivo in classificadores.items():
        df = pd.read_csv(arquivo)
        medias[nome] = df["acuracia"].mean() * 100
        desvios[nome] = df["acuracia"].std() * 100

    nomes    = list(classificadores.keys())
    cores_clf = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # =============================================================================
    # 2. GRÁFICO 1 — Barras: Acurácia por classificador
    # =============================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    x      = np.arange(len(nomes))
    vals   = [medias[nome] for nome in nomes]
    erros  = [desvios[nome] for nome in nomes]
    bars   = ax.bar(x, vals, color=cores_clf, alpha=0.85,
                    yerr=erros, capsize=5, error_kw={"elinewidth": 1.5})

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=11)
    ax.set_ylabel("Acurácia (%)", fontsize=12)
    ax.set_title("Comparação de Classificadores — Acurácia Média (3 Rodadas)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig("grafico_comparacao.png", dpi=150)
    plt.close()
    print("grafico_comparacao.png salvo")

    # =============================================================================
    # 3. GRÁFICO 2 — Evolução de acurácia nas 3 rodadas
    # =============================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    rodadas = [1, 2, 3]

    for nome, cor in zip(nomes, cores_clf):
        df   = pd.read_csv(classificadores[nome])
        vals = df["acuracia"] * 100
        ax.plot(rodadas, vals, marker="o", label=nome, color=cor, linewidth=2.5, markersize=8)

    ax.set_title("Evolução de Acurácia por Rodada — Todos os Classificadores", fontsize=13, fontweight="bold")
    ax.set_xticks(rodadas)
    ax.set_xticklabels(["Rodada 1\n(seed=42)", "Rodada 2\n(seed=38)", "Rodada 3\n(seed=123)"], fontsize=11)
    ax.set_ylabel("Acurácia (%)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=11, loc="best")
    ax.grid(linestyle="--", alpha=0.5)
    ax.set_ylim(70, 95)
    fig.tight_layout()
    plt.savefig("grafico_evolucao.png", dpi=150)
    plt.close()
    print("grafico_evolucao.png salvo")

   

    print("\nTodos os gráficos gerados com sucesso!")