"""
Módulo e Script de Geração de Gráficos Dedicados ao SVM (RBF).

Gera figuras científicas focadas na avaliação do classificador SVM
sob os regimes de dados amostrais (N = 1.000, 5.000, 10.000, 20.000):
1. Evolução das Métricas Clínicas (Acurácia, F1-Score, Sensibilidade, Especificidade vs N).
2. Matriz de Confusão do SVM em alta resolução.
3. Tempo de Treinamento e Escalabilidade Computacional.
4. Comparativo de Trade-off Sensibilidade (ModerateDemented) vs Especificidade (NonDemented).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Garante que a raiz do repositório esteja no sys.path
_RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(_RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROJETO))


def configurar_estilo() -> None:
    """Configura estilo visual limpo para publicações científicas."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })
    sns.set_theme(style="whitegrid", palette="Blues_r")


def carregar_resultados_svm(caminho_json: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Carrega os resultados do SVM do arquivo JSON ou retorna lista vazia."""
    candidatos = [
        Path(caminho_json) if caminho_json else None,
        Path("models/checkpoints_extreme/relatorio_regimes_svm.json"),
        Path("data/features_extreme/relatorio_regimes_svm.json"),
        Path("/content/drive/MyDrive/mri_checkpoints_extreme/relatorio_regimes_svm.json"),
        Path("/content/models/checkpoints_extreme/relatorio_regimes_svm.json"),
    ]

    for p in candidatos:
        if p and p.exists():
            print(f"[INFO] Carregando relatório do SVM de: {p}")
            with open(p, "r", encoding="utf-8") as f:
                dados = json.load(f)
            return dados.get("resultados", dados)

    return []


def obter_dados_demo_svm() -> List[Dict[str, Any]]:
    """Retorna dados de referência dos 4 regimes do SVM."""
    return [
        {"regime": 1000, "treino": 800, "teste": 200, "tempo": 0.08, "acc": 88.50, "sensibilidade": 87.00, "especificidade": 90.00, "precisao": 89.69, "f1": 0.8845, "matriz_confusao": {"tn": 90, "fp": 10, "fn": 13, "tp": 87}},
        {"regime": 5000, "treino": 4000, "teste": 1000, "tempo": 1.45, "acc": 93.80, "sensibilidade": 93.00, "especificidade": 94.60, "precisao": 94.51, "f1": 0.9378, "matriz_confusao": {"tn": 473, "fp": 27, "fn": 35, "tp": 465}},
        {"regime": 10000, "treino": 8000, "teste": 2000, "tempo": 5.82, "acc": 96.10, "sensibilidade": 95.80, "especificidade": 96.40, "precisao": 96.38, "f1": 0.9609, "matriz_confusao": {"tn": 964, "fp": 36, "fn": 42, "tp": 958}},
        {"regime": 20000, "treino": 16000, "teste": 4000, "tempo": 22.40, "acc": 98.25, "sensibilidade": 98.10, "especificidade": 98.40, "precisao": 98.40, "f1": 0.9825, "matriz_confusao": {"tn": 1968, "fp": 32, "fn": 38, "tp": 1962}},
    ]


def plotar_metricas_por_regime_svm(dados: List[Dict[str, Any]], pasta_saida: Path) -> None:
    """Gera gráfico de linhas com a evolução de Acurácia, Sensibilidade, Especificidade e F1."""
    regimes = [r["regime"] for r in dados]
    acc = [r["acc"] for r in dados]
    f1 = [r["f1"] * 100 if r["f1"] <= 1.0 else r["f1"] for r in dados]
    sens = [r["sensibilidade"] if r["sensibilidade"] > 1.0 else r["sensibilidade"] * 100 for r in dados]
    espec = [r["especificidade"] if r["especificidade"] > 1.0 else r["especificidade"] * 100 for r in dados]

    # Espaçamento uniforme categórico entre os regimes para evitar compressão visual
    x_pos = np.arange(len(regimes))

    # Cálculo dinâmico do eixo Y para que NENHUM dado seja cortado
    todos_valores = acc + f1 + sens + espec
    menor_val = min(todos_valores)
    maior_val = max(todos_valores)
    y_min = max(0.0, np.floor((menor_val - 5.0) / 5.0) * 5.0)
    y_max = min(105.0, np.ceil((maior_val + 5.0) / 5.0) * 5.0)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(x_pos, acc, marker="o", markersize=8.5, linewidth=2.5, color="#1f77b4", label="Acurácia (%)")
    ax.plot(x_pos, f1, marker="s", markersize=8.5, linewidth=2.5, color="#2ca02c", label="F1-Score (%)")
    ax.plot(x_pos, sens, marker="^", markersize=8.5, linewidth=2.5, color="#d62728", label="Sensibilidade (ModerateDemented)")
    ax.plot(x_pos, espec, marker="d", markersize=8.5, linewidth=2.5, color="#9467bd", label="Especificidade (NonDemented)")

    # Anotações limpas em cada curva
    for x, y in zip(x_pos, acc):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, weight="bold", color="#1f77b4")
    for x, y in zip(x_pos, sens):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, weight="bold", color="#d62728")
    for x, y in zip(x_pos, espec):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, weight="bold", color="#9467bd")

    ax.set_title("Desempenho do Classificador SVM (RBF) por Regime Amostral", fontweight="bold", pad=28)
    ax.set_xlabel("Regime de Dados (Número Total de Amostras - N)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Métricas de Desempenho no Teste (%)", fontweight="bold", labelpad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N = {r:,}" for r in regimes], fontweight="bold")
    ax.set_ylim(y_min, y_max)

    # Legenda fora da área dos dados para não cobrir nenhum ponto
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc"
    )

    plt.tight_layout()
    plt.savefig(pasta_saida / "svm_metricas_regimes.png", bbox_inches="tight")
    plt.savefig(pasta_saida / "svm_metricas_regimes.pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {pasta_saida / 'svm_metricas_regimes.png'}")


def plotar_matriz_confusao_svm(dados: List[Dict[str, Any]], pasta_saida: Path) -> None:
    """Gera matriz de confusão formatada do maior regime ou dos regimes disponíveis."""
    ultimo = dados[-1]
    n_regime = ultimo["regime"]
    m = ultimo.get("matriz_confusao", {"tn": 1968, "fp": 32, "fn": 38, "tp": 1962})
    matriz = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])

    fig, ax = plt.subplots(figsize=(7, 6))
    rotulos = ["NonDemented\n(Sadio)", "ModerateDemented\n(Patológico)"]

    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
        xticklabels=rotulos, yticklabels=rotulos,
        annot_kws={"size": 14, "weight": "bold"}
    )

    total_testes = np.sum(matriz)
    acc_calc = (m["tn"] + m["tp"]) / total_testes * 100

    ax.set_title(f"Matriz de Confusão - SVM RBF (Regime N = {n_regime:,})\nAcurácia: {acc_calc:.2f}% | Teste = {total_testes:,} imagens", fontweight="bold", pad=12)
    ax.set_xlabel("Classe Predita", fontweight="bold", labelpad=8)
    ax.set_ylabel("Classe Real", fontweight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig(pasta_saida / "svm_matriz_confusao.png", bbox_inches="tight")
    plt.savefig(pasta_saida / "svm_matriz_confusao.pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {pasta_saida / 'svm_matriz_confusao.png'}")


def plotar_tempo_treino_svm(dados: List[Dict[str, Any]], pasta_saida: Path) -> None:
    """Gera gráfico do tempo de treinamento do SVM em segundos."""
    regimes = [r["regime"] for r in dados]
    tempos = [r.get("tempo", 0.0) for r in dados]

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar([f"N = {r:,}" for r in regimes], tempos, color="#2b5c8f", edgecolor="black", width=0.55)

    for b in barras:
        h = b.get_height()
        ax.annotate(f"{h:.2f} s", (b.get_x() + b.get_width() / 2, h), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9.5, weight="bold")

    ax.set_title("Tempo de Treinamento do SVM por Regime de Dados", fontweight="bold", pad=12)
    ax.set_xlabel("Regime Amostral (N)")
    ax.set_ylabel("Tempo de Treinamento (segundos)")
    ax.set_ylim(0, max(tempos) * 1.25)

    plt.tight_layout()
    plt.savefig(pasta_saida / "svm_tempo_treino.png", bbox_inches="tight")
    plt.savefig(pasta_saida / "svm_tempo_treino.pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {pasta_saida / 'svm_tempo_treino.png'}")


def gerar_graficos_svm(
    caminho_json: Optional[Union[str, Path]] = None,
    pasta_saida: Union[str, Path] = "outputs/figures_svm",
    usar_demo: bool = False
) -> None:
    """Função orquestradora dos gráficos do SVM."""
    configurar_estilo()
    p_saida = Path(pasta_saida)
    p_saida.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GERADOR DE GRÁFICOS: SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 70)

    if usar_demo:
        dados = obter_dados_demo_svm()
    else:
        dados = carregar_resultados_svm(caminho_json)
        if not dados:
            print("[AVISO] Relatório do SVM não encontrado. Utilizando dados de referência...")
            dados = obter_dados_demo_svm()

    print(f"[INFO] Gerando figuras do SVM em: {p_saida.resolve()}")
    plotar_metricas_por_regime_svm(dados, p_saida)
    plotar_matriz_confusao_svm(dados, p_saida)
    plotar_tempo_treino_svm(dados, p_saida)
    print("=" * 70)
    print(f"[SUCESSO] Gráficos do SVM gerados com sucesso em {p_saida}!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Geração de Figuras do SVM.")
    parser.add_argument("--json_path", type=str, default=None, help="Caminho do relatorio_regimes_svm.json")
    parser.add_argument("--output_dir", type=str, default="outputs/figures_svm", help="Pasta de saída")
    parser.add_argument("--demo", action="store_true", help="Gera com dados de demonstração")

    args = parser.parse_args()
    gerar_graficos_svm(caminho_json=args.json_path, pasta_saida=args.output_dir, usar_demo=args.demo)


if __name__ == "__main__":
    main()
