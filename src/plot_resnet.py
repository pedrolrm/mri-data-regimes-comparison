"""
Módulo e Script de Geração de Gráficos Dedicados à ResNet-50 (RadImageNet).

Gera figuras científicas exclusivamente a partir de dados reais gerados
durante o treinamento convolucional da ResNet-50 (commits do Jonathan):
1. Evolução das Métricas por Regime Amostral (Acurácia, Sensibilidade, Especificidade, F1-Macro vs N).
2. Curvas de Treinamento por Época (Loss e Acurácia de Treino vs Validação).
3. Matriz de Confusão da ResNet-50 no regime N = 20.000.

IMPORTANTE: Opera estritamente com dados reais. Se o relatório ou logs
não forem localizados, interrompe a execução com FileNotFoundError.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

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
        "legend.fontsize": 10.5,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })
    sns.set_theme(style="whitegrid", palette="Reds_r")


def carregar_resultados_resnet(
    caminho_json: Optional[Union[str, Path]] = None,
    pasta_logs: Optional[Union[str, Path]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Carrega resultados reais da ResNet-50.
    Lança FileNotFoundError caso nenhum dado real seja encontrado.
    """
    candidatos_json = [
        Path(caminho_json) if caminho_json else None,
        Path("outputs/relatorio_regimes_resnet50_moderate.json"),
        Path("/content/drive/MyDrive/outputs/relatorio_regimes_resnet50_moderate.json"),
    ]

    for p in candidatos_json:
        if p and p.exists():
            print(f"[INFO] Carregando relatório real da ResNet-50 de: {p.resolve()}")
            with open(p, "r", encoding="utf-8") as f:
                dados = json.load(f)
            resultados = dados.get("resultados", dados)
            if resultados and isinstance(resultados, list):
                return resultados, []

    # Parsing a partir de logs em texto
    pasta_busca = Path(pasta_logs) if pasta_logs else Path("outputs/logs")
    if pasta_busca.exists():
        logs_txt = sorted(pasta_busca.glob("treinamento_resnet50_binario_*.txt"), reverse=True)
        if logs_txt:
            print(f"[INFO] Fazendo parsing do log de texto real: {logs_txt[0].resolve()}")
            res, epocas = parse_log_resnet_completo(logs_txt[0])
            if res:
                return res, epocas

    raise FileNotFoundError(
        "Não foi possível encontrar os resultados reais de treinamento da ResNet-50.\n"
        "Arquivos verificados:\n"
        "  - outputs/relatorio_regimes_resnet50_moderate.json\n"
        "  - outputs/logs/treinamento_resnet50_binario_*.txt\n\n"
        "Execute primeiro o treinamento da ResNet-50 (train_binary.py) para gerar os logs antes de criar os gráficos."
    )


def parse_log_resnet_completo(caminho_log: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extrai tanto a tabela final de regimes quanto o histórico de épocas do log de texto."""
    resultados_regimes = []
    historico_epocas = []

    padrao_regime = re.compile(
        r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)"
    )
    padrao_epoca = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)MB"
    )

    with open(caminho_log, "r", encoding="utf-8") as f:
        for linha in f:
            l = linha.strip()
            m_reg = padrao_regime.match(l)
            if m_reg:
                reg, tr, tempo, acc, sens, esp, f1 = m_reg.groups()
                resultados_regimes.append({
                    "regime": int(reg), "treino": int(tr), "tempo_s": float(tempo),
                    "acc": float(acc), "sensibilidade": float(sens), "especificidade": float(esp), "f1": float(f1)
                })
                continue

            m_ep = padrao_epoca.match(l)
            if m_ep:
                ep, tr_loss, tr_acc, val_loss, val_acc, val_f1, auc, tempo, vram = m_ep.groups()
                historico_epocas.append({
                    "epoca": int(ep), "train_loss": float(tr_loss), "train_acc": float(tr_acc),
                    "val_loss": float(val_loss), "val_acc": float(val_acc), "val_f1": float(val_f1),
                    "auc": float(auc), "tempo": float(tempo), "vram": float(vram)
                })

    return resultados_regimes, historico_epocas


def plotar_metricas_por_regime_resnet(dados: List[Dict[str, Any]], pasta_saida: Path) -> None:
    """Gera gráfico da evolução de métricas reais da ResNet-50 por regime."""
    regimes = [r["regime"] for r in dados]
    acc = [r["acc"] if r["acc"] > 1.0 else r["acc"] * 100 for r in dados]
    f1 = [r["f1"] * 100 if r["f1"] <= 1.0 else r["f1"] for r in dados]
    sens = [r["sensibilidade"] if r["sensibilidade"] > 1.0 else r["sensibilidade"] * 100 for r in dados]
    espec = [r["especificidade"] if r["especificidade"] > 1.0 else r["especificidade"] * 100 for r in dados]

    x_pos = np.arange(len(regimes))
    todos = acc + f1 + sens + espec
    y_min = max(0.0, np.floor((min(todos) - 5.0) / 5.0) * 5.0)
    y_max = min(105.0, np.ceil((max(todos) + 5.0) / 5.0) * 5.0)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(x_pos, acc, marker="s", markersize=8.5, linewidth=2.5, color="#d62728", label="Acurácia (%)")
    ax.plot(x_pos, f1, marker="o", markersize=8.5, linewidth=2.5, color="#ff7f0e", label="F1-Score (%)")
    ax.plot(x_pos, sens, marker="^", markersize=8.5, linewidth=2.5, color="#8c564b", label="Sensibilidade (ModerateDemented)")
    ax.plot(x_pos, espec, marker="d", markersize=8.5, linewidth=2.5, color="#e377c2", label="Especificidade (NonDemented)")

    for x, y in zip(x_pos, acc):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, weight="bold", color="#d62728")
    for x, y in zip(x_pos, sens):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, weight="bold", color="#8c564b")
    for x, y in zip(x_pos, espec):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, weight="bold", color="#e377c2")

    ax.set_title("Desempenho da ResNet-50 (RadImageNet) por Regime Amostral", fontweight="bold", pad=28)
    ax.set_xlabel("Regime de Dados (Número Total de Amostras - N)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Métricas de Desempenho no Teste (%)", fontweight="bold", labelpad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N = {r:,}" for r in regimes], fontweight="bold")
    ax.set_ylim(y_min, y_max)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc"
    )

    plt.tight_layout()
    plt.savefig(pasta_saida / "resnet_metricas_regimes.png", bbox_inches="tight")
    plt.savefig(pasta_saida / "resnet_metricas_regimes.pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {pasta_saida / 'resnet_metricas_regimes.png'}")


def plotar_curvas_aprendizado_resnet(epocas: List[Dict[str, Any]], pasta_saida: Path) -> None:
    """Gera gráficos de Loss e Acurácia por Época com indicação de Fases de Congelamento."""
    if not epocas:
        return

    num_epocas = [e["epoca"] for e in epocas]
    tr_loss = [e["train_loss"] for e in epocas]
    val_loss = [e["val_loss"] for e in epocas]
    tr_acc = [e["train_acc"] for e in epocas]
    val_acc = [e["val_acc"] for e in epocas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Loss
    ax1.plot(num_epocas, tr_loss, marker="o", linewidth=2.2, color="#1f77b4", label="Treino")
    ax1.plot(num_epocas, val_loss, marker="s", linewidth=2.2, color="#d62728", label="Validação")
    ax1.axvline(x=2.5, color="gray", linestyle=":", linewidth=1.8, label="Descongelamento (Fase 2)")
    ax1.set_title("Curva de Perda (Cross-Entropy Loss)", fontweight="bold")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Perda")
    ax1.legend(frameon=True)

    # Acurácia
    ax2.plot(num_epocas, tr_acc, marker="o", linewidth=2.2, color="#1f77b4", label="Treino")
    ax2.plot(num_epocas, val_acc, marker="s", linewidth=2.2, color="#d62728", label="Validação")
    ax2.axvline(x=2.5, color="gray", linestyle=":", linewidth=1.8, label="Descongelamento (Fase 2)")
    ax2.set_title("Curva de Acurácia (%)", fontweight="bold")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia (%)")
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("Convergência de Treinamento da ResNet-50 com Warmup e Fine-Tuning Diferencial", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(pasta_saida / "resnet_curvas_aprendizado.png", bbox_inches="tight")
    plt.savefig(pasta_saida / "resnet_curvas_aprendizado.pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {pasta_saida / 'resnet_curvas_aprendizado.png'}")


def plotar_matriz_confusao_resnet(dados: List[Dict[str, Any]], pasta_saida: Path) -> None:
    """Gera matriz de confusão da ResNet-50 no regime N = 20.000."""
    ultimo = dados[-1]
    n_regime = ultimo["regime"]
    m = ultimo.get("matriz_confusao")
    if not m:
        return

    matriz = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])

    fig, ax = plt.subplots(figsize=(7, 6))
    rotulos = ["NonDemented\n(Sadio)", "ModerateDemented\n(Patológico)"]

    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Reds", ax=ax, cbar=False,
        xticklabels=rotulos, yticklabels=rotulos,
        annot_kws={"size": 14, "weight": "bold"}
    )

    total_testes = np.sum(matriz)
    acc_calc = (m["tn"] + m["tp"]) / total_testes * 100

    ax.set_title(f"Matriz de Confusão - ResNet-50 (Regime N = {n_regime:,})\nAcurácia: {acc_calc:.2f}% | Teste = {total_testes:,} imagens", fontweight="bold", pad=12)
    ax.set_xlabel("Classe Predita", fontweight="bold", labelpad=8)
    ax.set_ylabel("Classe Real", fontweight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig(pasta_saida / "resnet_matriz_confusao.png", bbox_inches="tight")
    plt.savefig(pasta_saida / "resnet_matriz_confusao.pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {pasta_saida / 'resnet_matriz_confusao.png'}")


def gerar_graficos_resnet(
    caminho_json: Optional[Union[str, Path]] = None,
    pasta_logs: Optional[Union[str, Path]] = None,
    pasta_saida: Union[str, Path] = "outputs/figures_resnet"
) -> None:
    """Função orquestradora dos gráficos reais da ResNet-50."""
    configurar_estilo()
    p_saida = Path(pasta_saida)
    p_saida.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GERADOR DE GRÁFICOS: RESNET-50 (RADIMAGENET)")
    print("=" * 70)

    dados, epocas = carregar_resultados_resnet(caminho_json, pasta_logs)

    print(f"[INFO] Gerando figuras da ResNet-50 em: {p_saida.resolve()}")
    plotar_metricas_por_regime_resnet(dados, p_saida)
    plotar_matriz_confusao_resnet(dados, p_saida)
    plotar_curvas_aprendizado_resnet(epocas, p_saida)
    print("=" * 70)
    print(f"[SUCESSO] Gráficos da ResNet-50 gerados com sucesso em {p_saida}!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Geração de Figuras da ResNet-50.")
    parser.add_argument("--json_path", type=str, default=None, help="Caminho do relatorio_regimes_resnet50_moderate.json")
    parser.add_argument("--logs_dir", type=str, default="outputs/logs", help="Diretório de logs .txt")
    parser.add_argument("--output_dir", type=str, default="outputs/figures_resnet", help="Pasta de saída")

    args = parser.parse_args()
    gerar_graficos_resnet(caminho_json=args.json_path, pasta_logs=args.logs_dir, pasta_saida=args.output_dir)


if __name__ == "__main__":
    main()
