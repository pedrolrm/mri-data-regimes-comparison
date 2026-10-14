"""
Script de Geração de Gráficos Científicos para a Estratégia Full Unfreezing.

Gera figuras em alta resolução (300 DPI, PNG + PDF) em conformidade com o padrão IEEE:
1. 'full_unfreeze_metrics_regimes': Métricas (Acurácia, F1, Sensibilidade, Especificidade) vs N.
2. 'full_unfreeze_confusion_matrix': Matriz de confusão real do modelo Full Unfreezing.
3. 'full_unfreeze_learning_curves': Curvas de convergência época a época (Loss e Acurácia).
4. 'comparison_two_phase_vs_full_unfreeze': Gráfico de ablação comparativo direto entre
   Two-Phase Differential Fine-Tuning vs. Full Unfreezing Fine-Tuning.

Toda a formatação, títulos e eixos em inglês para publicação acadêmica.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Raiz do projeto no sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def set_academic_style() -> None:
    """Configura o estilo visual científico para publicação."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })
    sns.set_theme(style="whitegrid", palette="Purples_r")


def load_full_unfreeze_results(
    json_path: Optional[Union[str, Path]] = None,
    default_dir: Union[str, Path] = "outputs/full_unfreeze"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Carrega os resultados do treinamento com Full Unfreezing."""
    candidatos_json = [
        Path(json_path) if json_path else None,
        Path(default_dir) / "relatorio_regimes_resnet50_full_unfreeze_moderate.json",
        Path("outputs/full_unfreeze/relatorio_regimes_resnet50_full_unfreeze_moderate.json"),
        Path("outputs/relatorio_regimes_resnet50_full_unfreeze_moderate.json"),
    ]

    for p in candidatos_json:
        if p and p.exists():
            print(f"[INFO] Carregando resultados Full Unfreezing de: {p.resolve()}")
            with open(p, "r", encoding="utf-8") as f:
                content = json.load(f)
            resultados = content.get("resultados", content)
            if resultados and isinstance(resultados, list):
                # Extrai histórico de épocas do último regime se disponível
                historico = resultados[-1].get("historico_epocas", [])
                return resultados, historico

    tried = "\n".join([f"  - {str(p)}" for p in candidatos_json if p])
    raise FileNotFoundError(
        f"ERROR: Resultados de Full Unfreezing não encontrados.\n"
        f"Caminhos verificados:\n{tried}\n\n"
        f"Execute o treinamento primeiro: python src/train_full_unfreeze.py --task moderate --all_regimes"
    )


def load_two_phase_results() -> Optional[List[Dict[str, Any]]]:
    """Tenta carregar os resultados do Two-Phase Fine-Tuning para comparação."""
    candidatos = [
        Path("outputs/relatorio_regimes_resnet50_moderate.json"),
        Path("outputs/relatorio_regimes_resnet50_grouped.json"),
    ]
    for p in candidatos:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                content = json.load(f)
            res = content.get("resultados", content)
            if res and isinstance(res, list):
                return res
    return None


def plot_metrics_regimes(data: List[Dict[str, Any]], output_dir: Path) -> None:
    """Plota as métricas de desempenho sob Full Unfreezing idêntico ao estilo da imagem do SVM."""
    regimes = [r["regime"] for r in data]
    sens = [r["sensibilidade"] * 100 if r["sensibilidade"] <= 1.0 else r["sensibilidade"] for r in data]
    f1 = [r["f1_macro"] * 100 if r["f1_macro"] <= 1.0 else r["f1_macro"] for r in data]
    acc = [r["accuracy"] * 100 if r["accuracy"] <= 1.0 else r["accuracy"] for r in data]
    espec = [r["especificidade"] * 100 if r["especificidade"] <= 1.0 else r["especificidade"] for r in data]

    x_pos = np.arange(len(regimes))

    fig, ax = plt.subplots(figsize=(11, 6.8))

    # Curvas sólidas com marcadores idênticos ao padrão do SVM da imagem
    line_sens, = ax.plot(x_pos, sens, marker="^", markersize=9, linewidth=2.5, color="#d62728", label="Sensitivity (Moderate Demented)")
    line_f1, = ax.plot(x_pos, f1, marker="s", markersize=8.5, linewidth=2.5, color="#2ca02c", label="F1-Score (%)")
    line_acc, = ax.plot(x_pos, acc, marker="o", markersize=8.5, linewidth=2.5, color="#1f77b4", label="Accuracy (%)")
    line_espec, = ax.plot(x_pos, espec, marker="d", markersize=8.5, linewidth=2.5, color="#9467bd", label="Specificity (Non-Demented)")

    bbox_halo = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85)

    for i in range(len(regimes)):
        if i == 3:  # N = 20,000 (Acc e F1 são ambos 91.7%, Sens é 89.5%, Espec é 94.0%)
            ax.annotate(f"{espec[i]:.1f}%", (x_pos[i], espec[i]), textcoords="offset points", xytext=(0, 7), ha="center", va="bottom", fontsize=9, weight="bold", color="#9467bd", bbox=bbox_halo)
            ax.annotate(f"{sens[i]:.1f}%", (x_pos[i], sens[i]), textcoords="offset points", xytext=(0, -15), ha="center", va="top", fontsize=9, weight="bold", color="#d62728", bbox=bbox_halo)
            ax.annotate(f"{acc[i]:.1f}%", (x_pos[i], acc[i]), textcoords="offset points", xytext=(-10, 7), ha="right", va="bottom", fontsize=9, weight="bold", color="#1f77b4", bbox=bbox_halo)
            ax.annotate(f"{f1[i]:.1f}%", (x_pos[i], f1[i]), textcoords="offset points", xytext=(10, 7), ha="left", va="bottom", fontsize=9, weight="bold", color="#2ca02c", bbox=bbox_halo)
        else:  # N = 1,000, 5,000, 10,000 (métricas bem distribuídas)
            ax.annotate(f"{espec[i]:.1f}%", (x_pos[i], espec[i]), textcoords="offset points", xytext=(0, 7), ha="center", va="bottom", fontsize=9, weight="bold", color="#9467bd", bbox=bbox_halo)
            ax.annotate(f"{sens[i]:.1f}%", (x_pos[i], sens[i]), textcoords="offset points", xytext=(0, -15), ha="center", va="top", fontsize=9, weight="bold", color="#d62728", bbox=bbox_halo)
            ax.annotate(f"{acc[i]:.1f}%", (x_pos[i], acc[i]), textcoords="offset points", xytext=(0, 7), ha="center", va="bottom", fontsize=9, weight="bold", color="#1f77b4", bbox=bbox_halo)
            ax.annotate(f"{f1[i]:.1f}%", (x_pos[i], f1[i]), textcoords="offset points", xytext=(0, -15), ha="center", va="top", fontsize=9, weight="bold", color="#2ca02c", bbox=bbox_halo)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N = {r:,}" for r in regimes], fontweight="bold")
    ax.set_xlabel("Sample Regime (Total Sample Size - N)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Test Performance Metrics (%)", fontweight="bold", labelpad=10)
    ax.set_xlim(-0.25, len(regimes) - 0.75)
    ax.set_ylim(25, 102)

    ax.set_title("ResNet-50 (Full Unfreezing) Performance Across Data Regimes", fontweight="bold", pad=14, fontsize=13)

    # Legenda idêntica ao gráfico de referência: Coluna 1 (Sens, F1), Coluna 2 (Acc, Espec)
    ax.legend(
        [line_sens, line_f1, line_acc, line_espec],
        ["Sensitivity (Moderate Demented)", "F1-Score (%)", "Accuracy (%)", "Specificity (Non-Demented)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.savefig((output_dir / "full_unfreeze_metrics_regimes.png").as_posix(), bbox_inches="tight")
    plt.savefig((output_dir / "full_unfreeze_metrics_regimes.pdf").as_posix(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {output_dir / 'full_unfreeze_metrics_regimes.png'}")


def plot_confusion_matrix(data: List[Dict[str, Any]], output_dir: Path) -> None:
    """Plota a matriz de confusão real do modelo Full Unfreezing."""
    last = data[-1]
    n_regime = last["regime"]
    m = last.get("matriz_confusao", {})
    tn, fp, fn, tp = m.get("tn", 0), m.get("fp", 0), m.get("fn", 0), m.get("tp", 0)

    matrix = np.array([[tn, fp], [fn, tp]])
    total_test = tn + fp + fn + tp
    real_acc = (tn + tp) / total_test * 100.0 if total_test > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    class_labels = ["Non-Demented\n(Control)", "Moderate Demented\n(Pathological)"]

    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Purples", ax=ax, cbar=False,
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={"size": 14, "weight": "bold"}
    )

    ax.set_title(
        f"Full Unfreezing Confusion Matrix (N = {n_regime:,})\n"
        f"Accuracy: {real_acc:.2f}% | Test Samples: {total_test:,}",
        fontweight="bold", pad=12
    )
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig((output_dir / "full_unfreeze_confusion_matrix.png").as_posix(), bbox_inches="tight")
    plt.savefig((output_dir / "full_unfreeze_confusion_matrix.pdf").as_posix(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {output_dir / 'full_unfreeze_confusion_matrix.png'}")


def plot_learning_curves(epochs: List[Dict[str, Any]], output_dir: Path) -> None:
    """Plota as curvas de convergência sob Full Unfreezing."""
    if not epochs:
        return

    n_epochs = [e["epoca"] for e in epochs]
    tr_loss = [e["train_loss"] for e in epochs]
    val_loss = [e["val_loss"] for e in epochs]
    tr_acc = [e["train_acc"] for e in epochs]
    val_acc = [e["val_acc"] for e in epochs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Loss
    ax1.plot(n_epochs, tr_loss, marker="o", linewidth=2.2, color="#512da8", label="Training Loss")
    ax1.plot(n_epochs, val_loss, marker="s", linewidth=2.2, color="#d32f2f", label="Validation Loss")
    ax1.set_title("Cross-Entropy Loss Curve", fontweight="bold")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.legend(frameon=True)

    # Accuracy
    ax2.plot(n_epochs, tr_acc, marker="o", linewidth=2.2, color="#512da8", label="Training Accuracy")
    ax2.plot(n_epochs, val_acc, marker="s", linewidth=2.2, color="#d32f2f", label="Validation Accuracy")
    ax2.set_title("Accuracy Curve (%)", fontweight="bold")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("ResNet-50 Training Convergence (Full Unfreezing with Conservative LR)", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig((output_dir / "full_unfreeze_learning_curves.png").as_posix(), bbox_inches="tight")
    plt.savefig((output_dir / "full_unfreeze_learning_curves.pdf").as_posix(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {output_dir / 'full_unfreeze_learning_curves.png'}")


def plot_comparison_ablation(
    full_data: List[Dict[str, Any]],
    two_phase_data: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Plota a comparação direta de ablação:
    Two-Phase Differential Fine-Tuning vs. Full Unfreezing Fine-Tuning.
    """
    regimes = [r["regime"] for r in full_data]
    tp_dict = {r["regime"]: r for r in two_phase_data}

    # Interseção de regimes
    common_regimes = [r for r in regimes if r in tp_dict]
    if not common_regimes:
        return

    x_pos = np.arange(len(common_regimes))
    width = 0.35

    tp_acc = [tp_dict[r]["accuracy"] * 100 if tp_dict[r]["accuracy"] <= 1.0 else tp_dict[r]["accuracy"] for r in common_regimes]
    fu_dict = {r["regime"]: r for r in full_data}
    fu_acc = [fu_dict[r]["accuracy"] * 100 if fu_dict[r]["accuracy"] <= 1.0 else fu_dict[r]["accuracy"] for r in common_regimes]

    tp_f1 = [tp_dict[r]["f1_macro"] * 100 if tp_dict[r]["f1_macro"] <= 1.0 else tp_dict[r]["f1_macro"] for r in common_regimes]
    fu_f1 = [fu_dict[r]["f1_macro"] * 100 if fu_dict[r]["f1_macro"] <= 1.0 else fu_dict[r]["f1_macro"] for r in common_regimes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Acurácia Comparativa
    bars1 = ax1.bar(x_pos - width / 2, tp_acc, width, label="Two-Phase Differential (Warmup 2 Ep.)", color="#d32f2f", alpha=0.9)
    bars2 = ax1.bar(x_pos + width / 2, fu_acc, width, label="Full Unfreezing (Conservative LR)", color="#7b1fa2", alpha=0.9)
    ax1.set_title("Classification Accuracy Comparison (%)", fontweight="bold", pad=12)
    ax1.set_xlabel("Sample Regime (N)", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"N = {r:,}" for r in common_regimes], fontweight="bold")
    ax1.set_ylim(40, 105)
    ax1.legend(loc="lower right", frameon=True)

    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", (b.get_x() + b.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, weight="bold")
    for b in bars2:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", (b.get_x() + b.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, weight="bold")

    # Subplot 2: F1-Score Comparativo
    bars3 = ax2.bar(x_pos - width / 2, tp_f1, width, label="Two-Phase Differential (Warmup 2 Ep.)", color="#d32f2f", alpha=0.9)
    bars4 = ax2.bar(x_pos + width / 2, fu_f1, width, label="Full Unfreezing (Conservative LR)", color="#7b1fa2", alpha=0.9)
    ax2.set_title("Macro F1-Score Comparison (%)", fontweight="bold", pad=12)
    ax2.set_xlabel("Sample Regime (N)", fontweight="bold")
    ax2.set_ylabel("Macro F1-Score (%)", fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"N = {r:,}" for r in common_regimes], fontweight="bold")
    ax2.set_ylim(40, 105)
    ax2.legend(loc="lower right", frameon=True)

    for b in bars3:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", (b.get_x() + b.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, weight="bold", color="#d32f2f")
    for b in bars4:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", (b.get_x() + b.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, weight="bold", color="#7b1fa2")

    plt.suptitle("Ablation Study: Two-Phase Fine-Tuning vs. Full Unfreezing Across Data Regimes", fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(str(output_dir / "comparison_two_phase_vs_full_unfreeze.png"), bbox_inches="tight")
    plt.savefig(str(output_dir / "comparison_two_phase_vs_full_unfreeze.pdf"), bbox_inches="tight")
    plt.close()
    print(f"[OK] Salvo: {output_dir / 'comparison_two_phase_vs_full_unfreeze.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera gráficos acadêmicos para Full Unfreezing e Estudo de Ablação.")
    parser.add_argument("--json_path", type=str, default=None, help="Caminho para o JSON de resultados Full Unfreezing.")
    parser.add_argument("--output_dir", type=str, default="outputs/figures_full_unfreeze", help="Diretório de saída das figuras.")

    args = parser.parse_args()
    set_academic_style()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCIENTIFIC PLOT GENERATOR: FULL UNFREEZING & ABLATION STUDY")
    print("=" * 70)

    try:
        data, epochs = load_full_unfreeze_results(args.json_path)
    except FileNotFoundError as e:
        print(f"\n[AVISO] {e}")
        return

    plot_metrics_regimes(data, out)
    plot_confusion_matrix(data, out)
    if epochs:
        plot_learning_curves(epochs, out)

    two_phase_data = load_two_phase_results()
    if two_phase_data:
        print("[INFO] Resultados de Two-Phase detectados! Gerando gráfico comparativo de ablação...")
        plot_comparison_ablation(data, two_phase_data, out)

    print("=" * 70)
    print(f"[SUCESSO] Figuras geradas com sucesso em: {out.resolve()}")


if __name__ == "__main__":
    main()
