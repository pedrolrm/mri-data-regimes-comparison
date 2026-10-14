"""
Plotting Script for ResNet-50 across Data Regimes.

Generates publication-quality figures (300 DPI, PNG + PDF) in English:
1. Overall Performance: Accuracy, Sensitivity, Specificity, F1-Score vs Sample Regime N.
2. Training Time: Wall-clock latency (seconds and minutes) vs Sample Regime N.
3. Confusion Matrix: Real test evaluation matrix (N = 20,000, 4,000 test samples).
4. Learning Curves: Epoch-by-epoch loss and accuracy convergence.

All titles and descriptions are purely objective with zero unfreeze/freeze terminology.
Uses adaptive quadrant positioning with halo boxes to strictly prevent label overlap.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def set_academic_style() -> None:
    """Configures clean scientific visualization style in English."""
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
    sns.set_theme(style="whitegrid")


def load_results(
    json_path: Optional[Union[str, Path]] = None,
    default_dir: Union[str, Path] = "outputs/full_unfreeze"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads strictly real experimental results."""
    candidates_json = [
        Path(json_path) if json_path else None,
        Path(default_dir) / "relatorio_regimes_resnet50_full_unfreeze_moderate.json",
        Path("outputs/full_unfreeze/relatorio_regimes_resnet50_full_unfreeze_moderate.json"),
        Path("outputs/relatorio_regimes_resnet50_full_unfreeze_moderate.json"),
    ]

    for p in candidates_json:
        if p and p.exists():
            print(f"[INFO] Loading real results from: {p.resolve()}")
            with open(p, "r", encoding="utf-8") as f:
                content = json.load(f)
            resultados = content.get("resultados", content)
            if resultados and isinstance(resultados, list):
                historico = resultados[-1].get("historico_epocas", [])
                return resultados, historico

    tried = "\n".join([f"  - {str(p)}" for p in candidates_json if p])
    raise FileNotFoundError(
        f"ERROR: Results not found.\nChecked locations:\n{tried}"
    )


def plot_metrics_regimes(data: List[Dict[str, Any]], output_dirs: List[Path]) -> None:
    """
    Plots evaluation metrics vs Sample Regime N.
    Applies adaptive 4-quadrant layout with halo boxes to eliminate all label collisions.
    """
    regimes = [r["regime"] for r in data]
    sens = [r["sensibilidade"] * 100 if r["sensibilidade"] <= 1.0 else r["sensibilidade"] for r in data]
    f1 = [r["f1_macro"] * 100 if r["f1_macro"] <= 1.0 else r["f1_macro"] for r in data]
    acc = [r["accuracy"] * 100 if r["accuracy"] <= 1.0 else r["accuracy"] for r in data]
    espec = [r["especificidade"] * 100 if r["especificidade"] <= 1.0 else r["especificidade"] for r in data]

    x_pos = np.arange(len(regimes))

    fig, ax = plt.subplots(figsize=(11, 6.8))

    # Distinct line curves with standard markers
    line_sens, = ax.plot(x_pos, sens, marker="^", markersize=9, linewidth=2.5, color="#d62728", label="Sensitivity (Moderate Demented)")
    line_f1, = ax.plot(x_pos, f1, marker="s", markersize=8.5, linewidth=2.5, color="#2ca02c", label="F1-Score (%)")
    line_acc, = ax.plot(x_pos, acc, marker="o", markersize=8.5, linewidth=2.5, color="#1f77b4", label="Accuracy (%)")
    line_espec, = ax.plot(x_pos, espec, marker="d", markersize=8.5, linewidth=2.5, color="#9467bd", label="Specificity (Non-Demented)")

    bbox_halo = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9)

    # Adaptive non-overlapping quadrant placement
    for i in range(len(regimes)):
        # Specificity (highest) placed directly above with positive y offset
        ax.annotate(
            f"{espec[i]:.1f}%",
            (x_pos[i], espec[i]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            va="bottom",
            fontsize=9.5,
            weight="bold",
            color="#9467bd",
            bbox=bbox_halo
        )

        # Sensitivity (lowest) placed directly below with negative y offset
        ax.annotate(
            f"{sens[i]:.1f}%",
            (x_pos[i], sens[i]),
            textcoords="offset points",
            xytext=(0, -17),
            ha="center",
            va="top",
            fontsize=9.5,
            weight="bold",
            color="#d62728",
            bbox=bbox_halo
        )

        # Accuracy placed in Upper-Left quadrant to completely avoid F1 collision
        ax.annotate(
            f"{acc[i]:.1f}%",
            (x_pos[i], acc[i]),
            textcoords="offset points",
            xytext=(-14, 7),
            ha="right",
            va="bottom",
            fontsize=9.5,
            weight="bold",
            color="#1f77b4",
            bbox=bbox_halo
        )

        # F1-Score placed in Lower-Right quadrant to completely avoid Accuracy collision
        ax.annotate(
            f"{f1[i]:.1f}%",
            (x_pos[i], f1[i]),
            textcoords="offset points",
            xytext=(14, -13),
            ha="left",
            va="top",
            fontsize=9.5,
            weight="bold",
            color="#2ca02c",
            bbox=bbox_halo
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N = {r:,}" for r in regimes], fontweight="bold")
    ax.set_xlabel("Sample Regime (Total Sample Size - N)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Test Performance Metrics (%)", fontweight="bold", labelpad=10)
    ax.set_xlim(-0.35, len(regimes) - 0.65)
    ax.set_ylim(25, 103)

    # Objective title with zero freeze/unfreeze wording
    ax.set_title("ResNet-50 Performance Across Data Regimes", fontweight="bold", pad=14, fontsize=13)

    # Clean legend positioned below the x-axis to prevent any overlap
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
    for d in output_dirs:
        plt.savefig((d / "resnet_metrics_regimes.png").as_posix(), bbox_inches="tight")
        plt.savefig((d / "resnet_metrics_regimes.pdf").as_posix(), bbox_inches="tight")
        # Also save with alternative naming for backward compatibility
        plt.savefig((d / "full_unfreeze_metrics_regimes.png").as_posix(), bbox_inches="tight")
        plt.savefig((d / "full_unfreeze_metrics_regimes.pdf").as_posix(), bbox_inches="tight")
    plt.close()
    print("[OK] Saved Performance Across Data Regimes (clean anti-collision layout)")


def plot_training_time(data: List[Dict[str, Any]], output_dirs: List[Path]) -> None:
    """Plots training time (seconds and minutes) across sample regimes."""
    regimes = [r["regime"] for r in data]
    times_s = [r.get("tempo_s", 0.0) for r in data]
    x_pos = np.arange(len(regimes))

    fig, ax = plt.subplots(figsize=(9, 5.8))

    bar_color = "#1f77b4"
    bars = ax.bar(x_pos, times_s, width=0.48, color=bar_color, edgecolor="#0d47a1", linewidth=1.2, alpha=0.9, zorder=3)

    # Trend line connecting bars
    ax.plot(x_pos, times_s, color="#d32f2f", marker="o", markersize=7, linewidth=2.0, linestyle="--", zorder=4, label="Scaling Trend")

    # Annotate time values directly on top of each bar
    for b, t in zip(bars, times_s):
        h = b.get_height()
        minutes = t / 60.0
        label_text = f"{t:.1f} s\n({minutes:.1f} min)"
        ax.annotate(
            label_text,
            xy=(b.get_x() + b.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            va="bottom",
            fontsize=9.5,
            weight="bold",
            color="#0d47a1"
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N = {r:,}" for r in regimes], fontweight="bold")
    ax.set_xlabel("Sample Regime (Total Sample Size - N)", fontweight="bold", labelpad=10)
    ax.set_ylabel("Training Time (seconds)", fontweight="bold", labelpad=10)
    ax.set_ylim(0, max(times_s) * 1.25)
    ax.set_title("ResNet-50 Training Time Across Data Regimes", fontweight="bold", pad=14, fontsize=13)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    for d in output_dirs:
        plt.savefig((d / "resnet_training_time.png").as_posix(), bbox_inches="tight")
        plt.savefig((d / "resnet_training_time.pdf").as_posix(), bbox_inches="tight")
        plt.savefig((d / "full_unfreeze_training_time.png").as_posix(), bbox_inches="tight")
        plt.savefig((d / "full_unfreeze_training_time.pdf").as_posix(), bbox_inches="tight")
    plt.close()
    print("[OK] Saved Training Time Across Data Regimes")


def plot_confusion_matrix(data: List[Dict[str, Any]], output_dirs: List[Path]) -> None:
    """Plots the real test confusion matrix for the final regime (N = 20,000)."""
    last = data[-1]
    n_regime = last["regime"]
    m = last.get("matriz_confusao", {})
    tn, fp, fn, tp = m.get("tn", 0), m.get("fp", 0), m.get("fn", 0), m.get("tp", 0)

    matrix = np.array([[tn, fp], [fn, tp]])
    total_test = tn + fp + fn + tp
    real_acc = (tn + tp) / total_test * 100.0 if total_test > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    class_labels = ["Non-Demented\n(Control)", "Moderate Demented\n(Pathological)"]

    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
        xticklabels=class_labels, yticklabels=class_labels,
        annot_kws={"size": 14, "weight": "bold"}
    )

    # Objective title with zero unfreeze/freeze wording
    ax.set_title(
        f"ResNet-50 Confusion Matrix (N = {n_regime:,})\n"
        f"Accuracy: {real_acc:.2f}% | Test Samples: {total_test:,}",
        fontweight="bold", pad=12, fontsize=12
    )
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=8)

    plt.tight_layout()
    for d in output_dirs:
        plt.savefig((d / "resnet_confusion_matrix.png").as_posix(), bbox_inches="tight")
        plt.savefig((d / "resnet_confusion_matrix.pdf").as_posix(), bbox_inches="tight")
        plt.savefig((d / "full_unfreeze_confusion_matrix.png").as_posix(), bbox_inches="tight")
        plt.savefig((d / "full_unfreeze_confusion_matrix.pdf").as_posix(), bbox_inches="tight")
    plt.close()
    print("[OK] Saved Confusion Matrix")


def plot_learning_curves(epochs: List[Dict[str, Any]], output_dirs: List[Path]) -> None:
    """Plots convergence curves across epochs without unfreeze/freeze wording."""
    if not epochs:
        return

    n_epochs = [e["epoca"] for e in epochs]
    tr_loss = [e["train_loss"] for e in epochs]
    val_loss = [e["val_loss"] for e in epochs]
    tr_acc = [e["train_acc"] for e in epochs]
    val_acc = [e["val_acc"] for e in epochs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Loss
    ax1.plot(n_epochs, tr_loss, marker="o", linewidth=2.2, color="#1f77b4", label="Training Loss")
    ax1.plot(n_epochs, val_loss, marker="s", linewidth=2.2, color="#d32f2f", label="Validation Loss")
    ax1.set_title("Cross-Entropy Loss Curve", fontweight="bold")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    # Accuracy
    ax2.plot(n_epochs, tr_acc, marker="o", linewidth=2.2, color="#1f77b4", label="Training Accuracy")
    ax2.plot(n_epochs, val_acc, marker="s", linewidth=2.2, color="#d32f2f", label="Validation Accuracy")
    ax2.set_title("Accuracy Curve (%)", fontweight="bold")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.suptitle("ResNet-50 Training Convergence", fontweight="bold", y=0.98, fontsize=14)
    plt.tight_layout()
    for d in output_dirs:
        plt.savefig((d / "resnet_learning_curves.png").as_posix(), bbox_inches="tight")
        plt.savefig((d / "resnet_learning_curves.pdf").as_posix(), bbox_inches="tight")
        plt.savefig((d / "full_unfreeze_learning_curves.png").as_posix(), bbox_inches="tight")
        plt.savefig((d / "full_unfreeze_learning_curves.pdf").as_posix(), bbox_inches="tight")
    plt.close()
    print("[OK] Saved Learning Curves")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Scientific Figures for ResNet-50 across Data Regimes.")
    parser.add_argument("--json_path", type=str, default=None, help="Path to full unfreeze results JSON.")
    parser.add_argument("--output_dir", type=str, default="outputs/figures_full_unfreeze", help="Primary output directory.")

    args = parser.parse_args()
    set_academic_style()

    out_primary = Path(args.output_dir).resolve()
    out_resnet = Path("outputs/figures_resnet").resolve()
    out_primary.mkdir(parents=True, exist_ok=True)
    out_resnet.mkdir(parents=True, exist_ok=True)

    target_dirs = [out_primary, out_resnet]

    print("=" * 70)
    print("SCIENTIFIC PLOT GENERATOR: RESNET-50 DATA REGIMES")
    print("=" * 70)

    try:
        data, epochs = load_results(args.json_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    plot_metrics_regimes(data, target_dirs)
    plot_training_time(data, target_dirs)
    plot_confusion_matrix(data, target_dirs)
    if epochs:
        plot_learning_curves(epochs, target_dirs)

    print("=" * 70)
    print(f"[SUCCESS] All figures successfully updated and saved in:\n  - {out_primary}\n  - {out_resnet}")


if __name__ == "__main__":
    main()
