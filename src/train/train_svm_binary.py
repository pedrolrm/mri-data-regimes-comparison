"""
Script de Treinamento e Avaliação do SVM Binário sob Regimes de Dados.

Avalia o classificador Support Vector Machine (RBF) na tarefa de contraste extremo:
- Classe 0: NonDemented (Cérebro Saudável / Controle)
- Classe 1: ModerateDemented (Demência Moderada)

Compara a eficiência amostral (Sample Efficiency) através de regimes de dados:
N = [1000, 5000, 10000, 20000] amostras (ou conforme configurado via CLI).
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Garante que a raiz do projeto esteja no sys.path
_RAIZ_PROJETO = Path(__file__).resolve().parents[2]
if str(_RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROJETO))

from src.models.svm import ClassificadorAlzheimerSVM


def carregar_dados_extremos(
    caminho_features: Union[str, Path] = "data/features_extreme/features.npy",
    caminho_labels: Union[str, Path] = "data/features_extreme/labels.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega as características e rótulos. Trata tanto o formato já binário
    (gerado por extract_features.py) quanto o formato original de 4 classes.
    """
    p_feat = Path(caminho_features)
    p_lab = Path(caminho_labels)

    if not p_feat.exists():
        raise FileNotFoundError(f"Arquivo de características não encontrado: {p_feat}")
    if not p_lab.exists():
        raise FileNotFoundError(f"Arquivo de rótulos não encontrado: {p_lab}")

    print(f"[INFO] Carregando características de: {p_feat}")
    X_total = np.load(p_feat)
    y_total = np.load(p_lab)

    valores_unicos = np.unique(y_total)

    # Caso 1: O dataset já foi extraído como binário (apenas classes 0 e 1)
    if set(valores_unicos).issubset({0, 1}):
        X = X_total
        y = y_total.astype(np.int64)
    # Caso 2: O dataset contém as 4 classes originais (0: Mild, 1: Moderate, 2: Non, 3: VeryMild)
    else:
        print("[INFO] Dataset multiclasse detectado. Filtrando classes extremas (NonDemented e ModerateDemented)...")
        mascara_extrema = (y_total == 2) | (y_total == 1)
        X = X_total[mascara_extrema]
        y_bruto = y_total[mascara_extrema]
        # Mapeia: NonDemented (2) -> 0 | ModerateDemented (1) -> 1
        y = np.where(y_bruto == 1, 1, 0).astype(np.int64)

    print("=" * 70)
    print("EXTREME CONTRAST CLASSIFICATION: NonDemented (0) vs ModerateDemented (1)")
    print("=" * 70)
    print(f"Total loaded samples:        {len(X):,}")
    print(f"Feature dimension:           {X.shape[1]}")
    n_non = int(np.sum(y == 0))
    n_mod = int(np.sum(y == 1))
    pct_non = (n_non / len(y)) * 100 if len(y) > 0 else 0
    pct_mod = (n_mod / len(y)) * 100 if len(y) > 0 else 0
    print(f"  - NonDemented (Class 0):    {n_non:>6} ({pct_non:.1f}%)")
    print(f"  - ModerateDemented (Class 1): {n_mod:>6} ({pct_mod:.1f}%)")
    print("-" * 70)

    return X, y


def executar_experimento_regimes(
    X: np.ndarray,
    y: np.ndarray,
    regimes: List[int],
    dir_modelos: Union[str, Path] = "models/checkpoints_extreme",
    proporcao_teste: float = 0.2,
    C: float = 1.0,
    kernel: str = "rbf",
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Executes training and evaluation of SVM across sample data regimes.
    """
    dir_modelos = Path(dir_modelos)
    dir_modelos.mkdir(parents=True, exist_ok=True)

    resultados: List[Dict[str, Any]] = []

    for N in regimes:
        print(f"\n>>> REGIME N = {N:,} SAMPLES")
        print("-" * 60)

        # 1. Stratified subsampling maintaining 50% / 50%
        if N < len(X):
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=N, stratify=y, random_state=random_state
            )
        else:
            X_sub, y_sub = X.copy(), y.copy()

        # 2. Stratified 80% Train / 20% Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=proporcao_teste, stratify=y_sub, random_state=random_state
        )

        print(f"Train samples: {len(X_train):,} (Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)})")
        print(f"Test samples:  {len(X_test):,} (Class 0: {np.sum(y_test == 0)}, Class 1: {np.sum(y_test == 1)})")

        # 3. Model instantiation and training
        classificador = ClassificadorAlzheimerSVM(
            C=C,
            kernel=kernel,
            random_state=random_state
        )

        t0 = time.perf_counter()
        classificador.treinar(X_train, y_train)
        tempo_treino = time.perf_counter() - t0

        # 4. Evaluation on independent test set
        y_pred = classificador.prever(X_test)

        acc = accuracy_score(y_test, y_pred)
        sensibilidade = recall_score(y_test, y_pred, pos_label=1)
        matriz = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = matriz.ravel()
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_test, y_pred, pos_label=1)
        precisao = precision_score(y_test, y_pred, pos_label=1)

        print(f"\nTraining Time: {tempo_treino:.4f}s")
        print(f"Accuracy:      {acc * 100:.2f}%")
        print(f"Sensitivity:   {sensibilidade * 100:.2f}% (ModerateDemented Recall)")
        print(f"Specificity:   {especificidade * 100:.2f}% (NonDemented Recall)")
        print(f"Precision:     {precisao * 100:.2f}%")
        print(f"F1-Score:      {f1:.4f}")
        print("\nConfusion Matrix:")
        print(f"                          Pred: NonDemented    Pred: ModerateDemented")
        print(f"True: NonDemented                 {tn:>6}                    {fp:>6}")
        print(f"True: ModerateDemented            {fn:>6}                    {tp:>6}")

        # 5. Save model checkpoint
        caminho_checkpoint = dir_modelos / f"svm_extremo_{N}.joblib"
        classificador.salvar_modelo(caminho_checkpoint)
        print(f"[OK] Checkpoint saved: {caminho_checkpoint}")

        resultados.append({
            "regime": N,
            "treino": len(X_train),
            "teste": len(X_test),
            "tempo": round(tempo_treino, 4),
            "acc": round(acc * 100, 2),
            "sensibilidade": round(sensibilidade * 100, 2),
            "especificidade": round(especificidade * 100, 2),
            "precisao": round(precisao * 100, 2),
            "f1": round(f1, 4),
            "matriz_confusao": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            },
            "checkpoint": str(caminho_checkpoint)
        })

    # 6. Summary table in English
    print("\n" + "=" * 96)
    print("FINAL SUMMARY: SAMPLE REGIMES BENCHMARK (SVM RBF)")
    print("=" * 96)
    print(f"{'Regime (N)':<12} | {'Train N':<10} | {'Time (s)':<10} | {'Accuracy':<12} | {'Sensitivity':<15} | {'Specificity':<15} | {'F1-Score':<10}")
    print("-" * 96)
    for r in resultados:
        print(f"{r['regime']:<12} | {r['treino']:<10} | {r['tempo']:<10.4f} | {r['acc']:<10.2f}% | {r['sensibilidade']:<13.2f}% | {r['especificidade']:<13.2f}% | {r['f1']:<10.4f}")
    print("=" * 96 + "\n")

    # Save JSON report
    relatorio_json = dir_modelos / "relatorio_regimes_svm.json"
    with open(relatorio_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "task": "NonDemented vs ModerateDemented",
            "kernel": kernel,
            "C": C,
            "resultados": resultados
        }, f, indent=4, ensure_ascii=False)
    print(f"[SUCCESS] Structured report saved to: {relatorio_json}")

    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treinamento e Avaliação do SVM Binário (Extremos) sob Regimes de Dados."
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/features_extreme/features.npy",
        help="Caminho do arquivo de características .npy"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="data/features_extreme/labels.npy",
        help="Caminho do arquivo de rótulos .npy"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints_extreme",
        help="Diretório para salvar modelos e métricas"
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        type=int,
        default=[1000, 5000, 10000, 20000],
        help="Lista de regimes de dados N (padrão: 1000 5000 10000 20000)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporção do conjunto de teste (padrão: 0.2)"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Parâmetro de regularização C do SVM (padrão: 1.0)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="Função de kernel do SVM (padrão: rbf)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade (padrão: 42)"
    )

    args = parser.parse_args()

    X, y = carregar_dados_extremos(
        caminho_features=args.features_path,
        caminho_labels=args.labels_path
    )

    executar_experimento_regimes(
        X=X,
        y=y,
        regimes=args.regimes,
        dir_modelos=args.output_dir,
        proporcao_teste=args.test_size,
        C=args.C,
        kernel=args.kernel,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
