"""
Script de Avaliação Clínica e Inferência do Pipeline ResNet-50 Encoder -> SVM.

Permite avaliar um modelo SVM previamente treinado sob duas modalidades:
1. Avaliação via características pré-extraídas (.npy): avalia diretamente sobre matrizes
   de características de 2048 dimensões já calculadas.
2. Avaliação ponta-a-ponta via imagens brutas: carrega as imagens MRI, extrai os vetores
   latentes com a ResNet50Encoder pré-treinada no RadImageNet e infere com o SVM.

Calcula métricas clínicas completas:
- Acurácia Global
- Sensibilidade (Recall da classe ModerateDemented / detecção de patologia)
- Especificidade (Recall da classe NonDemented / confirmação de cérebro sadio)
- Precisão (Valor Preditivo Positivo)
- Valor Preditivo Negativo (VPN)
- F1-Score (Binário e Macro)
- Área sob a Curva ROC (ROC-AUC)
- Matriz de Confusão detalhada
- Latência média por amostra e Throughput
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torchvision import transforms

# Garante raiz do repositório no sys.path
_RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(_RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROJETO))

from src.models.encoder import ResNet50Encoder
from src.models.svm import ClassificadorAlzheimerSVM


def carregar_modelo_svm(caminho_modelo: Union[str, Path]) -> Tuple[Any, Any]:
    """
    Carrega o arquivo .joblib contendo o pipeline SVM.
    Suporta tanto arquivos salvos como dicionário {'scaler': ..., 'modelo': ...}
    quanto instâncias do ClassificadorAlzheimerSVM.
    """
    p = Path(caminho_modelo)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {p}")

    conteudo = joblib.load(p)

    if isinstance(conteudo, ClassificadorAlzheimerSVM):
        scaler = conteudo.scaler
        modelo = conteudo.modelo
    elif isinstance(conteudo, dict) and "scaler" in conteudo and "modelo" in conteudo:
        scaler = conteudo["scaler"]
        modelo = conteudo["modelo"]
    else:
        raise ValueError(f"Formato inesperado no arquivo de checkpoint: {p}")

    return scaler, modelo


def extrair_caracteristicas_de_imagens(
    diretorio_dados: Union[str, Path],
    classes: Tuple[str, str] = ("NonDemented", "ModerateDemented"),
    batch_size: int = 64,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carrega imagens brutas do diretório e extrai vetores de 2048 dimensões usando ResNet50Encoder.
    """
    dir_dados = Path(diretorio_dados)
    dispositivo = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    encoder = ResNet50Encoder(carregar_radimagenet=True, congelar_parametros=True, device=dispositivo)
    encoder.to(dispositivo)
    encoder.eval()

    amostras: List[Tuple[Path, int]] = []
    class_to_idx = {classes[0]: 0, classes[1]: 1}
    extensoes = {".jpg", ".jpeg", ".png"}

    for cls_nome in classes:
        pasta = dir_dados / cls_nome
        if not pasta.exists():
            raise FileNotFoundError(f"Pasta da classe não encontrada: {pasta}")
        rotulo = class_to_idx[cls_nome]
        for arq in sorted(pasta.iterdir()):
            if arq.suffix.lower() in extensoes:
                amostras.append((arq, rotulo))

    print(f"[INFO] Extraindo características de {len(amostras):,} imagens brutas...")
    lista_features = []
    lista_labels = []
    lista_nomes = []

    with torch.no_grad():
        for i in range(0, len(amostras), batch_size):
            lote_amostras = amostras[i:i + batch_size]
            tensores = []
            rotulos = []
            nomes = []

            for caminho_img, rotulo in lote_amostras:
                with Image.open(caminho_img) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    tensores.append(transform(img))
                rotulos.append(rotulo)
                nomes.append(caminho_img.name)

            lote_tensores = torch.stack(tensores).to(dispositivo)
            vetores = encoder(lote_tensores)
            lista_features.append(vetores.cpu().numpy().astype(np.float32))
            lista_labels.extend(rotulos)
            lista_nomes.extend(nomes)

    X = np.concatenate(lista_features, axis=0)
    y = np.array(lista_labels, dtype=np.int64)
    return X, y, lista_nomes


def avaliar_modelo(
    caminho_modelo: Union[str, Path],
    caminho_features: Optional[Union[str, Path]] = None,
    caminho_labels: Optional[Union[str, Path]] = None,
    diretorio_imagens: Optional[Union[str, Path]] = None,
    caminho_saida: Optional[Union[str, Path]] = None,
    classes: Tuple[str, str] = ("NonDemented", "ModerateDemented")
) -> Dict[str, Any]:
    """
    Executa a avaliação quantitativa e clínica do modelo SVM treinado.
    """
    scaler, modelo = carregar_modelo_svm(caminho_modelo)
    print("=" * 75)
    print(f"AVALIAÇÃO DO MODELO SVM: {Path(caminho_modelo).name}")
    print("=" * 75)

    # 1. Obtenção dos dados de teste
    if caminho_features and caminho_labels:
        print(f"[INFO] Carregando características pré-extraídas de {caminho_features}...")
        X_bruto = np.load(caminho_features)
        y_bruto = np.load(caminho_labels)

        # Trata filtro caso o arquivo venha com as 4 classes originais
        if not set(np.unique(y_bruto)).issubset({0, 1}):
            mascara = (y_bruto == 2) | (y_bruto == 1)
            X = X_bruto[mascara]
            y = np.where(y_bruto[mascara] == 1, 1, 0).astype(np.int64)
        else:
            X = X_bruto
            y = y_bruto.astype(np.int64)
    elif diretorio_imagens:
        X, y, _ = extrair_caracteristicas_de_imagens(diretorio_imagens, classes=classes)
    else:
        raise ValueError("É necessário informar (caminho_features e caminho_labels) OU diretorio_imagens.")

    print(f"Total de amostras avaliadas: {len(X):,}")
    print(f"  - {classes[0]} (Classe 0):     {np.sum(y == 0):>6} ({(np.sum(y == 0)/len(y))*100:.1f}%)")
    print(f"  - {classes[1]} (Classe 1): {np.sum(y == 1):>6} ({(np.sum(y == 1)/len(y))*100:.1f}%)")
    print("-" * 75)

    # 2. Inferência cronometrada
    t0 = time.perf_counter()
    X_escalado = scaler.transform(X)
    y_pred = modelo.predict(X_escalado)
    tempo_inferencia = time.perf_counter() - t0

    tempo_por_amostra_ms = (tempo_inferencia / len(X)) * 1000 if len(X) > 0 else 0.0
    throughput = len(X) / tempo_inferencia if tempo_inferencia > 0 else 0.0

    # 3. Cálculo dos scores de decisão para ROC-AUC
    tem_roc = False
    auc_score = 0.0
    if hasattr(modelo, "decision_function"):
        try:
            scores_decisao = modelo.decision_function(X_escalado)
            auc_score = roc_auc_score(y, scores_decisao)
            tem_roc = True
        except Exception:
            tem_roc = False
    elif hasattr(modelo, "predict_proba"):
        try:
            probabilidades = modelo.predict_proba(X_escalado)[:, 1]
            auc_score = roc_auc_score(y, probabilidades)
            tem_roc = True
        except Exception:
            tem_roc = False

    # 4. Cálculo das métricas clínicas
    acc = accuracy_score(y, y_pred)
    sensibilidade = recall_score(y, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precisao = precision_score(y, y_pred, pos_label=1)
    vpn = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_binario = f1_score(y, y_pred, pos_label=1)
    f1_macro = f1_score(y, y_pred, average="macro")

    # 5. Apresentação dos resultados
    print(f"\nRELATÓRIO CLÍNICO DE DESEMPENHO:")
    print("-" * 75)
    print(f"Acurácia Global:              {acc * 100:.2f}%")
    print(f"Sensibilidade (Recall Moderado): {sensibilidade * 100:.2f}%  [Capacidade de detectar demência]")
    print(f"Especificidade (Recall Sadio):   {especificidade * 100:.2f}%  [Capacidade de confirmar cérebro saudável]")
    print(f"Precisão (VPP):                 {precisao * 100:.2f}%")
    print(f"Valor Preditivo Negativo (VPN): {vpn * 100:.2f}%")
    print(f"F1-Score (Binário):             {f1_binario:.4f}")
    print(f"F1-Score (Macro):               {f1_macro:.4f}")
    if tem_roc:
        print(f"ROC AUC:                        {auc_score:.4f}")
    print(f"\nTempo Total de Inferência:      {tempo_inferencia:.4f} s")
    print(f"Latência Média por Imagem:      {tempo_por_amostra_ms:.3f} ms")
    print(f"Throughput de Classificação:    {throughput:.1f} amostras/s")

    print("\nMATRIZ DE CONFUSÃO:")
    print(f"                              Pred: {classes[0]}    Pred: {classes[1]}")
    print(f"Real: {classes[0]:<18}       {tn:>6} ({tn/(tn+fp)*100:>5.1f}%)        {fp:>6} ({fp/(tn+fp)*100:>5.1f}%)")
    print(f"Real: {classes[1]:<18}       {fn:>6} ({fn/(fn+tp)*100:>5.1f}%)        {tp:>6} ({tp/(fn+tp)*100:>5.1f}%)")

    relatorio_texto = classification_report(
        y, y_pred, target_names=[classes[0], classes[1]], digits=4
    )
    print("\nRELATÓRIO DETALHADO POR CLASSE (SCIKIT-LEARN):")
    print(relatorio_texto)
    print("=" * 75)

    resultados = {
        "timestamp": datetime.now().isoformat(),
        "modelo": str(Path(caminho_modelo).name),
        "total_amostras": int(len(X)),
        "metricas": {
            "acuracia": round(acc * 100, 2),
            "sensibilidade": round(sensibilidade * 100, 2),
            "especificidade": round(especificidade * 100, 2),
            "precisao": round(precisao * 100, 2),
            "vpn": round(vpn * 100, 2),
            "f1_binario": round(f1_binario, 4),
            "f1_macro": round(f1_macro, 4),
            "roc_auc": round(auc_score, 4) if tem_roc else None,
        },
        "matriz_confusao": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "performance_computacional": {
            "tempo_total_segundos": round(tempo_inferencia, 4),
            "latencia_media_ms": round(tempo_por_amostra_ms, 3),
            "throughput_amostras_por_seg": round(throughput, 1)
        }
    }

    if caminho_saida:
        p_saida = Path(caminho_saida)
        p_saida.parent.mkdir(parents=True, exist_ok=True)
        with open(p_saida, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)
        print(f"[OK] Relatório gravado com sucesso em: {p_saida}")

    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avaliação e Inferência Clínica do Pipeline ResNet-50 Encoder -> SVM."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Caminho do checkpoint .joblib treinado (ex.: models/checkpoints_extreme/svm_extremo_20000.joblib)"
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Caminho do arquivo .npy de características para avaliação direta"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="Caminho do arquivo .npy de rótulos"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Caminho do diretório de imagens para avaliação ponta a ponta"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/eval_results.json",
        help="Caminho do arquivo JSON onde as métricas serão gravadas"
    )

    args = parser.parse_args()

    avaliar_modelo(
        caminho_modelo=args.model_path,
        caminho_features=args.features_path,
        caminho_labels=args.labels_path,
        diretorio_imagens=args.data_dir,
        caminho_saida=args.output_json
    )


if __name__ == "__main__":
    main()
