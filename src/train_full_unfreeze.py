"""
Script de Treinamento da ResNet-50 via Fine-Tuning com Descongelamento Total (Full Unfreezing).

Nesta estratégia experimental de Transfer Learning:
1. Os pesos do backbone são inicializados com o modelo pré-treinado no RadImageNet (Lab-Rasool/RadImageNet).
2. A nova cabeça classificadora é inicializada para diagnóstico binário (2 classes).
3. Todas as camadas da rede (backbone convolucional + cabeça linear) são DESCONGELADAS desde a Época 1 (requires_grad = True).
4. Não há fase de aquecimento (warmup / linear probing) com camadas congeladas.

Hipótese Científica (Ablação para o Artigo):
Como a camada de saída começa com pesos aleatórios, os primeiros passos de gradiente com
loss elevada podem causar perturbação nos filtros convolucionais pré-treinados do RadImageNet
("gradient shock" / "catastrophic forgetting"), especialmente nos regimes de dados menores (N=1.000).
Isso permite comparar diretamente:
- Two-Phase Differential Fine-Tuning (Fase 1 congelada + Fase 2 diferencial) vs.
- Full Unfreezing Fine-Tuning (Descongelamento total desde o início).

Totalmente otimizado para a NVIDIA GeForce RTX 5060 (AMP bfloat16, TF32).
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

# Raiz do projeto no sys.path
_RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(_RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROJETO))

from src.models.resnet50 import ResNet50Classifier
from utils.preprocess_mri import redimensionar_com_padding_simetrico


class DualLogger:
    """Duplica a saída entre terminal e arquivo de log (.txt)."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def log(self, message: str = "") -> None:
        self.stdout.write(message + "\n")
        self.stdout.flush()
        self.file.write(message + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file and not self.file.closed:
            self.file.close()


class TransformacaoComPaddingDinamico:
    """Padroniza dimensões em 224x224 com padding simétrico."""

    def __init__(self, tamanho: int = 224) -> None:
        self.tamanho = tamanho

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.tamanho, self.tamanho):
            return redimensionar_com_padding_simetrico(img, tamanho_alvo=self.tamanho)
        return img


def fixar_sementes(seed: int = 42) -> None:
    """Garante reprodutibilidade científica."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configurar_ambiente_rtx5060(logger: DualLogger) -> Tuple[torch.device, torch.dtype]:
    """Configura aceleração máxima por hardware na RTX 5060 (Blackwell/Ada)."""
    if not torch.cuda.is_available():
        logger.log("[AVISO] CUDA não encontrada. Executando na CPU.")
        return torch.device("cpu"), torch.float32

    dispositivo = torch.device("cuda:0")
    nome_gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    logger.log("=" * 80)
    logger.log("CONFIGURAÇÃO DE HARDWARE (FULL UNFREEZING - RTX 5060)")
    logger.log("=" * 80)
    logger.log(f"Dispositivo:           {nome_gpu}")
    logger.log(f"VRAM Total:            {vram_gb:.2f} GB")
    logger.log(f"Versão PyTorch:        {torch.__version__}")
    logger.log(f"Versão CUDA:           {torch.version.cuda}")

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        logger.log("Matmul Precision:      high (TF32 habilitado para Tensor Cores)")

    if torch.cuda.is_bf16_supported():
        dtype_amp = torch.bfloat16
        logger.log("Precisão Mista (AMP):  torch.bfloat16 (Nativo RTX 5060)")
    else:
        dtype_amp = torch.float16
        logger.log("Precisão Mista (AMP):  torch.float16 (GradScaler)")

    logger.log("=" * 80)
    return dispositivo, dtype_amp


class BinaryDatasetWrapper(Dataset):
    """Encapsula imagens com seus novos rótulos binários (0 ou 1) e transformações."""

    def __init__(self, base_dataset: ImageFolder, indices: np.ndarray, binary_labels: np.ndarray, transform=None) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.binary_labels = binary_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        caminho_imagem, _ = self.base_dataset.samples[real_idx]
        img = Image.open(caminho_imagem)

        if self.transform:
            img = self.transform(img)

        rotulo_binario = int(self.binary_labels[idx])
        return img, rotulo_binario


def preparar_datasets_binarios(
    caminho_dados: Path,
    tarefa: str = "moderate",
    n_regime: Optional[int] = None,
    balancear_agrupamento: bool = True,
    proporcao_teste: float = 0.2,
    seed: int = 42,
    logger: Optional[DualLogger] = None,
) -> Tuple[Dataset, Dataset, List[str], Optional[torch.Tensor]]:
    """Cria os conjuntos de treino e teste estratificados conforme regime N."""
    media_norm = [0.485, 0.456, 0.406]
    desvio_norm = [0.229, 0.224, 0.225]

    transform_treino = transforms.Compose([
        TransformacaoComPaddingDinamico(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=media_norm, std=desvio_norm),
    ])

    transform_teste = transforms.Compose([
        TransformacaoComPaddingDinamico(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=media_norm, std=desvio_norm),
    ])

    dataset_bruto = ImageFolder(root=str(caminho_dados))
    classes_originais = dataset_bruto.classes
    alvos_originais = np.array(dataset_bruto.targets)

    idx_mild = classes_originais.index("MildDemented")
    idx_moderate = classes_originais.index("ModerateDemented")
    idx_non = classes_originais.index("NonDemented")
    idx_verymild = classes_originais.index("VeryMildDemented")

    indices_non = np.where(alvos_originais == idx_non)[0]
    indices_mild = np.where(alvos_originais == idx_mild)[0]
    indices_mod = np.where(alvos_originais == idx_moderate)[0]
    indices_verymild = np.where(alvos_originais == idx_verymild)[0]

    pesos_classe = None
    rng = np.random.default_rng(seed)

    if tarefa == "moderate":
        nomes_classes = ["NonDemented", "ModerateDemented"]
        if n_regime is not None and n_regime < (len(indices_non) + len(indices_mod)):
            n_per_class = n_regime // 2
            sub_non = rng.choice(indices_non, size=n_per_class, replace=False)
            sub_mod = rng.choice(indices_mod, size=n_per_class, replace=False)
            indices_filtrados = np.concatenate([sub_non, sub_mod])
            labels_binarios = np.concatenate([np.zeros(len(sub_non)), np.ones(len(sub_mod))])
        else:
            indices_filtrados = np.concatenate([indices_non, indices_mod])
            labels_binarios = np.concatenate([np.zeros(len(indices_non)), np.ones(len(indices_mod))])

        if logger:
            logger.log(f"Tarefa Selecionada:      Extremos (NonDemented vs ModerateDemented)")
            if n_regime:
                logger.log(f"Regime de Amostras:      N = {len(indices_filtrados)}")
            logger.log(f"NonDemented:             {(labels_binarios == 0).sum()} amostras")
            logger.log(f"ModerateDemented:        {(labels_binarios == 1).sum()} amostras")
            logger.log(f"Total Selecionado:       {len(indices_filtrados)}")

    elif tarefa == "grouped":
        nomes_classes = ["NonDemented", "Demented"]
        if balancear_agrupamento:
            total_disponivel = 20000
            if n_regime is not None and n_regime < total_disponivel:
                n_non = n_regime // 2
                n_dem = n_regime - n_non
                n_mild = n_dem // 3 + (1 if n_dem % 3 > 0 else 0)
                n_mod = n_dem // 3 + (1 if n_dem % 3 > 1 else 0)
                n_vmild = n_dem - (n_mild + n_mod)

                sub_non = rng.choice(indices_non, size=n_non, replace=False)
                sub_mild = rng.choice(indices_mild, size=n_mild, replace=False)
                sub_mod = rng.choice(indices_mod, size=n_mod, replace=False)
                sub_vmild = rng.choice(indices_verymild, size=n_vmild, replace=False)

                indices_filtrados = np.concatenate([sub_non, sub_mild, sub_mod, sub_vmild])
                labels_binarios = np.concatenate([np.zeros(len(sub_non)), np.ones(len(sub_mild) + len(sub_mod) + len(sub_vmild))])
            else:
                n_mild = 3334
                n_mod = 3333
                n_vmild = 3333
                sub_non = rng.choice(indices_non, size=10000, replace=False)
                sub_mild = rng.choice(indices_mild, size=n_mild, replace=False)
                sub_mod = rng.choice(indices_mod, size=n_mod, replace=False)
                sub_vmild = rng.choice(indices_verymild, size=n_vmild, replace=False)

                indices_filtrados = np.concatenate([sub_non, sub_mild, sub_mod, sub_vmild])
                labels_binarios = np.concatenate([np.zeros(len(sub_non)), np.ones(len(sub_mild) + len(sub_mod) + len(sub_vmild))])
        else:
            indices_demented = np.concatenate([indices_mild, indices_mod, indices_verymild])
            indices_filtrados = np.concatenate([indices_non, indices_demented])
            labels_binarios = np.concatenate([np.zeros(len(indices_non)), np.ones(len(indices_demented))])
            pesos_classe = torch.tensor([1.0, 10000.0 / 30000.0], dtype=torch.float32)

        if logger:
            logger.log(f"Tarefa Selecionada:      Agrupado (NonDemented vs Demented)")
            if n_regime:
                logger.log(f"Regime de Amostras:      N = {len(indices_filtrados)}")
    else:
        raise ValueError(f"Tarefa desconhecida: {tarefa}. Escolha 'moderate' ou 'grouped'.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=proporcao_teste, random_state=seed)
    idx_treino_rel, idx_teste_rel = next(splitter.split(indices_filtrados, labels_binarios))

    idx_treino_real = indices_filtrados[idx_treino_rel]
    labels_treino = labels_binarios[idx_treino_rel]

    idx_teste_real = indices_filtrados[idx_teste_rel]
    labels_teste = labels_binarios[idx_teste_rel]

    if logger:
        logger.log(f"Divisão Treino/Teste:    Treino = {len(idx_treino_real)} | Teste = {len(idx_teste_real)}")

    dataset_treino = BinaryDatasetWrapper(dataset_bruto, idx_treino_real, labels_treino, transform=transform_treino)
    dataset_teste = BinaryDatasetWrapper(dataset_bruto, idx_teste_real, labels_teste, transform=transform_teste)

    return dataset_treino, dataset_teste, nomes_classes, pesos_classe


def treinar_uma_epoca(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    dispositivo: torch.device,
    dtype_amp: torch.dtype,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Executa uma época de treino sob Full Unfreezing."""
    modelo.train()
    perda_total = 0.0
    acertos = 0
    total = 0
    usar_scaler = (scaler is not None and dtype_amp == torch.float16)

    for imagens, rotulos in dataloader:
        imagens = imagens.to(dispositivo, non_blocking=True)
        rotulos = rotulos.to(dispositivo, non_blocking=True)

        otimizador.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if dispositivo.type == "cuda" else "cpu", dtype=dtype_amp):
            logits = modelo(imagens)
            perda = criterio(logits, rotulos)

        if usar_scaler:
            scaler.scale(perda).backward()
            scaler.step(otimizador)
            scaler.update()
        else:
            perda.backward()
            otimizador.step()

        perda_total += perda.item() * imagens.size(0)
        _, preds = torch.max(logits, 1)
        acertos += (preds == rotulos).sum().item()
        total += rotulos.size(0)

    return (perda_total / total), (acertos / total)


def avaliar(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
    dtype_amp: torch.dtype,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Avalia o classificador em teste/validação."""
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    todas_predicoes = []
    todos_rotulos = []
    todas_probs_positivas = []

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(dispositivo, non_blocking=True)
            rotulos = rotulos.to(dispositivo, non_blocking=True)

            with torch.amp.autocast(device_type="cuda" if dispositivo.type == "cuda" else "cpu", dtype=dtype_amp):
                logits = modelo(imagens)
                perda = criterio(logits, rotulos)

            perda_total += perda.item() * imagens.size(0)
            probs = torch.softmax(logits.float(), dim=1)
            _, preds = torch.max(logits, 1)

            acertos += (preds == rotulos).sum().item()
            total += rotulos.size(0)

            todas_predicoes.extend(preds.cpu().numpy())
            todos_rotulos.extend(rotulos.cpu().numpy())
            todas_probs_positivas.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(todos_rotulos)
    y_pred = np.array(todas_predicoes)
    y_score = np.array(todas_probs_positivas)

    loss_media = perda_total / total
    acc = acertos / total
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = 0.5

    return loss_media, acc, f1_macro, auc, y_true, y_pred, y_score


def executar_experimento_full_unfreeze(
    caminho_dados: Path,
    tarefa: str = "moderate",
    n_regime: Optional[int] = None,
    balancear_agrupamento: bool = True,
    epocas: int = 10,
    batch_size: int = 64,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    paciencia: int = 4,
    seed: int = 42,
    pasta_saida: Path = Path("outputs/full_unfreeze"),
    logger: Optional[DualLogger] = None,
) -> Dict:
    """
    Executa o treinamento do modelo usando Full Unfreezing:
    - Backbone carregado com pesos RadImageNet.
    - TODAS as camadas descongeladas desde a época 1 (sem fase de aquecimento congelada).
    """
    fixar_sementes(seed)
    dispositivo, dtype_amp = configurar_ambiente_rtx5060(logger)

    tag_tarefa = "binary_moderate" if tarefa == "moderate" else "binary_grouped"
    tag_regime = f"_regime_{n_regime}" if n_regime is not None else "_full"
    pasta_checkpoints = pasta_saida / "checkpoints"
    pasta_checkpoints.mkdir(parents=True, exist_ok=True)
    caminho_melhor_modelo = pasta_checkpoints / f"best_resnet50_full_unfreeze_{tag_tarefa}{tag_regime}.pt"

    dataset_treino, dataset_teste, nomes_classes, pesos_classe = preparar_datasets_binarios(
        caminho_dados=caminho_dados,
        tarefa=tarefa,
        n_regime=n_regime,
        balancear_agrupamento=balancear_agrupamento,
        proporcao_teste=0.2,
        seed=seed,
        logger=logger,
    )

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    loader_teste = DataLoader(
        dataset_teste,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logger.log("\nInstanciando ResNet-50 com RadImageNet (FULL UNFREEZING)...")
    modelo = ResNet50Classifier(
        num_classes=2,
        carregar_radimagenet=True,
        dropout_rate=0.3,
        device=dispositivo,
    ).to(dispositivo)

    # =========================================================================
    # ESTRATÉGIA FULL UNFREEZING:
    # Descongela explicitamente TODOS os parâmetros da rede a partir do passo 0.
    # Não há época de aquecimento com backbone congelado.
    # =========================================================================
    modelo.unfreeze_backbone()
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logger.log(f"[FULL UNFREEZING] Parâmetros Treináveis na Época 1: {trainable_params:,} / {total_params:,} (100% Descongelado)")

    if pesos_classe is not None and dispositivo.type == "cuda":
        criterio = nn.CrossEntropyLoss(weight=pesos_classe.to(dispositivo))
    else:
        criterio = nn.CrossEntropyLoss()

    grupos_params = modelo.get_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
    otimizador = AdamW(grupos_params)
    scheduler = CosineAnnealingLR(otimizador, T_max=epocas, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if (dispositivo.type == "cuda" and dtype_amp == torch.float16) else None

    logger.log("=" * 80)
    logger.log(f"TREINAMENTO: FULL UNFREEZING ({tag_tarefa.upper()} | {epocas} ÉPOCAS | BATCH: {batch_size})")
    logger.log(f"Backbone LR: {backbone_lr} | Head LR: {head_lr} | Sem Warmup Congelado")
    logger.log("=" * 80)
    logger.log(f"{'Época':<7} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val F1-Mac':<10} | {'AUC':<7} | {'Tempo':<8} | {'VRAM'}")
    logger.log("-" * 80)

    melhor_val_f1 = -1.0
    epocas_sem_melhora = 0
    tempo_inicio_total = time.time()
    historico_epocas = []

    for epoca in range(1, epocas + 1):
        tempo_inicio_epoca = time.time()

        loss_treino, acc_treino = treinar_uma_epoca(
            modelo=modelo,
            dataloader=loader_treino,
            criterio=criterio,
            otimizador=otimizador,
            dispositivo=dispositivo,
            dtype_amp=dtype_amp,
            scaler=scaler,
        )

        loss_val, acc_val, f1_macro_val, auc_val, _, _, _ = avaliar(
            modelo=modelo,
            dataloader=loader_teste,
            criterio=criterio,
            dispositivo=dispositivo,
            dtype_amp=dtype_amp,
        )

        scheduler.step()
        duracao_epoca = time.time() - tempo_inicio_epoca

        vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
        marca_melhor = " *" if f1_macro_val > melhor_val_f1 else ""

        logger.log(
            f"{epoca:<7} | {loss_treino:<10.4f} | {acc_treino*100:<9.2f}% | "
            f"{loss_val:<10.4f} | {acc_val*100:<9.2f}% | {f1_macro_val:<10.4f}{marca_melhor:<2} | "
            f"{auc_val:<7.4f} | {duracao_epoca:<7.2f}s | {vram_mb:.0f}MB"
        )

        historico_epocas.append({
            "epoca": epoca,
            "train_loss": loss_treino,
            "train_acc": acc_treino * 100,
            "val_loss": loss_val,
            "val_acc": acc_val * 100,
            "val_f1": f1_macro_val,
            "auc": auc_val,
            "tempo_s": duracao_epoca,
        })

        if f1_macro_val > melhor_val_f1:
            melhor_val_f1 = f1_macro_val
            epocas_sem_melhora = 0
            caminho_melhor_modelo.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoca,
                "strategy": "full_unfreezing",
                "model_state_dict": modelo.state_dict(),
                "optimizer_state_dict": otimizador.state_dict(),
                "val_acc": acc_val,
                "val_f1_macro": f1_macro_val,
                "val_auc": auc_val,
                "classes": nomes_classes,
                "tarefa": tarefa,
            }, caminho_melhor_modelo)
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                logger.log(f"\n[EARLY STOPPING] Parada antecipada após {paciencia} épocas sem melhora.")
                break

    tempo_total = time.time() - tempo_inicio_total
    logger.log("=" * 80)
    logger.log(f"Treino Full Unfreezing concluído em {tempo_total:.2f}s ({tempo_total/60:.2f} min).")
    logger.log(f"Melhor modelo salvo em: {caminho_melhor_modelo.resolve()}")
    logger.log("=" * 80)

    # Avaliação Final com o Melhor Checkpoint
    checkpoint = torch.load(caminho_melhor_modelo, map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(checkpoint["model_state_dict"])

    loss_final, acc_final, f1_macro_final, auc_final, y_reais, y_preditos, _ = avaliar(
        modelo=modelo,
        dataloader=loader_teste,
        criterio=criterio,
        dispositivo=dispositivo,
        dtype_amp=dtype_amp,
    )

    relatorio = classification_report(y_reais, y_preditos, target_names=nomes_classes, digits=4, zero_division=0)
    matriz = confusion_matrix(y_reais, y_preditos)

    tn, fp, fn, tp = matriz.ravel()
    sensibilidade = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    especificidade = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    total_n = len(dataset_treino) + len(dataset_teste)

    logger.log(f"\n>>> RESULTADOS FINAIS - FULL UNFREEZING (REGIME N={total_n}) <<<")
    logger.log(f"Amostras: Treino = {len(dataset_treino)} | Teste = {len(dataset_teste)}")
    logger.log(f"Acurácia Geral: {acc_final*100:.2f}% | F1-Macro: {f1_macro_final:.4f} | ROC-AUC: {auc_final:.4f}")
    logger.log(f"Sensibilidade: {sensibilidade*100:.2f}% | Especificidade: {especificidade*100:.2f}%")
    logger.log(f"Tempo Total:   {tempo_total:.2f} s")
    logger.log("\n--- Relatório de Classificação ---")
    logger.log(relatorio)
    logger.log("--- Matriz de Confusão ---")
    logger.log(str(matriz))
    logger.log("=" * 80 + "\n")

    return {
        "estrategia": "full_unfreezing",
        "tarefa": tag_tarefa,
        "regime": n_regime if n_regime is not None else total_n,
        "classes": nomes_classes,
        "treino": len(dataset_treino),
        "teste": len(dataset_teste),
        "tempo_s": tempo_total,
        "accuracy": float(acc_final),
        "f1_macro": float(f1_macro_final),
        "roc_auc": float(auc_final),
        "sensibilidade": float(sensibilidade),
        "especificidade": float(especificidade),
        "matriz_confusao": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "historico_epocas": historico_epocas,
    }


def resolver_caminho_dados(caminho_fornecido: Optional[str] = None) -> Path:
    """Localiza o diretório do dataset de ressonâncias."""
    candidatos = [
        Path(caminho_fornecido) if caminho_fornecido else None,
        Path("data/processed_padding"),
        Path("data/processed"),
        Path("data/Alzheimer_MRI_4_Classes_Dataset"),
        _RAIZ_PROJETO / "data" / "processed_padding",
    ]
    for c in candidatos:
        if c and c.exists() and (c / "NonDemented").exists():
            return c.resolve()

    raise FileNotFoundError(
        "Não foi possível localizar o dataset de Alzheimer MRI. Verifique se data/processed_padding existe."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treinamento ResNet-50 RadImageNet com Full Unfreezing (sem aquecimento congelado)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Caminho para o dataset de MRI pré-processado.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="moderate",
        choices=["moderate", "grouped", "both"],
        help="Tarefa binária: 'moderate' (Extremos) ou 'grouped' (Agrupado).",
    )
    parser.add_argument(
        "--regimes",
        type=int,
        nargs="+",
        default=None,
        help="Lista de tamanhos de regimes (ex: --regimes 1000 5000 10000 20000).",
    )
    parser.add_argument(
        "--all_regimes",
        action="store_true",
        help="Executa o benchmark nos 4 regimes de dados: 1000, 5000, 10000 e 20000.",
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Para a tarefa 'grouped', usa todas as imagens sem balanceamento estrito.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Quantidade de épocas de treinamento (padrão: 10).")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamanho do batch (padrão: 64 para RTX 5060).")
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-5,
        help="Learning rate do backbone (padrão: 1e-5 - taxa ultrabaixa para proteger os filtros pré-treinados contra gradientes ruidosos).",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=1e-4,
        help="Learning rate da cabeça classificadora (padrão: 1e-4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Define taxa uniforme única para toda a rede (ex: --lr 1e-5).",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Decaimento de peso L2 (padrão: 1e-4).")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers do DataLoader (padrão: 2 para Windows).")
    parser.add_argument("--patience", type=int, default=4, help="Paciência do Early Stopping (padrão: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Semente de reprodutibilidade (padrão: 42).")
    parser.add_argument("--output_dir", type=str, default="outputs/full_unfreeze", help="Pasta de saída para checkpoints e logs.")
    parser.add_argument("--log_file", type=str, default=None, help="Arquivo de log personalizado.")

    args = parser.parse_args()

    # Se --lr foi especificado, aplica taxa uniforme para backbone e head
    backbone_lr = args.lr if args.lr is not None else args.backbone_lr
    head_lr = args.lr if args.lr is not None else args.head_lr

    pasta_saida = Path(args.output_dir)
    pasta_logs = pasta_saida / "logs"
    pasta_logs.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho_log = Path(args.log_file) if args.log_file else (pasta_logs / f"treinamento_full_unfreeze_{args.task}_{timestamp}.txt")

    logger = DualLogger(caminho_log)
    logger.log("=== SESSÃO DE TREINAMENTO: RESNET-50 FULL UNFREEZING (RADIMAGENET) ===")
    logger.log(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Arquivo de Log: {caminho_log.resolve()}\n")

    caminho_dados = resolver_caminho_dados(args.data_dir)
    balancear = not args.no_balance

    if args.all_regimes:
        lista_regimes = [1000, 5000, 10000, 20000]
    elif args.regimes:
        lista_regimes = args.regimes
    else:
        lista_regimes = [None]

    try:
        tarefas_para_rodar = ["moderate", "grouped"] if args.task == "both" else [args.task]
        todos_resultados = {}

        for t in tarefas_para_rodar:
            todos_resultados[t] = []
            for reg in lista_regimes:
                res = executar_experimento_full_unfreeze(
                    caminho_dados=caminho_dados,
                    tarefa=t,
                    n_regime=reg,
                    balancear_agrupamento=balancear,
                    epocas=args.epochs,
                    batch_size=args.batch_size,
                    backbone_lr=backbone_lr,
                    head_lr=head_lr,
                    weight_decay=args.weight_decay,
                    num_workers=args.num_workers,
                    paciencia=args.patience,
                    seed=args.seed,
                    pasta_saida=pasta_saida,
                    logger=logger,
                )
                todos_resultados[t].append(res)

        for t, res_lista in todos_resultados.items():
            nome_exp = "Extremos (NonDemented vs ModerateDemented)" if t == "moderate" else "Agrupado (NonDemented vs Demented)"
            logger.log("=" * 105)
            logger.log(f"TABELA DE RESULTADOS (FULL UNFREEZING) - {nome_exp.upper()}")
            logger.log("=" * 105)
            logger.log(
                f"{'Regime (N)':<12} | {'N Treino':<10} | {'Tempo (s)':<11} | {'Acurácia':<10} | "
                f"{'Sensibilidade':<15} | {'Especificidade':<16} | {'F1-Score':<8}"
            )
            logger.log("-" * 105)
            for r in res_lista:
                logger.log(
                    f"{r['regime']:<12} | {r['treino']:<10} | {r['tempo_s']:<10.1f}s | "
                    f"{r['accuracy']*100:<9.2f}% | {r['sensibilidade']*100:<14.2f}% | "
                    f"{r['especificidade']*100:<15.2f}% | {r['f1_macro']:<8.4f}"
                )
            logger.log("=" * 105 + "\n")

            caminho_json = pasta_saida / f"relatorio_regimes_resnet50_full_unfreeze_{t}.json"
            with open(caminho_json, "w", encoding="utf-8") as f_json:
                json.dump({"estrategia": "full_unfreezing", "tarefa": t, "timestamp": datetime.now().isoformat(), "resultados": res_lista}, f_json, indent=4)
            logger.log(f"[INFO] Relatório Full Unfreezing salvo em: {caminho_json}\n")

    finally:
        logger.log(f"\nSessão finalizada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.close()


if __name__ == "__main__":
    main()
