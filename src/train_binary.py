"""
Script de Treinamento Binário da ResNet-50 com Pesos RadImageNet para Alzheimer MRI.

Suporta duas formulações de classificação binária essenciais para o artigo:
1. 'moderate_vs_nondemented' (Extremos):
   - Classe 0: NonDemented (10.000 imagens)
   - Classe 1: ModerateDemented (10.000 imagens)
   - Exclui os estágios intermediários (Mild e VeryMild) para avaliar o contraste máximo.

2. 'demented_vs_nondemented' (Agrupamento Completo):
   - Classe 0: NonDemented (10.000 imagens)
   - Classe 1: Demented (agrupando MildDemented + ModerateDemented + VeryMildDemented)
   - Permite modo balanceado (10k vs 10k) ou full (10k vs 30k ponderado).

Otimizado para aceleração por hardware na NVIDIA GeForce RTX 5060 (8GB):
- Precisão Mista Automática (AMP) nativa com bfloat16.
- Pipeline PyTorch otimizado (cuDNN benchmark, pin_memory, prefetch_factor).
- Treinamento em 2 fases (warmup com backbone congelado + fine-tuning diferencial).
- Gravação detalhada de logs em .txt e salvamento de checkpoints.
"""

import argparse
from datetime import datetime
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

# Garante que a raiz do projeto esteja no sys.path
_RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(_RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROJETO))

# Importação dos módulos do projeto
from src.models.resnet50 import ResNet50Classifier
from utils.preprocess_mri import redimensionar_com_padding_simetrico


class DualLogger:
    """
    Logger que duplica toda a saída tanto para o console (stdout) quanto
    para um arquivo de texto (.txt) persistente em disco.
    """

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
    """
    Transformação customizada para imagens brutas caso necessário.
    """

    def __init__(self, tamanho: int = 224) -> None:
        self.tamanho = tamanho

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.tamanho, self.tamanho):
            return redimensionar_com_padding_simetrico(img, tamanho_alvo=self.tamanho)
        return img


def fixar_sementes(seed: int = 42) -> None:
    """Fixa sementes para reprodutibilidade científica."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configurar_ambiente_rtx5060(logger: DualLogger) -> Tuple[torch.device, torch.dtype]:
    """Configura o hardware para máxima performance na RTX 5060 8GB."""
    if not torch.cuda.is_available():
        logger.log("[AVISO] CUDA não detectada! Executando na CPU (lento).")
        return torch.device("cpu"), torch.float32

    dispositivo = torch.device("cuda:0")
    nome_gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    logger.log("=" * 75)
    logger.log("CONFIGURAÇÃO DE ACELERAÇÃO DE HARDWARE (RTX 5060)")
    logger.log("=" * 75)
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
        logger.log("Precisão Mista (AMP):  torch.float16 (com GradScaler)")

    logger.log("=" * 75)
    return dispositivo, dtype_amp


class BinaryDatasetWrapper(Dataset):
    """
    Dataset wrapper que mapeia imagens e novos rótulos binários (0 ou 1)
    com transformações específicas de treino ou validação.
    """

    def __init__(
        self,
        base_dataset: ImageFolder,
        indices: np.ndarray,
        labels_binarios: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.labels_binarios = labels_binarios
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        indice_real = self.indices[idx]
        caminho_img, _ = self.base_dataset.samples[indice_real]

        with Image.open(caminho_img) as img:
            if self.transform is not None:
                img_t = self.transform(img)
            else:
                img_t = transforms.ToTensor()(img)

        label_bin = int(self.labels_binarios[idx])
        return img_t, label_bin

    def __len__(self) -> int:
        return len(self.indices)


def preparar_datasets_binarios(
    caminho_dados: Path,
    tarefa: str,
    n_regime: Optional[int] = None,
    balancear_agrupamento: bool = True,
    proporcao_teste: float = 0.2,
    seed: int = 42,
    logger: Optional[DualLogger] = None,
) -> Tuple[Dataset, Dataset, List[str], Optional[torch.Tensor]]:
    """
    Cria a partição e o mapeamento binário conforme a formulação solicitada,
    com suporte a regimes de dados (1000, 5000, 10000, 20000).

    Tarefas:
    1. 'moderate': NonDemented (0) vs ModerateDemented (1) [até 20k imagens]
    2. 'grouped': NonDemented (0) vs Demented (1) [Mild+Moderate+VeryMild]
    """
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
    classes_originais = dataset_bruto.classes  # ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
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
            logger.log(f"Configuração da Tarefa:  Extremos (NonDemented vs ModerateDemented)")
            if n_regime:
                logger.log(f"Regime Selecionado:      N = {len(indices_filtrados)}")
            logger.log(f"Amostras NonDemented:    {(labels_binarios == 0).sum()}")
            logger.log(f"Amostras Moderate:       {(labels_binarios == 1).sum()}")
            logger.log(f"Total Selecionado:       {len(indices_filtrados)}")

    elif tarefa == "grouped":
        nomes_classes = ["NonDemented", "Demented"]
        if balancear_agrupamento:
            total_disponivel = 20000
            if n_regime is not None and n_regime < total_disponivel:
                n_non = n_regime // 2
                n_dem = n_regime - n_non
                n_mild = n_dem // 3 + (1 if n_dem % 3 > 0 else 0)
                n_mod_sub = n_dem // 3 + (1 if n_dem % 3 > 1 else 0)
                n_verymild = n_dem - n_mild - n_mod_sub

                sub_non = rng.choice(indices_non, size=n_non, replace=False)
                sub_mild = rng.choice(indices_mild, size=n_mild, replace=False)
                sub_mod = rng.choice(indices_mod, size=n_mod_sub, replace=False)
                sub_verymild = rng.choice(indices_verymild, size=n_verymild, replace=False)

                indices_demented = np.concatenate([sub_mild, sub_mod, sub_verymild])
                indices_filtrados = np.concatenate([sub_non, indices_demented])
                labels_binarios = np.concatenate([np.zeros(len(sub_non)), np.ones(len(indices_demented))])
            else:
                sub_mild = rng.choice(indices_mild, size=3334, replace=False)
                sub_mod = rng.choice(indices_mod, size=3333, replace=False)
                sub_verymild = rng.choice(indices_verymild, size=3333, replace=False)

                indices_demented = np.concatenate([sub_mild, sub_mod, sub_verymild])
                indices_filtrados = np.concatenate([indices_non, indices_demented])
                labels_binarios = np.concatenate([np.zeros(len(indices_non)), np.ones(len(indices_demented))])

            if logger:
                logger.log("Configuração da Tarefa:  Agrupamento Balanceado (NonDemented vs Demented)")
                if n_regime:
                    logger.log(f"Regime Selecionado:      N = {len(indices_filtrados)}")
                logger.log(f"Amostras NonDemented:    {(labels_binarios == 0).sum()}")
                logger.log(f"Amostras Demented:       {(labels_binarios == 1).sum()}")
                logger.log(f"Total Selecionado:       {len(indices_filtrados)}")
        else:
            indices_demented = np.concatenate([indices_mild, indices_mod, indices_verymild])
            indices_filtrados = np.concatenate([indices_non, indices_demented])
            labels_binarios = np.concatenate([np.zeros(len(indices_non)), np.ones(len(indices_demented))])
            pesos_classe = torch.tensor([3.0, 1.0], dtype=torch.float32)

            if logger:
                logger.log("Configuração da Tarefa:  Agrupamento Completo (NonDemented vs Demented - 40k)")
                logger.log(f"Amostras NonDemented:    {len(indices_non)}")
                logger.log(f"Amostras Demented:       {len(indices_demented)}")
                logger.log("Ponderação de Perda:     [3.0, 1.0] para equilibrar prevalência 1:3")
    else:
        raise ValueError(f"Tarefa desconhecida: {tarefa}. Escolha 'moderate' ou 'grouped'.")

    # Divisão Treino/Teste estratificada (80% treino, 20% teste)
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
    """Executa uma época de treino binário acelerado."""
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
    """Avalia o classificador binário retornando Perda, Acurácia, F1, ROC-AUC e predições."""
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

            todas_predicoes.extend(preds.cpu().numpy().tolist())
            todos_rotulos.extend(rotulos.cpu().numpy().tolist())
            todas_probs_positivas.extend(probs[:, 1].float().cpu().numpy().tolist())

    perda_media = perda_total / total
    acuracia = acertos / total
    y_true = np.array(todos_rotulos)
    y_pred = np.array(todas_predicoes)
    y_prob = np.array(todas_probs_positivas)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_score = 0.0

    return perda_media, acuracia, f1_macro, auc_score, y_true, y_pred, y_prob


def executar_experimento_binario(
    caminho_dados: Path,
    tarefa: str,
    n_regime: Optional[int] = None,
    balancear_agrupamento: bool = True,
    epocas: int = 15,
    epocas_congeladas: int = 2,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    paciencia: int = 5,
    seed: int = 42,
    pasta_saida: Path = Path("outputs"),
    logger: Optional[DualLogger] = None,
) -> Dict[str, float]:
    """Executa o ciclo completo de treino e teste binário para um determinado regime."""
    dispositivo, dtype_amp = configurar_ambiente_rtx5060(logger)
    fixar_sementes(seed)

    tag_tarefa = f"binary_{tarefa}" + ("_balanced" if (tarefa == "grouped" and balancear_agrupamento) else "")
    tag_regime = f"_regime_{n_regime}" if n_regime else ""
    pasta_checkpoints = pasta_saida / "checkpoints"
    pasta_checkpoints.mkdir(parents=True, exist_ok=True)
    caminho_melhor_modelo = pasta_checkpoints / f"best_resnet50_{tag_tarefa}{tag_regime}.pt"

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

    logger.log("\nInstanciando ResNet-50 Binária (num_classes=2) com RadImageNet...")
    modelo = ResNet50Classifier(
        num_classes=2,
        carregar_radimagenet=True,
        dropout_rate=0.3,
        device=dispositivo,
    ).to(dispositivo)

    if epocas_congeladas > 0:
        logger.log(f"[FASE 1] Congelando backbone pelas primeiras {epocas_congeladas} épocas...")
        modelo.freeze_backbone()
    else:
        logger.log("[FASE 1] Fine-tuning direto em toda a rede.")

    if pesos_classe is not None and dispositivo.type == "cuda":
        criterio = nn.CrossEntropyLoss(weight=pesos_classe.to(dispositivo))
    else:
        criterio = nn.CrossEntropyLoss()

    grupos_params = modelo.get_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
    otimizador = AdamW(grupos_params)
    scheduler = CosineAnnealingLR(otimizador, T_max=epocas, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if (dispositivo.type == "cuda" and dtype_amp == torch.float16) else None

    logger.log("=" * 80)
    logger.log(f"TREINAMENTO BINÁRIO ({tag_tarefa.upper()} | {epocas} ÉPOCAS | BATCH: {batch_size})")
    logger.log("=" * 80)
    logger.log(f"{'Época':<7} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val F1-Mac':<10} | {'AUC':<7} | {'Tempo':<8} | {'VRAM'}")
    logger.log("-" * 80)

    melhor_val_f1 = -1.0
    epocas_sem_melhora = 0
    tempo_inicio_total = time.time()

    for epoca in range(1, epocas + 1):
        tempo_inicio_epoca = time.time()

        if epocas_congeladas > 0 and epoca == (epocas_congeladas + 1):
            logger.log(f"\n[FASE 2] Descongelando backbone para Fine-Tuning completo...")
            modelo.unfreeze_backbone()
            grupos_atualizados = modelo.get_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
            otimizador = AdamW(grupos_atualizados)
            scheduler = CosineAnnealingLR(otimizador, T_max=(epocas - epocas_congeladas), eta_min=1e-6)

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

        if f1_macro_val > melhor_val_f1:
            melhor_val_f1 = f1_macro_val
            epocas_sem_melhora = 0
            caminho_melhor_modelo.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoca,
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
                logger.log(f"\n[EARLY STOPPING] Parada antecipada após {paciencia} épocas sem melhora no F1.")
                break

    tempo_total = time.time() - tempo_inicio_total
    logger.log("=" * 80)
    logger.log(f"Treino concluído em {tempo_total:.2f}s ({tempo_total/60:.2f} min).")
    logger.log(f"Melhor modelo salvo em: {caminho_melhor_modelo.resolve()}")
    logger.log("=" * 80)

    # Avaliação Final no Teste com o Melhor Checkpoint
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

    # Cálculo explícito de Sensibilidade e Especificidade
    tn, fp, fn, tp = matriz.ravel()
    sensibilidade = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    especificidade = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    total_n = len(dataset_treino) + len(dataset_teste)

    logger.log(f"\n>>> RESULTADOS FINAIS - {tag_tarefa.upper()} (REGIME N={total_n}) <<<")
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
        "tarefa": tag_tarefa,
        "regime": n_regime if n_regime is not None else total_n,
        "classes": nomes_classes,
        "treino": len(dataset_treino),
        "teste": len(dataset_teste),
        "tempo_s": tempo_total,
        "accuracy": acc_final,
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "f1_macro": f1_macro_final,
        "auc": auc_final,
        "loss": loss_final,
    }


def resolver_caminho_dados(argumento_dados: Optional[str]) -> Path:
    """Detecta automaticamente o diretório de dados."""
    raiz = Path(__file__).resolve().parent.parent
    if argumento_dados:
        p = Path(argumento_dados)
        if not p.is_absolute():
            p = (raiz / p).resolve()
        if p.exists():
            return p

    candidato_processed = raiz / "data" / "processed"
    if candidato_processed.is_dir() and any(p.is_dir() for p in candidato_processed.iterdir() if p.name != ".git"):
        return candidato_processed

    candidato_raw = raiz / "data" / "raw"
    if candidato_raw.is_dir() and any(p.is_dir() for p in candidato_raw.iterdir() if p.name != ".git"):
        return candidato_raw

    return candidato_processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treinamento Binário da ResNet-50 com RadImageNet (NonDemented vs Moderate ou Agrupado) em Regimes de Dados."
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Caminho do diretório de dados (padrão: data/processed).")
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["moderate", "grouped", "both"],
        help="Tarefa binária: 'moderate' (Non vs Mod), 'grouped' (Non vs Todos Dementes), ou 'both' (ambos).",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        type=int,
        default=None,
        help="Lista de regimes N (ex: --regimes 1000 5000 10000 20000).",
    )
    parser.add_argument(
        "--all_regimes",
        action="store_true",
        help="Executa o benchmark nos 4 regimes de dados: 1000, 5000, 10000 e 20000.",
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Para a tarefa 'grouped', usa todas as 30k de Demented (proporção 1:3 com ponderação de perda) em vez de 10k vs 10k.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Quantidade de épocas de treinamento (padrão: 10).")
    parser.add_argument("--freeze_epochs", type=int, default=2, help="Épocas de aquecimento com backbone congelado (padrão: 2).")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamanho do lote de imagens (padrão: 64 para RTX 5060).")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="Learning rate do backbone (padrão: 1e-4).")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Learning rate da cabeça classificadora (padrão: 1e-3).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Decaimento de peso L2 (padrão: 1e-4).")
    parser.add_argument("--num_workers", type=int, default=2, help="Processos para carregamento de dados (padrão: 2 para Windows).")
    parser.add_argument("--patience", type=int, default=4, help="Paciência do Early Stopping (padrão: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Semente de reprodutibilidade (padrão: 42).")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Pasta para salvar saídas.")
    parser.add_argument("--log_file", type=str, default=None, help="Arquivo .txt personalizado para os logs.")

    args = parser.parse_args()

    pasta_saida = Path(args.output_dir)
    pasta_logs = pasta_saida / "logs"
    pasta_logs.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho_log = Path(args.log_file) if args.log_file else (pasta_logs / f"treinamento_resnet50_binario_{args.task}_{timestamp}.txt")

    logger = DualLogger(caminho_log)
    logger.log("=== SESSÃO DE TREINAMENTO BINÁRIO RESNET-50 RADIMAGENET ===")
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
                res = executar_experimento_binario(
                    caminho_dados=caminho_dados,
                    tarefa=t,
                    n_regime=reg,
                    balancear_agrupamento=balancear,
                    epocas=args.epochs,
                    epocas_congeladas=args.freeze_epochs,
                    batch_size=args.batch_size,
                    backbone_lr=args.backbone_lr,
                    head_lr=args.head_lr,
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
            logger.log(f"TABELA DE RESULTADOS POR REGIME - {nome_exp.upper()}")
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

    finally:
        logger.log(f"\nSessão finalizada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.close()


if __name__ == "__main__":
    main()
