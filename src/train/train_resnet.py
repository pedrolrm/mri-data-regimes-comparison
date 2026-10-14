"""
Script de Treinamento da ResNet-50 com Pesos RadImageNet para Alzheimer MRI.

Otimizado especificamente para aceleração por hardware na NVIDIA GeForce RTX 5060 (8GB):
- Precisão Mista Automática (AMP) nativa com bfloat16 (arquitetura Blackwell).
- Otimizações de pipeline PyTorch: torch.backends.cudnn.benchmark, pin_memory,
  prefetch_factor, persistent_workers e zero_grad(set_to_none=True).
- Suporte a Regimes de Dados (N=1000, 5000, 10000 ou Full Dataset) com amostragem
  estratificada para comparação direta com modelos clássicos (SVM/Autoencoder).
- Treinamento em 2 fases (warmup com backbone congelado + fine-tuning com taxas diferenciais).
- Registro detalhado de métricas em tempo real e exportação do log completo para arquivo .txt.
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image

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
    Transformação customizada para imagens brutas caso o usuário não tenha
    rodado o preprocess_mri.py previamente. Aplica o padding dinâmico no Pillow.
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
    """Fixa sementes para reprodutibilidade científica dos experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configurar_ambiente_rtx5060(logger: DualLogger) -> Tuple[torch.device, torch.dtype]:
    """
    Configura e valida os hiperparâmetros de hardware para obter a máxima taxa
    de transferência de tensores na RTX 5060 8GB.
    """
    if not torch.cuda.is_available():
        logger.log("[AVISO] CUDA não detectada! Executando na CPU (lento).")
        return torch.device("cpu"), torch.float32

    dispositivo = torch.device("cuda:0")
    nome_gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    logger.log("=" * 75)
    logger.log("CONFIGURAÇÃO DE ACELERAÇÃO DE HARDWARE")
    logger.log("=" * 75)
    logger.log(f"Dispositivo:           {nome_gpu}")
    logger.log(f"VRAM Total:            {vram_gb:.2f} GB")
    logger.log(f"Versão PyTorch:        {torch.__version__}")
    logger.log(f"Versão CUDA:           {torch.version.cuda}")

    # 1. Ativa cuDNN Benchmark para escolher os melhores algoritmos de convolução
    torch.backends.cudnn.benchmark = True

    # 2. Ativa precisão TensorFloat-32 (TF32) para aceleração de matrizes na arquitetura Blackwell
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        logger.log("Matmul Precision:      high (TF32 habilitado para Tensor Cores)")

    # 3. Detecta suporte nativo a bfloat16
    if torch.cuda.is_bf16_supported():
        dtype_amp = torch.bfloat16
        logger.log("Precisão Mista (AMP):  torch.bfloat16 (Nativo RTX 5060, sem risco de underflow)")
    else:
        dtype_amp = torch.float16
        logger.log("Precisão Mista (AMP):  torch.float16 (com GradScaler)")

    logger.log("=" * 75)
    return dispositivo, dtype_amp


class SubsetImageFolder(Dataset):
    """
    Wrapper para aplicar transformações específicas de treino ou validação
    em um subset gerado a partir de um ImageFolder raiz.
    """

    def __init__(self, subset: Subset, transform: Optional[transforms.Compose] = None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.subset)


def preparar_datasets(
    caminho_dados: Path,
    tamanho_regime: Optional[int] = None,
    proporcao_teste: float = 0.2,
    seed: int = 42,
    logger: Optional[DualLogger] = None,
) -> Tuple[Dataset, Dataset, List[str]]:
    """
    Carrega o dataset e cria a divisão estratificada (treino/teste) de acordo com o regime.
    """
    # Normalização padrão do RadImageNet / ImageNet
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

    # Carrega dataset base sem transformações para aplicar depois da partição
    dataset_bruto = ImageFolder(root=str(caminho_dados))
    nomes_classes = dataset_bruto.classes
    todos_alvos = np.array(dataset_bruto.targets)
    total_amostras = len(dataset_bruto)

    if logger:
        logger.log(f"Diretório dos dados:   {caminho_dados.resolve()}")
        logger.log(f"Classes detectadas:    {nomes_classes}")
        logger.log(f"Total de imagens:      {total_amostras}")

    indices_totais = np.arange(total_amostras)

    # Aplicação de regime de dados (subamostragem balanceada/estratificada)
    if tamanho_regime is not None and tamanho_regime < total_amostras:
        fracao = tamanho_regime / total_amostras
        splitter_regime = StratifiedShuffleSplit(n_splits=1, train_size=fracao, random_state=seed)
        indices_regime, _ = next(splitter_regime.split(indices_totais, todos_alvos))
        alvos_regime = todos_alvos[indices_regime]
        if logger:
            logger.log(f"Regime Selecionado:    N = {tamanho_regime} amostras")
    else:
        indices_regime = indices_totais
        alvos_regime = todos_alvos
        if logger:
            logger.log(f"Regime Selecionado:    Full Dataset (N = {total_amostras})")

    # Divisão treino/teste estratificada (ex: 80% treino, 20% teste)
    splitter_split = StratifiedShuffleSplit(n_splits=1, test_size=proporcao_teste, random_state=seed)
    idx_treino_rel, idx_teste_rel = next(splitter_split.split(indices_regime, alvos_regime))

    indices_treino = indices_regime[idx_treino_rel]
    indices_teste = indices_regime[idx_teste_rel]

    if logger:
        logger.log(f"Divisão de Amostras:   Treino = {len(indices_treino)} | Teste = {len(indices_teste)}")

    subset_treino_raw = Subset(dataset_bruto, indices_treino)
    subset_teste_raw = Subset(dataset_bruto, indices_teste)

    dataset_treino = SubsetImageFolder(subset_treino_raw, transform=transform_treino)
    dataset_teste = SubsetImageFolder(subset_teste_raw, transform=transform_teste)

    return dataset_treino, dataset_teste, nomes_classes


def treinar_uma_epoca(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    dispositivo: torch.device,
    dtype_amp: torch.dtype,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """
    Executa uma época de treinamento com aceleração AMP.
    """
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
            predicoes = modelo(imagens)
            perda = criterio(predicoes, rotulos)

        if usar_scaler:
            scaler.scale(perda).backward()
            scaler.step(otimizador)
            scaler.update()
        else:
            perda.backward()
            otimizador.step()

        perda_total += perda.item() * imagens.size(0)
        _, rotulos_pred = torch.max(predicoes, 1)
        acertos += (rotulos_pred == rotulos).sum().item()
        total += rotulos.size(0)

    perda_media = perda_total / total
    acuracia = acertos / total
    return perda_media, acuracia


def avaliar(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
    dtype_amp: torch.dtype,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Avalia o modelo calculando perda, acurácia, F1-macro e vetores de predições.
    """
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    todas_predicoes: List[int] = []
    todos_rotulos: List[int] = []

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(dispositivo, non_blocking=True)
            rotulos = rotulos.to(dispositivo, non_blocking=True)

            with torch.amp.autocast(device_type="cuda" if dispositivo.type == "cuda" else "cpu", dtype=dtype_amp):
                predicoes = modelo(imagens)
                perda = criterio(predicoes, rotulos)

            perda_total += perda.item() * imagens.size(0)
            _, rotulos_pred = torch.max(predicoes, 1)
            acertos += (rotulos_pred == rotulos).sum().item()
            total += rotulos.size(0)

            todas_predicoes.extend(rotulos_pred.cpu().numpy().tolist())
            todos_rotulos.extend(rotulos.cpu().numpy().tolist())

    perda_media = perda_total / total
    acuracia = acertos / total
    f1_macro = f1_score(todos_rotulos, todas_predicoes, average="macro", zero_division=0)

    return perda_media, acuracia, f1_macro, np.array(todos_rotulos), np.array(todas_predicoes)


def executar_experimento(
    caminho_dados: Path,
    tamanho_regime: Optional[int],
    epocas: int = 25,
    epocas_congeladas: int = 2,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    paciencia: int = 7,
    seed: int = 42,
    pasta_saida: Path = Path("outputs"),
    logger: Optional[DualLogger] = None,
) -> Dict[str, float]:
    """
    Executa o ciclo completo de treino e teste para um regime de dados especificado.
    """
    dispositivo, dtype_amp = configurar_ambiente_rtx5060(logger)
    fixar_sementes(seed)

    tag_regime = f"regime_{tamanho_regime}" if tamanho_regime else "regime_full"
    pasta_checkpoints = pasta_saida / "checkpoints"
    pasta_checkpoints.mkdir(parents=True, exist_ok=True)
    caminho_melhor_modelo = pasta_checkpoints / f"best_resnet50_{tag_regime}.pt"

    # Preparação dos dados
    dataset_treino, dataset_teste, nomes_classes = preparar_datasets(
        caminho_dados=caminho_dados,
        tamanho_regime=tamanho_regime,
        proporcao_teste=0.2,
        seed=seed,
        logger=logger,
    )

    # DataLoaders de alto desempenho para RTX 5060
    loader_treino = DataLoader(
        dataset_treino,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    loader_teste = DataLoader(
        dataset_teste,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    # Instanciação do Modelo ResNet-50 com RadImageNet
    logger.log("\nInstanciando ResNet-50 com pesos do RadImageNet...")
    modelo = ResNet50Classifier(
        num_classes=len(nomes_classes),
        carregar_radimagenet=True,
        dropout_rate=0.3,
        device=dispositivo,
    ).to(dispositivo)

    # Fase 1: Aquecimento (Warmup) com backbone congelado
    if epocas_congeladas > 0:
        logger.log(f"[FASE 1] Congelando backbone convolucional pelas primeiras {epocas_congeladas} épocas...")
        modelo.freeze_backbone()
    else:
        logger.log("[FASE 1] Fine-tuning direto em todos os parâmetros da rede.")

    criterio = nn.CrossEntropyLoss()

    grupos_parametros = modelo.get_parameter_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    otimizador = AdamW(grupos_parametros)
    scheduler = CosineAnnealingLR(otimizador, T_max=epocas, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda") if (dispositivo.type == "cuda" and dtype_amp == torch.float16) else None

    logger.log("=" * 75)
    logger.log(f"INICIANDO TREINAMENTO ({epocas} ÉPOCAS | BATCH SIZE: {batch_size})")
    logger.log("=" * 75)
    logger.log(f"{'Época':<7} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val F1-Mac':<10} | {'Tempo':<8} | {'VRAM (MB)'}")
    logger.log("-" * 75)

    melhor_val_f1 = -1.0
    epocas_sem_melhora = 0
    tempo_inicio_total = time.time()

    for epoca in range(1, epocas + 1):
        tempo_inicio_epoca = time.time()

        # Transição da Fase 1 para a Fase 2 (descongelamento)
        if epocas_congeladas > 0 and epoca == (epocas_congeladas + 1):
            logger.log(f"\n[FASE 2] Descongelando backbone convolucional para Fine-Tuning completo...")
            modelo.unfreeze_backbone()
            # Reconstrói os grupos de parâmetros com taxas de aprendizado apropriadas
            grupos_atualizados = modelo.get_parameter_groups(
                backbone_lr=backbone_lr,
                head_lr=head_lr,
                weight_decay=weight_decay,
            )
            otimizador = AdamW(grupos_atualizados)
            scheduler = CosineAnnealingLR(otimizador, T_max=(epocas - epocas_congeladas), eta_min=1e-6)

        # Passo de treino
        loss_treino, acc_treino = treinar_uma_epoca(
            modelo=modelo,
            dataloader=loader_treino,
            criterio=criterio,
            otimizador=otimizador,
            dispositivo=dispositivo,
            dtype_amp=dtype_amp,
            scaler=scaler,
        )

        # Passo de validação / teste
        loss_val, acc_val, f1_macro_val, _, _ = avaliar(
            modelo=modelo,
            dataloader=loader_teste,
            criterio=criterio,
            dispositivo=dispositivo,
            dtype_amp=dtype_amp,
        )

        scheduler.step()
        duracao_epoca = time.time() - tempo_inicio_epoca

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2)

        # Log da época
        marca_melhor = " *" if f1_macro_val > melhor_val_f1 else ""
        logger.log(
            f"{epoca:<7} | {loss_treino:<10.4f} | {acc_treino*100:<9.2f}% | "
            f"{loss_val:<10.4f} | {acc_val*100:<9.2f}% | {f1_macro_val:<10.4f}{marca_melhor:<2} | "
            f"{duracao_epoca:<7.2f}s | {vram_mb:.0f} MB"
        )

        # Checagem de melhor modelo
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
                "classes": nomes_classes,
            }, caminho_melhor_modelo)
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                logger.log(f"\n[EARLY STOPPING] Treino interrompido após {paciencia} épocas sem melhora no F1-Macro.")
                break

    tempo_total = time.time() - tempo_inicio_total
    logger.log("=" * 75)
    logger.log(f"Treino concluído em {tempo_total:.2f}s ({tempo_total/60:.2f} min).")
    logger.log(f"Melhor modelo salvo em: {caminho_melhor_modelo.resolve()}")
    logger.log("=" * 75)

    # Avaliação Final com o Melhor Checkpoint
    checkpoint = torch.load(caminho_melhor_modelo, map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(checkpoint["model_state_dict"])

    loss_final, acc_final, f1_macro_final, y_reais, y_preditos = avaliar(
        modelo=modelo,
        dataloader=loader_teste,
        criterio=criterio,
        dispositivo=dispositivo,
        dtype_amp=dtype_amp,
    )

    # Relatório de Classificação e Matriz de Confusão no padrão do artigo
    relatorio = classification_report(
        y_reais,
        y_preditos,
        target_names=nomes_classes,
        digits=4,
        zero_division=0,
    )
    matriz = confusion_matrix(y_reais, y_preditos)

    logger.log(f"\n>>> RESULTADOS FINAIS RESNET-50 - {tag_regime.upper()} <<<")
    logger.log(f"Amostras: Treino = {len(dataset_treino)} | Teste = {len(dataset_teste)}")
    logger.log(f"Acurácia Geral: {acc_final*100:.2f}% | F1-Macro: {f1_macro_final:.4f}")
    logger.log("\n--- Relatório de Classificação ---")
    logger.log(relatorio)
    logger.log("--- Matriz de Confusão ---")
    logger.log(str(matriz))
    logger.log("=" * 75 + "\n")

    return {
        "regime": tamanho_regime if tamanho_regime else len(dataset_treino) + len(dataset_teste),
        "accuracy": acc_final,
        "f1_macro": f1_macro_final,
        "loss": loss_final,
    }


def resolver_caminho_dados(argumento_dados: Optional[str]) -> Path:
    """
    Detecta automaticamente o caminho dos dados, priorizando 'data/processed' e
    fazendo fallback para 'data/raw' se necessário.
    """
    raiz = Path(__file__).resolve().parents[2]

    if argumento_dados:
        p = Path(argumento_dados)
        if not p.is_absolute():
            p = (raiz / p).resolve()
        if p.exists():
            return p

    # Prioridade 1: data/processed
    candidato_processed = raiz / "data" / "processed"
    if candidato_processed.is_dir():
        pastas = [p for p in candidato_processed.iterdir() if p.is_dir() and p.name != ".git"]
        if pastas:
            return candidato_processed

    # Prioridade 2: data/raw
    candidato_raw = raiz / "data" / "raw"
    if candidato_raw.is_dir():
        pastas = [p for p in candidato_raw.iterdir() if p.is_dir() and p.name != ".git"]
        if pastas:
            return candidato_raw

    # Retorna padrão mesmo se vazio para mensagem de erro amigável
    return candidato_processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treinamento otimizado da ResNet-50 com pesos RadImageNet na RTX 5060 (8GB)."
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Caminho do diretório de dados (padrão: data/processed ou data/raw).")
    parser.add_argument("--regime", type=int, default=None, help="Tamanho do regime de dados N (ex: 1000, 5000, 10000). Omitir para Full.")
    parser.add_argument("--all_regimes", action="store_true", help="Executa todos os 4 regimes (1000, 5000, 10000, Full) em sequência.")
    parser.add_argument("--epochs", type=int, default=25, help="Quantidade de épocas de treinamento (padrão: 25).")
    parser.add_argument("--freeze_epochs", type=int, default=2, help="Épocas com backbone congelado antes do fine-tuning (padrão: 2).")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamanho do lote de imagens (padrão: 64, ideal para 8GB VRAM).")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="Learning rate do backbone convolucional (padrão: 1e-4).")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="Learning rate da camada classificadora (padrão: 1e-3).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Decaimento de pesos L2 (padrão: 1e-4).")
    parser.add_argument("--num_workers", type=int, default=2, help="Processos para carregamento de dados (padrão: 2 para Windows).")
    parser.add_argument("--patience", type=int, default=7, help="Paciência do Early Stopping (padrão: 7).")
    parser.add_argument("--seed", type=int, default=42, help="Semente de reprodutibilidade (padrão: 42).")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Pasta para salvar logs e checkpoints.")
    parser.add_argument("--log_file", type=str, default=None, help="Nome do arquivo .txt para salvar os logs de treino.")

    args = parser.parse_args()

    pasta_saida = Path(args.output_dir)
    pasta_logs = pasta_saida / "logs"
    pasta_logs.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_file:
        caminho_log = Path(args.log_file)
    else:
        regime_str = f"n{args.regime}" if args.regime else ("all_regimes" if args.all_regimes else "full")
        caminho_log = pasta_logs / f"treinamento_resnet50_{regime_str}_{timestamp}.txt"

    logger = DualLogger(caminho_log)
    logger.log(f"=== SESSÃO DE TREINAMENTO RESNET-50 RADIMAGENET ===")
    logger.log(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log gravado em: {caminho_log.resolve()}\n")

    caminho_dados = resolver_caminho_dados(args.data_dir)

    # Validação da pasta de imagens
    if not caminho_dados.exists() or not any(p.is_dir() for p in caminho_dados.iterdir() if p.name != ".git"):
        logger.log(f"[ERRO CRÍTICO] Diretório de imagens não encontrado ou vazio: {caminho_dados}")
        logger.log("Certifique-se de descompactar o dataset em 'data/raw' e rodar 'python utils/preprocess_mri.py'.")
        logger.close()
        sys.exit(1)

    try:
        if args.all_regimes:
            logger.log(">>> MODO TODOS OS REGIMES ATIVADO (1000, 5000, 10000, FULL) <<<")
            lista_regimes = [1000, 5000, 10000, None]
            resultados = []
            for reg in lista_regimes:
                res = executar_experimento(
                    caminho_dados=caminho_dados,
                    tamanho_regime=reg,
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
                resultados.append(res)

            logger.log("=" * 75)
            logger.log("RESUMO COMPARATIVO DE REGIMES (RESNET-50 RADIMAGENET)")
            logger.log("=" * 75)
            logger.log(f"{'Regime (N)':<15} | {'Acurácia':<12} | {'F1-Macro':<12} | {'Loss':<12}")
            logger.log("-" * 75)
            for r in resultados:
                nome_r = f"N = {r['regime']}"
                logger.log(f"{nome_r:<15} | {r['accuracy']*100:<11.2f}% | {r['f1_macro']:<12.4f} | {r['loss']:<12.4f}")
            logger.log("=" * 75)
        else:
            executar_experimento(
                caminho_dados=caminho_dados,
                tamanho_regime=args.regime,
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
    finally:
        logger.log(f"\nSessão finalizada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.close()


if __name__ == "__main__":
    main()
