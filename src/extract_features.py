"""
Módulo e Script de Extração de Características com ResNet-50 RadImageNet.

Este script é responsável por alimentar imagens pré-processadas de Ressonância Magnética (MRI)
pela arquitetura ResNet-50 pré-treinada no RadImageNet (definida em `src.models.encoder.ResNet50Encoder`),
extraindo o espaço latente de 2048 dimensões obtido após o Global Average Pooling.

Por padrão, opera sobre as classes extremas da Doença de Alzheimer para avaliação de contraste máximo:
- Classe 0: NonDemented (Controle / Saudável)
- Classe 1: ModerateDemented (Demência Moderada)

As representações latentes densas geradas são salvas em arquivos NumPy (.npy) juntamente com
metadados estruturados (.json) para alimentação e treinamento de classificadores Support Vector Machine (SVM).
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

# Garante que a raiz do projeto esteja no sys.path para importações modulares
_RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(_RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROJETO))

from src.models.encoder import ResNet50Encoder


class DatasetExtracaoMRI(Dataset):
    """
    Dataset customizado para extração de características de imagens MRI.
    
    Filtra rigorosamente apenas as classes solicitadas (ex.: NonDemented e ModerateDemented)
    e garante que qualquer imagem seja convertida para RGB de 3 canais e padronizada para 224x224.
    """

    def __init__(
        self,
        diretorio_dados: Union[str, Path],
        classes_selecionadas: Sequence[str] = ("NonDemented", "ModerateDemented"),
        mapeamento_classes: Optional[Dict[str, int]] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.diretorio_dados = Path(diretorio_dados)
        self.classes_selecionadas = list(classes_selecionadas)
        
        # Mapeamento padrão: ordem de classes_selecionadas (ex.: NonDemented -> 0, ModerateDemented -> 1)
        if mapeamento_classes is not None:
            self.class_to_idx = mapeamento_classes
        else:
            self.class_to_idx = {cls_nome: idx for idx, cls_nome in enumerate(self.classes_selecionadas)}
        
        self.idx_to_class = {idx: cls_nome for cls_nome, idx in self.class_to_idx.items()}
        
        # Transformação padrão se não especificada
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        self.amostras: List[Tuple[Path, int]] = []
        self._indexar_amostras()

    def _indexar_amostras(self) -> None:
        """Varre o diretório e indexa os arquivos pertencentes às classes selecionadas."""
        extensoes_validas = {".jpg", ".jpeg", ".png", ".bmp"}

        for nome_classe in self.classes_selecionadas:
            pasta_classe = self.diretorio_dados / nome_classe
            if not pasta_classe.exists():
                raise FileNotFoundError(
                    f"A pasta da classe '{nome_classe}' não foi encontrada em: {pasta_classe}"
                )

            rotulo = self.class_to_idx[nome_classe]
            arquivos = sorted([
                p for p in pasta_classe.iterdir()
                if p.is_file() and p.suffix.lower() in extensoes_validas
            ])

            for arq in arquivos:
                self.amostras.append((arq, rotulo))

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        caminho_imagem, rotulo = self.amostras[idx]

        with Image.open(caminho_imagem) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensor_imagem = self.transform(img)

        return tensor_imagem, rotulo, caminho_imagem.name


def extrair_espaco_latente(
    diretorio_dados: Union[str, Path] = "data/processed",
    diretorio_saida: Union[str, Path] = "data/features_extreme",
    classes: Sequence[str] = ("NonDemented", "ModerateDemented"),
    batch_size: int = 128,
    num_workers: int = 2,
    device: Optional[Union[str, torch.device]] = None,
    carregar_radimagenet: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Executa a extração do espaço latente de informações (2048 dimensões)
    utilizando a ResNet50Encoder pré-treinada no RadImageNet.

    Parâmetros:
        diretorio_dados: Caminho raiz contendo as pastas das classes de imagens.
        diretorio_saida: Caminho onde serão gravados os arquivos .npy e .json.
        classes: Tupla com as classes a extrair (padrão: NonDemented e ModerateDemented).
        batch_size: Tamanho do lote de inferência.
        num_workers: Quantidade de processos para carregamento assíncrono dos dados.
        device: Dispositivo de execução ('cuda' ou 'cpu'). Se None, detecta automaticamente.
        carregar_radimagenet: Se True, carrega os pesos médicos do RadImageNet.
        cache_dir: Diretório local de cache dos pesos do Hugging Face.

    Retorna:
        Tuple contendo (matriz_features, vetor_labels, metadados).
    """
    caminho_dados = Path(diretorio_dados)
    caminho_saida = Path(diretorio_saida)
    caminho_saida.mkdir(parents=True, exist_ok=True)

    if device is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dispositivo = torch.device(device)

    print("=" * 70)
    print("EXTRAÇÃO DE CARACTERÍSTICAS COM RESNET-50 RADIMAGENET")
    print("=" * 70)
    print(f"Diretório de dados:    {caminho_dados}")
    print(f"Diretório de saída:    {caminho_saida}")
    print(f"Classes selecionadas:  {list(classes)}")
    print(f"Dispositivo:           {dispositivo}")
    if dispositivo.type == "cuda":
        print(f"Nome da GPU:           {torch.cuda.get_device_name(0)}")
    print(f"Batch Size:            {batch_size}")
    print("=" * 70)

    # 1. Instancia o dataset com as classes selecionadas
    dataset = DatasetExtracaoMRI(
        diretorio_dados=caminho_dados,
        classes_selecionadas=classes
    )
    print(f"Total de amostras localizadas: {len(dataset):,}")
    print(f"Mapeamento de rótulos (class_to_idx): {dataset.class_to_idx}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dispositivo.type == "cuda"),
        drop_last=False
    )

    # 2. Instancia o encoder oficial do repositório
    print("\n[INFO] Inicializando a arquitetura ResNet50Encoder...")
    encoder = ResNet50Encoder(
        carregar_radimagenet=carregar_radimagenet,
        congelar_parametros=True,
        cache_dir=cache_dir,
        device=dispositivo
    )
    encoder.to(dispositivo)
    encoder.eval()

    lista_features: List[np.ndarray] = []
    lista_labels: List[np.ndarray] = []
    lista_nomes: List[str] = []

    # 3. Loop de inferência otimizado sem gradientes
    print("\n[INFO] Iniciando extração do espaço latente...")
    t_inicio = time.perf_counter()

    with torch.no_grad():
        for imagens, rotulos, nomes in tqdm(dataloader, desc="Extraindo Características"):
            imagens = imagens.to(dispositivo, non_blocking=True)
            vetores_latentes = encoder(imagens)  # (B, 2048)

            lista_features.append(vetores_latentes.detach().cpu().numpy().astype(np.float32))
            lista_labels.append(rotulos.numpy().astype(np.int64))
            lista_nomes.extend(nomes)

    tempo_total = time.perf_counter() - t_inicio
    matriz_features = np.concatenate(lista_features, axis=0)
    vetor_labels = np.concatenate(lista_labels, axis=0)
    vetor_nomes = np.array(lista_nomes, dtype=object)

    velocidade = len(dataset) / tempo_total if tempo_total > 0 else 0.0

    print("\n" + "=" * 70)
    print("RESUMO DA EXTRAÇÃO:")
    print("=" * 70)
    print(f"Tempo total:              {tempo_total:.2f} s ({velocidade:.1f} imagens/s)")
    print(f"Formato da matriz X:      {matriz_features.shape} (amostras, dimensões)")
    print(f"Formato do vetor y:       {vetor_labels.shape}")
    print(f"Tipo de dados (features): {matriz_features.dtype}")
    for cls_nome, idx in dataset.class_to_idx.items():
        qtd = int(np.sum(vetor_labels == idx))
        print(f"  - {cls_nome:<18} (idx {idx}): {qtd:>6} amostras")
    print("=" * 70)

    # 4. Gravação dos arquivos para alimentação do SVM
    caminho_features = caminho_saida / "features.npy"
    caminho_labels = caminho_saida / "labels.npy"
    caminho_filenames = caminho_saida / "filenames.npy"
    caminho_classes_json = caminho_saida / "classes.json"

    print(f"[INFO] Salvando arrays NumPy em {caminho_saida}...")
    np.save(caminho_features, matriz_features)
    np.save(caminho_labels, vetor_labels)
    np.save(caminho_filenames, vetor_nomes)

    metadados = {
        "timestamp": datetime.now().isoformat(),
        "classes_selecionadas": list(classes),
        "class_to_idx": dataset.class_to_idx,
        "idx_to_class": {str(k): v for k, v in dataset.idx_to_class.items()},
        "total_amostras": int(len(vetor_labels)),
        "dimensao_caracteristicas": int(matriz_features.shape[1]),
        "contagem_por_classe": {
            cls_nome: int(np.sum(vetor_labels == idx))
            for cls_nome, idx in dataset.class_to_idx.items()
        },
        "tempo_extracao_segundos": round(tempo_total, 2),
        "arquivos": {
            "features": caminho_features.name,
            "labels": caminho_labels.name,
            "filenames": caminho_filenames.name
        }
    }

    with open(caminho_classes_json, "w", encoding="utf-8") as f:
        json.dump(metadados, f, indent=4, ensure_ascii=False)

    print(f"[SUCESSO] Extração concluída com sucesso! Metadados gravados em {caminho_classes_json}")
    return matriz_features, vetor_labels, metadados


def main() -> None:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Extração de Características MRI com ResNet-50 RadImageNet para Classificação SVM."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Caminho do diretório com imagens pré-processadas (padrão: data/processed)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features_extreme",
        help="Caminho de saída para salvar os vetores latentes (padrão: data/features_extreme)"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["NonDemented", "ModerateDemented"],
        help="Classes a serem extraídas (padrão: NonDemented ModerateDemented)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Tamanho do batch para inferência (padrão: 128)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Número de workers do DataLoader (padrão: 2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de execução ('cuda' ou 'cpu')"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Diretório de cache local para os pesos Hugging Face"
    )

    args = parser.parse_args()

    extrair_espaco_latente(
        diretorio_dados=args.data_dir,
        diretorio_saida=args.output_dir,
        classes=args.classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
