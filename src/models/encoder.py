"""
Módulo de Extração de Características com ResNet-50 pré-treinada no RadImageNet.

Este módulo implementa a classe `ResNet50Encoder`, projetada especificamente para
pesquisas médicas de classificação de Alzheimer com imagens de Ressonância Magnética (MRI).
A rede atua como um extrator de características estático, convertendo imagens padronizadas
de 224x224 (3 canais) em vetores densos de 2048 dimensões para consumo por classificadores
como o Support Vector Machine (SVM).
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download


class ResNet50Encoder(nn.Module):
    """
    Extrator de características baseado na arquitetura ResNet-50 com pesos pré-treinados
    no RadImageNet (dataset em larga escala de imagens médicas radiológicas).

    A camada final totalmente conectada (`fc`) é substituída por `nn.Identity()`,
    retornando diretamente o vetor de 2048 dimensões obtido após o Global Average Pooling.
    Por padrão, todos os parâmetros são congelados (`requires_grad = False`).

    Atributos:
        model (torchvision.models.ResNet): Instância da ResNet-50 modificada.
        dim_saida (int): Dimensão do vetor de características de saída (2048).
    """

    def __init__(
        self,
        carregar_radimagenet: bool = True,
        congelar_parametros: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Inicializa o extrator de características ResNet-50.

        Parâmetros:
            carregar_radimagenet (bool): Se True, baixa e carrega os pesos médicos do RadImageNet.
            congelar_parametros (bool): Se True, desativa o cálculo de gradientes em toda a rede.
            cache_dir (str ou Path, opcional): Diretório local para salvar o download dos pesos do Hugging Face.
            device (str ou torch.device, opcional): Dispositivo para mapeamento dos pesos ('cpu', 'cuda', etc.).
        """
        super().__init__()

        self.dim_saida: int = 2048
        self._device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Instancia a arquitetura ResNet-50 pura sem pesos ImageNet padrão
        self.model = models.resnet50(weights=None)

        # 2. Carrega os pesos especializados do RadImageNet, se solicitado
        if carregar_radimagenet:
            self._carregar_pesos_radimagenet(cache_dir=cache_dir)

        # 3. Substitui a camada de classificação por Identity para retornar o vetor cru de 2048 dimensões
        self.model.fc = nn.Identity()

        # 4. Congela os pesos da rede por padrão para atuar como extrator estático
        if congelar_parametros:
            self.congelar_pesos()

        # Define a rede em modo de avaliação (desativa dropout e fixa estatísticas de batch normalization)
        self.eval()

    def _carregar_pesos_radimagenet(
        self,
        repo_id: str = "Lab-Rasool/RadImageNet",
        filename: str = "ResNet50.pt",
        cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Método interno para baixar e carregar com segurança os pesos pré-treinados
        do repositório Hugging Face do RadImageNet.

        Trata variações de formatação do checkpoint (remoção de prefixos 'module.' ou 'backbone.'
        e descarte de pesos da camada de classificação original 'fc').

        Parâmetros:
            repo_id (str): Identificador do repositório no Hugging Face Hub.
            filename (str): Nome do arquivo de pesos a ser baixado.
            cache_dir (str ou Path, opcional): Diretório de cache local para os pesos.
        """
        print(f"[INFO] Baixando/carregando pesos RadImageNet ({repo_id}/{filename})...")

        # Baixa ou recupera do cache local usando a biblioteca huggingface_hub
        caminho_pesos = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir) if cache_dir else None
        )

        # Carregamento seguro do checkpoint com mapeamento para o dispositivo alvo
        checkpoint = torch.load(caminho_pesos, map_location=self._device, weights_only=False)

        # Trata o formato do checkpoint (objeto de modelo completo ou dicionário state_dict)
        if isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            raise TypeError(f"Formato do checkpoint não suportado: {type(checkpoint)}")

        # Limpeza e compatibilização do state_dict
        state_dict_limpo = {}
        for chave, tensor in state_dict.items():
            chave_limpa = chave

            # Remove prefixos gerados por DataParallel ou wrappers de treinamento
            if chave_limpa.startswith("module."):
                chave_limpa = chave_limpa[len("module."):]
            if chave_limpa.startswith("backbone."):
                chave_limpa = chave_limpa[len("backbone."):]

            # Descarta os pesos da camada 'fc' do modelo pré-treinado original
            if chave_limpa.startswith("fc."):
                continue

            state_dict_limpo[chave_limpa] = tensor

        # Carrega os pesos na arquitetura ResNet-50 com strict=False para tolerar a ausência da camada fc
        mensagem_carregamento = self.model.load_state_dict(state_dict_limpo, strict=False)
        print(f"[SUCESSO] Pesos do RadImageNet carregados com sucesso! (Chaves ausentes esperadas: {mensagem_carregamento.missing_keys})")

    def congelar_pesos(self) -> None:
        """
        Congela todos os parâmetros da rede, desativando a computação de gradientes
        (`requires_grad = False`) para que a rede atue estritamente como extrator de características.
        """
        for parametro in self.model.parameters():
            parametro.requires_grad = False

    def descongelar_pesos(self) -> None:
        """
        Descongela todos os parâmetros da rede, permitindo ajuste fino (fine-tuning) se necessário.
        """
        for parametro in self.model.parameters():
            parametro.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa o fluxo de inferência para extração de características.

        Parâmetros:
            x (torch.Tensor): Tensor de entrada no formato (Batch_Size, 3, 224, 224).

        Retorna:
            torch.Tensor: Tensor contendo os vetores de características com formato (Batch_Size, 2048).
        """
        # A ResNet-50 executa convoluções -> BatchNorm -> Pooling Global -> fc (Identity)
        caracteristicas = self.model(x)
        return caracteristicas


if __name__ == "__main__":
    print("=" * 70)
    print("Demonstração da Arquitetura ResNet50Encoder")
    print("=" * 70)

    # Instancia a arquitetura sem baixar os pesos para teste rápido de fluxo
    encoder = ResNet50Encoder(carregar_radimagenet=False, congelar_parametros=True)

    # Tensor sintético simulando um lote de 2 imagens MRI pré-processadas (224x224, 3 canais)
    lote_simulado = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        features = encoder(lote_simulado)

    print(f"Shape de entrada: {lote_simulado.shape}")
    print(f"Shape do vetor de características extraído: {features.shape}")
    print(f"Dimensão esperada por imagem: {encoder.dim_saida}")
    print(f"Todos os parâmetros congelados: {all(not p.requires_grad for p in encoder.parameters())}")
    print("=" * 70)
