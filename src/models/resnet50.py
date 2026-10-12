"""
Módulo da Rede Convolucional ResNet-50 com pesos pré-treinados no RadImageNet.

Este módulo implementa o modelo `ResNet50Classifier`, projetado especificamente para
classificação multiclasse de Ressonâncias Magnéticas (MRI) no diagnóstico dos estágios
da Doença de Alzheimer (MildDemented, ModerateDemented, NonDemented, VeryMildDemented).

A rede utiliza os pesos de transferência médica do RadImageNet (Lab-Rasool/RadImageNet)
com mapeamento completo das camadas convolucionais para a arquitetura do PyTorch/Torchvision,
permitindo tanto linear probing (backbone congelado) quanto fine-tuning com taxas de aprendizado
diferenciais.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download


class ResNet50Classifier(nn.Module):
    """
    Classificador Convolucional baseado na arquitetura ResNet-50 com inicialização
    pré-treinada especializada no RadImageNet (dataset radiológico de larga escala).

    A cabeça original de 1000 classes do ImageNet é substituída por um classificador
    multiclasse adaptado com Dropout para regularização e Linear(2048, num_classes).

    Atributos:
        num_classes (int): Quantidade de classes alvo (padrão: 4 para Alzheimer MRI).
        dropout_rate (float): Taxa de dropout antes da camada totalmente conectada.
        model (torchvision.models.ResNet): Instância da ResNet-50 do Torchvision.
    """

    def __init__(
        self,
        num_classes: int = 4,
        carregar_radimagenet: bool = True,
        dropout_rate: float = 0.3,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Inicializa o classificador ResNet-50.

        Parâmetros:
            num_classes (int): Número de classes de saída (padrão: 4).
            carregar_radimagenet (bool): Se True, baixa e carrega os pesos médicos do RadImageNet.
            dropout_rate (float): Probabilidade de desativação de neurônios no Dropout (padrão: 0.3).
            cache_dir (str ou Path, opcional): Diretório local para cache dos pesos baixados.
            device (str ou torch.device, opcional): Dispositivo alvo ('cuda' ou 'cpu').
        """
        super().__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self._device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Instancia a arquitetura ResNet-50 padrão do Torchvision
        self.model = models.resnet50(weights=None)

        # 2. Carrega e mapeia com precisão os 318 parâmetros convolucionais do RadImageNet
        if carregar_radimagenet:
            self._carregar_pesos_radimagenet(cache_dir=cache_dir)

        # 3. Substitui a camada 'fc' por uma cabeça de classificação customizada
        dim_entrada_fc = self.model.fc.in_features  # 2048
        if dropout_rate > 0.0:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(dim_entrada_fc, num_classes),
            )
        else:
            self.model.fc = nn.Linear(dim_entrada_fc, num_classes)

        # Inicialização He/Kaiming para a nova camada de classificação
        camada_linear = self.model.fc[-1] if isinstance(self.model.fc, nn.Sequential) else self.model.fc
        nn.init.kaiming_normal_(camada_linear.weight, mode="fan_out", nonlinearity="relu")
        if camada_linear.bias is not None:
            nn.init.constant_(camada_linear.bias, 0.0)

    def _carregar_pesos_radimagenet(
        self,
        repo_id: str = "Lab-Rasool/RadImageNet",
        filename: str = "ResNet50.pt",
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Baixa o checkpoint oficial do RadImageNet via Hugging Face Hub e mapeia
        as chaves de blocos sequenciais ('backbone.X.') diretamente para os nomes de camadas
        nativos do torchvision ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.').
        """
        print(f"[INFO] Obtendo pesos RadImageNet ({repo_id}/{filename})...")

        # Como o repositório é 100% público, token=False evita avisos de tokens de login expirados na máquina local
        caminho_pesos = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=False,
        )

        checkpoint = torch.load(caminho_pesos, map_location=self._device, weights_only=False)

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
            raise TypeError(f"Formato de checkpoint inválido: {type(checkpoint)}")

        # Mapeamento exato das camadas do RadImageNet para a ResNet-50 torchvision
        mapeamento_radimagenet = {
            "backbone.0.": "conv1.",
            "backbone.1.": "bn1.",
            "backbone.4.": "layer1.",
            "backbone.5.": "layer2.",
            "backbone.6.": "layer3.",
            "backbone.7.": "layer4.",
        }

        state_dict_limpo: Dict[str, torch.Tensor] = {}
        for chave, tensor in state_dict.items():
            chave_limpa = chave

            if chave_limpa.startswith("module."):
                chave_limpa = chave_limpa[len("module."):]

            # Ignora camada de classificação do pré-treino
            if chave_limpa.startswith("fc.") or "fc." in chave_limpa:
                continue

            mapeada = False
            for prefixo, destino in mapeamento_radimagenet.items():
                if chave_limpa.startswith(prefixo):
                    chave_limpa = destino + chave_limpa[len(prefixo):]
                    mapeada = True
                    break

            if not mapeada and chave_limpa.startswith("backbone."):
                chave_limpa = chave_limpa[len("backbone."):]

            state_dict_limpo[chave_limpa] = tensor

        mensagem = self.model.load_state_dict(state_dict_limpo, strict=False)
        total_carregadas = len(state_dict_limpo)
        print(
            f"[SUCESSO] {total_carregadas} tensores do RadImageNet carregados com sucesso! "
            f"(Chaves não carregadas esperadas: {mensagem.missing_keys})"
        )

    def freeze_backbone(self) -> None:
        """
        Congela todos os pesos convolucionais do backbone, mantendo apenas a cabeça
        de classificação (`fc`) com gradientes ativos (`requires_grad = True`).
        """
        for nome, parametro in self.model.named_parameters():
            if not nome.startswith("fc"):
                parametro.requires_grad = False
            else:
                parametro.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """
        Descongela todos os pesos da rede inteira para fine-tuning fim-a-fim.
        """
        for parametro in self.model.parameters():
            parametro.requires_grad = True

    def unfreeze_layers(self, layers: List[str]) -> None:
        """
        Descongela seletivamente apenas as camadas especificadas (ex: ['layer4', 'fc']).

        Parâmetros:
            layers (list): Lista com prefixos das camadas a serem descongeladas.
        """
        for nome, parametro in self.model.named_parameters():
            if any(nome.startswith(camada) for camada in layers):
                parametro.requires_grad = True
            else:
                parametro.requires_grad = False

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Union[List[torch.nn.Parameter], float]]]:
        """
        Retorna grupos de parâmetros para otimizadores (AdamW/SGD) com taxas de aprendizado
        diferenciais (transfer learning ideal: taxa menor no backbone e maior na cabeça nova).

        Parâmetros:
            backbone_lr (float): Taxa de aprendizado para os blocos convolucionais pré-treinados.
            head_lr (float): Taxa de aprendizado para a camada linear de classificação.
            weight_decay (float): Decaimento de peso L2.

        Retorna:
            list: Lista de dicionários no padrão exigido pelo PyTorch Optimizer.
        """
        backbone_params = []
        head_params = []

        for nome, parametro in self.model.named_parameters():
            if not parametro.requires_grad:
                continue
            if nome.startswith("fc"):
                head_params.append(parametro)
            else:
                backbone_params.append(parametro)

        grupos = []
        if backbone_params:
            grupos.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            })
        if head_params:
            grupos.append({
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            })
        return grupos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa a inferência direta através da rede.

        Parâmetros:
            x (torch.Tensor): Imagens de entrada no formato (Batch, 3, 224, 224).

        Retorna:
            torch.Tensor: Logits das classes no formato (Batch, num_classes).
        """
        return self.model(x)


if __name__ == "__main__":
    print("=" * 70)
    print("Verificação do ResNet50Classifier (RadImageNet)")
    print("=" * 70)
    modelo = ResNet50Classifier(num_classes=4, carregar_radimagenet=True)
    x = torch.randn(4, 3, 224, 224)
    saida = modelo(x)
    print(f"Entrada: {x.shape} -> Saída (Logits): {saida.shape}")
    print("=" * 70)
