"""
Módulo de modelos e arquiteturas de redes neurais do projeto.
"""

from .encoder import ResNet50Encoder
from .svm import ClassificadorAlzheimerSVM

__all__ = ["ResNet50Encoder", "ClassificadorAlzheimerSVM"]
