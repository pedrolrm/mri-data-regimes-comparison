"""
MRI Data Regimes Comparison Package.

Modular subpackages:
- `src.models`: Neural network backbones and classifier architectures.
- `src.features`: Latent representation extraction pipelines.
- `src.train`: Training routines for ResNet-50 and SVM across data regimes.
- `src.eval`: Standalone evaluation, inference, and metric calculation.
- `src.plots`: Publication-quality scientific figures generation.
"""

from src import eval, features, models, plots, train

__all__ = ["models", "features", "train", "eval", "plots"]
