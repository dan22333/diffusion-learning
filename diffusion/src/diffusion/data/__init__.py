"""Datasets. Phase 0- overfits a fixed batch — synthetic (toy) or real (cifar);
the full CIFAR-10 learning run arrives in Phase 0.5."""
from .cifar import make_cifar_batch
from .toy import make_toy_batch

__all__ = ["make_toy_batch", "make_cifar_batch"]
