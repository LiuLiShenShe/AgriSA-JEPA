"""Frozen-backbone linear classification probe over mean-pooled patch tokens.

The head is one ``Linear(1280 -> C)`` applied to the mean of the encoder's 256
patch tokens (``[N, 256, 1280] -> [B, 1280] -> [B, C]``).  No CLS token is
assumed (I-JEPA ViT-H emits patch tokens only), and no cropping beyond a
224x224 resize-aspect-preserve: the image is resized such that the short side
is 224 and a center 224x224 crop is taken, replicating the ImageNet-style
geometry the encoder was pretrained with.

Exactly one head exists, shared by all backbones; a smaller head is a smaller
free variable, not a different architecture.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.downstream.features import EMBED_DIM, TOKENS_PER_TILE


class LinearClassificationProbe(nn.Module):
    """``Linear(embed_dim -> num_classes)`` over mean-pooled patch tokens."""

    def __init__(self, embed_dim: int = EMBED_DIM, num_classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != TOKENS_PER_TILE:
            raise ValueError(
                f'Expected [N, {TOKENS_PER_TILE}, D] tokens, got {tuple(tokens.shape)}')
        pooled = tokens.mean(dim=1)                       # [N, D]
        return self.head(pooled)                          # [N, C]

    @torch.no_grad()
    def predict(self, tokens: torch.Tensor) -> np.ndarray:
        return self.forward(tokens).argmax(dim=1).cpu().numpy()


def train_classifier(
    probe: LinearClassificationProbe,
    train_tokens: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: torch.device = torch.device('cpu'),
    verbose: bool = False,
) -> Dict[str, object]:
    """Train the classifier head on cached mean-pool-friendly token features.

    ``train_tokens`` is ``[N, 256, 1280]`` fp16 (cast per batch);
    ``train_labels`` is ``[N]`` int64.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe.to(device)
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    n = train_tokens.shape[0]
    generator = torch.Generator().manual_seed(seed)
    history: list = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        total_loss, total_n = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            tokens = train_tokens[idx].to(device).float()
            labels = train_labels[idx].to(device)
            logits = probe(tokens)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            total_n += len(idx)
        history.append(total_loss / max(total_n, 1))
        if verbose:
            print(f'  epoch {epoch + 1}/{epochs} loss {history[-1]:.4f}', flush=True)
    probe.eval()
    return {'loss_history': history, 'final_loss': history[-1] if history else None}


@torch.no_grad()
def evaluate_classifier(
    probe: LinearClassificationProbe,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, object]:
    """Accuracy and per-class recall / macro-F1 on cached features."""
    n = tokens.shape[0]
    preds: list = []
    for start in range(0, n, batch_size):
        batch = tokens[start:start + batch_size].to(device).float()
        preds.append(probe(batch).argmax(dim=1).cpu())
    pred = torch.cat(preds).numpy()
    true = labels.numpy()
    num_classes = int(torch.max(torch.cat([labels, torch.tensor(pred)])).item()) + 1
    tp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for c in range(num_classes):
        tp[c] = int(((pred == c) & (true == c)).sum())
        fn[c] = int(((true == c) & (pred != c)).sum())
    recall = {c: (tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0)
              for c in range(num_classes)}
    # precision from confusion
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p_, t_ in zip(pred, true):
        cm[t_, p_] += 1
    precision = {c: (cm[c, c] / cm[:, c].sum() if cm[:, c].sum() else 0.0)
                 for c in range(num_classes)}
    f1 = {c: (2 * precision[c] * recall[c] / (precision[c] + recall[c])
              if precision[c] + recall[c] else 0.0) for c in range(num_classes)}
    acc = float((pred == true).mean())
    macro_f1 = float(np.mean(list(f1.values())))
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'per_class_f1': f1,
        'per_class_recall': recall,
        'per_class_precision': precision,
        'n_images': n,
    }