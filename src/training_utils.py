#!/usr/bin/env python3
"""
Shared training-loop helpers.

`train_epoch` was originally defined in the superseded whole-genome classifier
and is used unchanged by the hierarchical MoE trainer (train_moe.py) and,
through it, by the flat baseline (train_flat.py). It lives here so the training
pipeline stands on its own.
"""

import torch
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device, clamp_weights=True):
    """Train for one epoch

    clamp_weights: clip non-BatchNorm params to [-1.5, 1.5] after each step. This is
    tuned for binarized (straight-through-estimator) training; pass False for a
    standard float32 model, where it would otherwise be an uncontrolled extra
    regularizer confounding a binary-vs-float32 accuracy comparison.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Convert binary outputs to logits for cross-entropy
        if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
            loss = criterion(output, target)
        else:
            output_float = output.float()
            loss = criterion(output_float, target)

        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if clamp_weights:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    # keep BatchNorm params unclipped
                    if "bn" in n.lower() or "batchnorm" in n.lower():
                        continue
                    p.clamp_(-1.5, 1.5)

        
        running_loss += loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        # Update correct and total counts
        correct += (pred == target).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc
