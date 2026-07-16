"""
lora.py — Low-Rank Adaptation (LoRA) for UNETR++'s Efficient Paired
Attention (EPA) blocks.
"""
import math
import torch
import torch.nn as nn

from peft_methods.freezing import freeze_model


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a parallel trainable low-rank branch.
    y = base(x) + scaling * (x @ A^T @ B^T),   scaling = alpha / r
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_features, out_features = base_linear.in_features, base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # matches the original LoRA paper's init

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge(self):
        """Fold the LoRA branch into the base weight, permanently. After this,
        self.base is a plain nn.Linear with the adaptation baked in — no LoRA
        branch left to compute. Use for zero-added-latency inference/deployment."""
        with torch.no_grad():
            self.base.weight += (self.lora_B @ self.lora_A) * self.scaling
        return self.base


def apply_lora_to_model(model: nn.Module, target_attr="qkvv",
                         r: int = 8, alpha: int = 16, dropout: float = 0.0):
    """Walks the model and replaces every module's target_attr(s) with a
    LoRA-wrapped version, in place. target_attr may be a single name
    ("qkvv") or a list of names (["qkvv", "out_proj", "out_proj2"]) to
    adapt several projections at once.

    NOTE: do NOT include "E" and "F" together in target_attr — EPA sets
    self.E = self.F (literally the same shared weight object, by design).
    Targeting both independently would silently break that intentional
    sharing, turning one shared projection into two separately-adapted
    copies. Confirmed via `epa.E is epa.F -> True`.

    Returns the list of created LoRALinear modules — keep this reference if
    you want to call .merge() on all of them later for inference.
    """
    attrs = [target_attr] if isinstance(target_attr, str) else list(target_attr)
    lora_layers = []
    for module in model.modules():
        for attr in attrs:
            if hasattr(module, attr):
                target = getattr(module, attr)
                if isinstance(target, nn.Linear):
                    wrapped = LoRALinear(target, r=r, alpha=alpha, dropout=dropout)
                    setattr(module, attr, wrapped)
                    lora_layers.append(wrapped)
    return lora_layers


def freeze_for_lora(model: nn.Module, trainable_prefixes=("out1", "out2", "out3")):
    """Freeze everything except LoRA params and the given trainable prefixes(the output heads)"""
    return freeze_model(
        model,
        is_trainable_fn=lambda name: ("lora_" in name) or any(name.startswith(p) for p in trainable_prefixes)
    )


def merge_all(lora_layers):
    """Call .merge() on every LoRA layer created by apply_lora_to_model —
    used when training is done, before saving an inference-ready checkpoint."""
    for layer in lora_layers:
        layer.merge()


def save_lora_state(model, path, trainable_prefixes=("out1", "out2", "out3")):
    """Save ONLY the LoRA + trainable-head parameters"""
    state = {k: v for k, v in model.state_dict().items()
              if "lora_" in k or k.startswith(trainable_prefixes)}
    torch.save(state, path)