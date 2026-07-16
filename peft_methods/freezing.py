import torch

def freeze_model(model, is_trainable_fn):
    """Freeze all parameters except those for which is_trainable_fn(name) is
    True. Returns frozen_bn: the list of BatchNorm modules that ended up
    fully frozen, so their running stats stay locked at pretrained values.
    """
    for name, param in model.named_parameters():
        param.requires_grad = is_trainable_fn(name)

    frozen_bn = [
        m for m in model.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        and not any(p.requires_grad for p in m.parameters(recurse=False))
    ]
    for m in frozen_bn:
        m.eval()

    return frozen_bn


def set_train_mode(model, frozen_bn):
    model.train()
    for m in frozen_bn:
        m.eval()


def trainable_param_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100 * trainable / total