
def unfreeze_mh(module):
    """Unfreezes the shared network of a multi-head module. Its trainable parameters
    are made trainable (requires_grad=True) and the module is set to `train` mode.
    Parameters
    ----------
    multihead: MultiHead
        Multi-head network module
    """
    module.train()
    for param in module.parameters():
        param.requires_grad = True


def freeze_mh(module):
    """Freezes the shared network of a multi-head module. Its trainable parameters
    are fixed (requires_grad=False) and the module is set to `eval` mode.

    Parameters
    ----------
    multihead: MultiHead
        Multi-head network module
    """
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
