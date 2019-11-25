
def module_unfreeze(module):
    """Unfreezes the shared network of module. Its trainable parameters
    are made trainable (requires_grad=True) and the module is set to `train` mode.
    Parameters
    ----------
    module: MultiHead
        Multi-head network module
    """
    module.train()
    for param in module.parameters():
        param.requires_grad = True


def module_freeze(module):
    """Freezes the shared network of module. Its trainable parameters
    are fixed (requires_grad=False) and the module is set to `eval` mode.

    Parameters
    ----------
    module: MultiHead
        Multi-head network module
    """
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
