import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm


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


def get_batch_norm_layers(module, current_name):
    """Find and return a list of all batch norm modules in the given module"""
    bns = list()
    for i, (name, m) in enumerate(module.named_children()):
        iter_name = current_name + [name]
        if isinstance(m, _BatchNorm):
            m.bn_name = ".".join(iter_name)
            bns.append(m)
        else:
            bns.extend(get_batch_norm_layers(m, iter_name))
    return bns


def adapt_batch_norm(network, loader, device, n_iter=20, forward_params_fn=None):
    """
    In place update of the batch norm weights and bias for preparing a pre-trained network for a domain-switch.

    Parameters
    ----------
    forward_params_fn: callable
        A callable that is passed the batch and return the list of parameters to pass to the forward function of the
        'network' module.
    """
    if forward_params_fn is None:
        def default_forward_params(batch):
            return [batch[0].to(device)]
        forward_params_fn = default_forward_params

    bns = {bn.bn_name: bn for bn in get_batch_norm_layers(network, [])}

    if np.any([not bn.affine for bn in bns.values()]):
        warnings.warn("some layers have 'affine' disabled. They cannot be fixed by this approach, and will therefore be"
                      "ignored")
        bns = {name: bn for name, bn in bns.items() if bn.affine}

    means = defaultdict(list)
    vars = defaultdict(list)

    def hook_fn(bn, _in):
        size = _in[0].size()
        if len(size) != 4:  # support for NCHW only
            raise ValueError("Invalid shape {}".format(size))
        n_features = size[1]
        d_in = _in[0].detach().permute(1, 0, 2, 3).contiguous().view(n_features, -1)
        means[bn.bn_name].append(torch.mean(d_in, dim=1, keepdim=False).cpu().numpy())
        vars[bn.bn_name].append(torch.var(d_in, dim=1, keepdim=False, unbiased=True).cpu().numpy())

    hooks = [bn.register_forward_pre_hook(hook_fn) for bn in bns.values()]

    # forward samples into the network
    network.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_iter:
                break
            _ = network(*forward_params_fn(batch))

    def eps_std(var, eps):
        return torch.sqrt(var + eps)

    for name, bn in bns.items():
        mu_t = torch.tensor(np.mean(np.array(means[name]), axis=0))
        var_t = torch.tensor(np.mean(np.array(vars[name]), axis=0))
        mu_s = bn.running_mean.detach().cpu()
        var_s = bn.running_var.detach().cpu()
        gamma_s, beta_s = bn.weight.detach().cpu(), bn.bias.detach().cpu()

        gamma_t = gamma_s * eps_std(var_t, bn.eps) / eps_std(var_s, bn.eps)
        beta_t = beta_s + gamma_s * (mu_t - mu_s) / eps_std(var_s, bn.eps)

        # adapt/update old batch norm
        bn.weight = torch.nn.Parameter(gamma_t)
        bn.bias = torch.nn.Parameter(beta_t)
        bn.running_mean = mu_t
        bn.running_var = var_t
        bn.num_batches_tracked = torch.tensor(n_iter, dtype=torch.long)

    for hook in hooks:
        hook.remove()

    network.to(device)

    return network


def forward(multihead, x, sources):
    """Forward samples through the multihead network"""
    results = multihead.forward(x, sources)
    return {source_name: source_results for source_name, source_results in results.items()}


def compute_loss(multihead, x, y, sources, loss_fn, aggreg_fn=torch.mean, return_losses=False):
    """Forward samples into a multihead network and computes the loss
    Parameters
    ----------
    multihead: MultiHead
    x: torch.Tensor
    y: torch.Tensor
    sources: torch.Tensor
    loss_fn: callable
    aggreg_fn: callable
    return_losses: bool
    :return:
    """
    losses, losses_per_task = list(), dict()
    for source, results in forward(multihead, x, sources).items():
        source_losses = loss_fn(results["logits"], y[results["which"]])
        losses_per_task[source] = source_losses.detach().cpu().numpy()
        losses.append(source_losses)
    loss = aggreg_fn(torch.cat(losses))
    if return_losses:
        return loss, losses_per_task
    else:
        return loss


def rescale_head_grads(multihead, sources):
    """
    Rescale the heads gradients based on the number of samples that passed through the
    head during this iteration
    Parameters
    ----------
    multihead: MultiHead
        Multihead network
    sources: torch.tensor
        Batch sources strings.
    """
    sources = sources.numpy()
    batch_size = sources.shape[0]
    values, counts = np.unique(sources, return_counts=True)
    for index, count in zip(values, counts):
        head = multihead.heads[multihead.dataset.name(index)]
        for p in head.parameters():
            p.grad *= batch_size / count