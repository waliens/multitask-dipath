from mtdp.models.densenet import build_densenet
from mtdp.models.resnet import build_resnet


def build_model(arch, **kwargs):
    """Get a network by architecture.

    Parameters
    ----------
    arch: str
        Architecture name. Supported architectures:
            `{'densenet121', 'densenet169', 'densenet201', 'densenet161',
              'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}`
    kwargs: dict
    """
    if "densenet" in arch:
        return build_densenet(arch=arch, **kwargs)
    elif "resnet" in arch:
        return build_resnet(arch=arch, **kwargs)
    else:
        raise ValueError("Unknown architecture")