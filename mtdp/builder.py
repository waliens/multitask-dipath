from mtdp.components import PooledFeatureExtractor
from mtdp.models.densenet import build_densenet
from mtdp.models.resnet import build_resnet


def build_model(arch, pool=False, **kwargs):
    """Get a network by architecture.

    Parameters
    ----------
    arch: str
        Architecture name. Supported architectures:
            `{'densenet121', 'densenet169', 'densenet201', 'densenet161',
              'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}`
    pool: bool
        True for adding a global pooling layer to the model.
    kwargs: dict
    """
    if "densenet" in arch:
        model = build_densenet(arch=arch, **kwargs)
    elif "resnet" in arch:
        model = build_resnet(arch=arch, **kwargs)
    else:
        raise ValueError("Unknown architecture")

    if pool:
        return PooledFeatureExtractor(model)
    else:
        return model