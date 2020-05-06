import re

from torch.utils import model_zoo
from torchvision.models.densenet import DenseNet, model_urls as densenet_urls
from mtdp.components import FeaturesInterface
from mtdp.models._util import load_dox_url, clean_state_dict

MTDP_URLS = {
    "densenet121": ("https://dox.uliege.be/index.php/s/G72InP4xmJvOrVp/download", "densenet121-mh-best-191205-141200.pth")
}


class NoHeadDenseNet(DenseNet, FeaturesInterface):
    def forward(self, x):
        return self.features(x)

    def n_features(self):
        return self.features[-1].num_features


def build_densenet(pretrained=False, arch="densenet201", model_class=NoHeadDenseNet, **kwargs):
    r"""Densenet-XXX model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        arch (str): Type of densenet (among: densenet121, densenet169, densenet201 and densenet161)

        pretrained (str|None): If "imagenet", returns a model pre-trained on ImageNet. If "mtdp" returns a model pre-trained
                           in multi-task on digital pathology data. Otherwise (None), random weights.
        model_class (nn.Module): Actual densenet module class
    """
    params = {
        "densenet121": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 24, 16)},
        "densenet169": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 32, 32)},
        "densenet201": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 48, 32)},
        "densenet161": {"num_init_features": 96, "growth_rate": 48, "block_config": (6, 12, 36, 24)}
    }
    model = model_class(**(params[arch]), **kwargs)
    if isinstance(pretrained, str):
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        if pretrained == "imagenet":
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            state_dict = model_zoo.load_url(densenet_urls[arch])
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        elif pretrained == "mtdp":
            if arch not in MTDP_URLS:
                raise ValueError("No pretrained weights for multi task pretraining with architecture '{}'".format(arch))
            url, filename = MTDP_URLS[arch]
            state_dict = load_dox_url(url, filename, map_location="cpu")
            state_dict = clean_state_dict(state_dict, prefix="features.", filter=lambda k: not k.startswith("heads."))
        else:
            raise ValueError("Unknown pre-training source")
        model.load_state_dict(state_dict)
    return model
