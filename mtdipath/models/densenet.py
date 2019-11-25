import re

from torch.utils import model_zoo
from torchvision.models.densenet import DenseNet, model_urls as densenet_urls
from mtdipath.components import FeaturesInterface


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
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        model_class (nn.Module): Actual densenet module class
    """
    params = {
        "densenet121": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 24, 16)},
        "densenet169": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 32, 32)},
        "densenet201": {"num_init_features": 64, "growth_rate": 32, "block_config": (6, 12, 48, 32)},
        "densenet161": {"num_init_features": 96, "growth_rate": 48, "block_config": (6, 12, 36, 24)}
    }
    model = model_class(**(params[arch]), **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(densenet_urls[arch])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model