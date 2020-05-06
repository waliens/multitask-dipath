from torch.utils import model_zoo
from torchvision.models.resnet import ResNet, model_urls as resnet_urls, BasicBlock, Bottleneck
from mtdp.components import FeaturesInterface
from mtdp.models._util import load_dox_url, clean_state_dict

MTDP_URLS = {
    "resnet50": ("https://dox.uliege.be/index.php/s/kvABLtVuMxW8iJy/download", "resnet50-mh-best-191205-141200.pth")
}


class NoHeadResNet(ResNet, FeaturesInterface):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def build_resnet(pretrained=None, arch="resnet50", model_class=NoHeadResNet, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        arch (str): Type of densenet (among: resnet18, resnet34, resnet50, resnet101 and resnet152)
        pretrained (str|None): If "imagenet", returns a model pre-trained on ImageNet. If "mtdp" returns a model
                              pre-trained in multi-task on digital pathology data. Otherwise (None), random weights.
        model_class (nn.Module): Actual resnet module class
    """
    params = {
        "resnet18": [BasicBlock, [2, 2, 2, 2]],
        "resnet34": [BasicBlock, [3, 4, 6, 3]],
        "resnet50": [Bottleneck, [3, 4, 6, 3]],
        "resnet101": [Bottleneck, [3, 4, 23, 3]],
        "resnet152":  [Bottleneck, [3, 8, 36, 3]]
    }
    model = model_class(*params[arch], **kwargs)
    if isinstance(pretrained, str):
        if pretrained == "imagenet":
            url = resnet_urls[arch]  # default imagenet
            state_dict = model_zoo.load_url(url)
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