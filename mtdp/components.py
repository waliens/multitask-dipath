from abc import abstractmethod
from torch import nn


class FeaturesInterface(object):
    @abstractmethod
    def n_features(self):
        pass


class Head(nn.Module):
    """A head is a simple neural network that can be used as a task-specific predictor
    in a multi-task network. It features a global average pooling followed by a linear
    layer (no activation).
    """

    def __init__(self, n_features, n_classes=2):
        """
        Parameters
        ----------
        n_features: int
            The number of input features after global average pooling
        n_classes: int
            The number of classes (i.e. output features)
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Conv2d(
            n_features,
            out_channels=n_classes,
            kernel_size=1
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.linear(x)
        return x.view(x.size(0), -1)


class PooledFeatureExtractor(nn.Module, FeaturesInterface):
    """This module applies a global average pooling on features produced by a module.
    """

    def __init__(self, features):
        """
        Parameters
        ----------
        features: nn.Module
            A network producing a set of feature maps. `features` should have a `n_features()` method
            returning how many features maps it produces.
        """
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.features(x))

    def n_features(self):
        return self.features.n_features()
