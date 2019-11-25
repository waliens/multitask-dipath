from torch import nn

from mtdipath.components import Head


class SingleHead(nn.Module):
    """A single task network, similar to usual classification architectures. A network produces
    feature maps that are then process by a head composed of global average pooling and a linear
    layer producing the logits.
    """

    def __init__(self, features, head):
        """
        Parameters
        ----------
        features: nn.Module, FeaturesInterface
            The module producing the features map
        head: nn.Module
            The head module
        """
        super().__init__()
        self.features = features
        self.head = head

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class MultiHead(nn.Module):
    """A multi-task network composed of a shared network producing features and several heads,
    one per task, producing the task-specific predictions. All samples of a given task are
    routed only through this task's specific head.
    """

    def __init__(self, dataset, features):
        """
        Parameters
        ----------
        dataset: MultiTaskDataset
            The dataset for which the multi-head network must be built.
        features: nn.Module, FeaturesInterface
            The shared network module. Should have a `n_features()` function returning
            the number of feature maps it produces.
        """
        super().__init__()
        self._dataset = dataset
        self.heads = nn.ModuleDict({
            name: Head(n_features=features.n_features(), n_classes=n_classes)
            for name, n_classes in dataset.n_classes_per_dataset.items()
        })
        self._name_to_index = self._dataset.name_to_index

    def forward(self, x, sources):
        """
        Parameters
        ----------
        x: torch.Tensor
            Batch of images.
        sources: torch.Tensor
            A vector (same size as the batch) where `sources[i]` is the source index
            for sample `x[i]` of the dataset. Source index should be a unique identifier
            consistent with indexes defined by the dataset.

        Returns
        -------
        results: dict
            A dictionary mapping task name with another dictionary. `results[task_name]["logits"]` contains
            the logits for the all the samples of the task `task_name` contained in `x` (same order as the
            order they appear in the batch). `results[task_name]["which"]` is a binary mask indicating which
            samples of the batch actually belonged to task `task_name`

        """
        f = self.features(x)  # extract features for all inputs
        results = {}
        for name, head in self.heads.items():
            which = sources == self._name_to_index[name]
            if which.nonzero().size(0) > 0:
                results[name] = {
                    "logits": head(f[which]),
                    "which": which
                }
        return results

    def get_single_head(self, task_name):
        """Creates a single-head network by assembling the shared network and the given
        task's head.

        Parameters
        ----------
        task_name: str
            Name of the dataset for which the single-head network should be extracted.

        Returns
        -------
        singlehead: SingleHead
        """
        return SingleHead(self.features, self.heads[task_name])

    @property
    def dataset(self):
        return self._dataset