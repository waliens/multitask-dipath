import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def datasets_size_cumsum(datasets):
    sizes = np.array([len(d) for d in datasets])
    cumsum = np.concatenate([np.array([0]), np.cumsum(sizes[:-1], dtype=np.int)])
    return sizes, cumsum


def get_sample_indexes(index, cumsum):
    dataset_index = np.searchsorted(cumsum, index, side="right") - 1
    relative_index = index - cumsum[dataset_index]
    return dataset_index, relative_index


def merge_dicts(dicts):
    out_dict = dict()
    for d in dicts:
        for k, v in d.items():
            if k in out_dict and v != out_dict[k]:
                raise ValueError("Value mismatch for key {} and value {} (!= {})".format(k, v, out_dict[k]))
            out_dict[k] = v
    return out_dict


def get_image_folders_if_not_empty(paths, **kwargs):
    folders = list()
    for path in paths:
        try:
            folders.append(ImageFolder(path, **kwargs))
        except RuntimeError as e:
            print("skip dataset at '{}' because: \"{}\"".format(path, str(e)))
            pass
    return folders


def add_group(dataset: ImageFolder, index, do_add_group=False):
    # this is what ImageFolder normally returns
    original_tuple = dataset[index]
    if not do_add_group:
        return original_tuple

    # the image file path
    path = dataset.imgs[index][0]
    # make a new tuple that includes original and the path
    group = os.path.basename(path).split("_", 1)[0]
    return original_tuple + (group,)


class MultiSetImageFolder(Dataset):
    """A classification dataset splitted in several sets structured as follows: {base_path}/{set_name}/{cls}/*
    Image filename can be prefixed with a group identifier which can optionally be returned.
    """

    def __init__(self, base_path, sets, do_add_group=False, **kwargs):
        """
        Parameters
        ----------
        base_path: str
            Base path of the dataset folder
        sets: list
            List of set folder names as strings.
        do_add_group: bool
            True to append group identifier (optional), default: `False`.
        kwargs: dict
            Parameters to be transferred to the actual `ImageFolder`.
        """
        super().__init__()
        self._datasets = get_image_folders_if_not_empty([os.path.join(base_path, _set) for _set in sets], **kwargs)
        self._sizes, self._cumsum_sizes = datasets_size_cumsum(self._datasets)
        self.class_to_idx = merge_dicts([d.class_to_idx for d in self._datasets])
        self.classes = list(self.class_to_idx.keys())
        self.do_add_group = do_add_group

    def __getitem__(self, index):
        dataset_index, relative_index = get_sample_indexes(index, self._cumsum_sizes)
        return add_group(self._datasets[dataset_index], relative_index, do_add_group=self.do_add_group)

    def __len__(self):
        return self._cumsum_sizes[-1] + len(self._datasets[-1])

    @property
    def n_classes(self):
        """Total number of classes in the dataset"""
        return len(self.classes)

    @property
    def root(self):
        """The name of the dataset folder"""
        return os.path.dirname(self._datasets[0].root)


class MultiImageFolders(Dataset):
    """Multiple tasks, each being represented by a Dataset. Each dataset can be identified by its name (name of
    the dataset folder, should be unique) or an index.
    """
    def __init__(self, datasets, indexes=None):
        """
        Parameters
        ----------
        datasets: iterable
            List of datasets, each of which represents a task.
        indexes: iterable
            List of indexes associated with every dataset (optional). Default: each dataset get its index
            in `datasets` as index.
        """
        if indexes is not None and len(indexes) != len(datasets):
            raise ValueError("indexes should have the same size as datasets")
        self._datasets = datasets
        # array of actual indexes, maps internal with external ids
        self._indexes = list(range(len(datasets))) if indexes is None else indexes
        # maps external id with internal id
        self._index_to_dataset = {i: d for i, d in zip(self._indexes, self._datasets)}
        # maps dataset name with external id
        self._dataset_name_to_index = {self.name(i): i for i in self._indexes}
        self._sizes, self._cumsum_sizes = datasets_size_cumsum(self._datasets)
        self._check_name_unicity()

    def _check_name_unicity(self):
        """Check whether all datasets have different names"""
        names = set()
        for i, dataset in enumerate(self.datasets):
            name = self.name(i)
            if name in names:
                raise ValueError("several datasets in the MultiImageFolders have the same name '{}' (folder name)".format(name))
            names.add(self.name(i))

    def __getitem__(self, index):
        dataset_index, relative_index = get_sample_indexes(index, self._cumsum_sizes)
        sample = self._datasets[dataset_index][relative_index]
        sample = sample + (self._indexes[dataset_index],)  # store the dataset index in the returned data
        return sample

    def __len__(self):
        return self._cumsum_sizes[-1] + len(self._datasets[-1])

    @property
    def datasets(self):
        """List of datasets"""
        return self._datasets

    @property
    def name_to_index(self):
        """Get the map for dataset name to dataset index"""
        return self._dataset_name_to_index

    def dataset_by_name(self, name):
        """Get the dataset object from the name
        Parameters
        ----------
        name: str
            Name of the dataset
        """
        return self._index_to_dataset[self.name_to_index[name]]

    @property
    def weights(self):
        """Return a weight vector for the samples so that each dataset has the same probability '1 / len(datasets)'
        of being sampled

        Returns
        -------
        weights: ndarray
            Dimensions (len(self),). Sample weights.
        """
        return np.repeat([1 / (len(self._datasets) * self._sizes)], self._sizes)

    @property
    def n_classes_per_dataset(self):
        """Return the number of classes for each dataset"""
        return {name: len(v) for name, v in self.classes_per_dataset.items()}

    @property
    def classes_per_dataset(self):
        """Return the classes for each dataset"""
        return {self.name(i): list(d.class_to_idx.keys()) for i, d in enumerate(self._datasets)}

    @property
    def class_to_idx_per_dataset(self):
        """Return class indexes for each dataset"""
        return {self.name(i): d.class_to_idx for i, d in enumerate(self._datasets)}

    @property
    def n_classes(self):
        return sum([len(a) for a in self.classes_per_dataset.values()])

    def report(self):
        print("Multi dataset with {} sub-dataset(s) and {} samples.".format(len(self._datasets), len(self)))
        for i, d in self._index_to_dataset.items():
            print("> {} ({}): {} samples, classes {}".format(self.name(i), i, len(d), d.classes))

    def name(self, index):
        return os.path.basename(self._index_to_dataset[index].root)

    @property
    def names(self):
        return [self.name(i) for i, _ in enumerate(self._datasets)]
