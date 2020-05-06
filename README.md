# mtdp

Library containing implementation related to the [research paper](http://hdl.handle.net/2268/247134) "_Multi-task pre-training of deep neural networks for digital pathology_" (Mormont _et al._).

It can be used to load our pre-trained models or to build a multi-task classification architecture.

## Loading our pre-trained weights.

> For an example, check the file [`examples/feature_extract.py`](https://github.com/waliens/multitask-dipath/blob/master/examples/feature_extract.py).

The library provides a `build_model` function to build a model and initialize it with our
pre-trained weights. To load our weights, the parameter `pretrained` should be set to `mtdp`.

```python
from mtdp import build_model

model = build_model(arch="densenet121", pretrained="mtdp")
```

Alternatively, `pretrained` can be set to `imagenet` to load ImageNet pre-trained weights from PyTorch.

We currently provide pre-trained weights for the following architectures:

- `densenet121`
- `resnet50`


See an example script performing feature extraction using one of our model in the `examples` folder (file `feature_extract.py`).

### Raw model files

If you want to bypass the library and download the raw PyTorch model files, you can access them at the following URLs:

- `densenet121`: [https://dox.uliege.be/index.php/s/G72InP4xmJvOrVp/download](https://dox.uliege.be/index.php/s/G72InP4xmJvOrVp/download)
- `resnet50`: [https://dox.uliege.be/index.php/s/kvABLtVuMxW8iJy/download](https://dox.uliege.be/index.php/s/kvABLtVuMxW8iJy/download)


## Building a multi-task architecture

> For an example, see the [`examples/multi_task_train.py`](https://github.com/waliens/multitask-dipath/blob/master/examples/multi_task_train.py) file.

Several steps for building the architecture:

1. define a `DatasetFolder`/`ImageFolder` for each of your individual dataset,
2. instantiate a `MultiImageFolders` object with all your dataset objects,
3. instantiate a `MultiHead` PyTorch module by passing it the `MultiImageFolders` from step 2. The
module will use the information of the tasks in order to build the multi-task architecture.



