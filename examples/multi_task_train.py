import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from mtdp import build_model
from mtdp.helpers import compute_loss, rescale_head_grads
from mtdp.loader import MultiImageFolders
from mtdp.networks import MultiHead


if __name__ == "__main__":
    LR = 1e-4
    BATCH_SIZE = 8
    DEVICE = "cpu"
    INPUT_SIZE = 224

    """
    All your tasks should be provided as Dataset (e.g. ImageFolder, or custom implementation) to the 
    `MultiImageFolders` class which will provide an unified interface to them (as if they were a single 
    Dataset). Dataset root folder name should be unique.
    """
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    paths = ["path_to_dataset1", "path_to_dataset2"]
    dataset = MultiImageFolders([ImageFolder(path, transform) for path in paths])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    """
    Build the backbone (shared) network. Pooling will be added automatically later, so it is disabled.
    """
    backbone = build_model(arch="densenet121", pretrained="imagenet", pool=False)

    """
    The `MultiHead` class will build a multi-task network based on the passed `MultiImageFolders` object and the
    backbone network.
    """
    multihead = MultiHead(dataset, backbone)
    device = torch.device(DEVICE)
    multihead.to(device)

    # Training
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
    optimizer = torch.optim.SGD(multihead.parameters(), lr=LR)
    multihead.train()
    for i, (x, y, sources) in enumerate(loader):
        x = x.to(device)
        loss = compute_loss(multihead, x, y, sources, loss_fn)
        optimizer.zero_grad()
        loss.backward()
        rescale_head_grads(multihead, sources)
        optimizer.step()
        print("> train iter #{}: {}".format(i, loss.detach().cpu()))