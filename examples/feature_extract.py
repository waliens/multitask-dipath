import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from mtdp import build_model


if __name__ == "__main__":
    """Loading a model and using it as feature extractor
    """
    device = torch.device("cpu")
    model = build_model(arch="densenet121", pretrained="mtdp", pool=True)
    model.to(device)

    input_size = 244
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    train_dataset = ImageFolder("TRAIN_FOLDER", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16)
    test_dataset = ImageFolder("TEST_FOLDER", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16)

    with torch.no_grad():
        model.eval()
        # train
        features = list()
        classes = list()
        for i, (x, y) in enumerate(train_loader):
            print("> train iter #{}".format(i + 1))
            out = model.forward(x.to(device))
            features.append(out.detach().cpu().numpy().squeeze())
            classes.append(y.cpu().numpy())

        features = np.vstack(features)
        classes = np.hstack(classes)

        print("Train svm.")
        svm = LinearSVC(C=0.01)
        svm.fit(features, classes)

        # predict
        preds = list()
        y_test = list()
        for i, (x_test, y) in enumerate(test_loader):
            print("> test iter #{}".format(i + 1))
            out = model.forward(x_test.to(device))
            preds.append(svm.predict(out.detach().cpu().numpy().squeeze()))
            y_test.append(y.cpu().numpy())

        preds = np.hstack(preds)
        y_test = np.hstack(y_test)

        print("test accuracy:", accuracy_score(y_test, preds))

