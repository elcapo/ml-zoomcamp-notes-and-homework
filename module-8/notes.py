import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from torchvision import models, transforms
    from PIL import Image
    return Image, mo, models, pd, torch, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Machine Learning Zoomcamp

    ## Module 8: **Deep Learning**
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    repository_root = (
        "https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/"
    )

    chapters = pd.DataFrame(
        [
            {
                "title": "Fashion Classification",
                "youtube_id": "it1Lu7NmMpw",
                "contents": repository_root + "08-deep-learning/01-fashion-classification.md",
            },
            {
                "title": "TensorFlow and Keras",
                "youtube_id": "R6o_CUmoN9Q",
                "contents": repository_root + "08-deep-learning/02-tensorflow-keras.md",
            },
            {
                "title": "Pre-trained Convolutional Neural Networks",
                "youtube_id": "qGDXEz-cr6M",
                "contents": repository_root + "08-deep-learning/03-pretrained-models.md",
            },
            {
                "title": "Convolutional Neural Networks",
                "youtube_id": "BN-fnYzbdc8",
                "contents": repository_root + "08-deep-learning/04-conv-neural-nets.md",
            },
            {
                "title": "Transfer Learning",
                "youtube_id": "WKHylqfNmq4",
                "contents": repository_root + "08-deep-learning/05-transfer-learning.md",
            },
            {
                "title": "Adjusing the Learning Rate",
                "youtube_id": "2gPmRRGz0Hc",
                "contents": repository_root + "08-deep-learning/06-learning-rate.md",
            },
            {
                "title": "Checkpointing",
                "youtube_id": "NRpGUx0o3Ps",
                "contents": repository_root + "08-deep-learning/07-checkpointing.md",
            },
            {
                "title": "Adding More Layers",
                "youtube_id": "bSRRrorvAZs",
                "contents": repository_root + "08-deep-learning/08-more-layers.md",
            },
            {
                "title": "Regularization and Dropout",
                "youtube_id": "74YmhVM6FTM",
                "contents": repository_root + "08-deep-learning/09-dropout.md",
            },
            {
                "title": "Data Augmentation",
                "youtube_id": "aoPfVsS3BDE",
                "contents": repository_root + "08-deep-learning/10-augmentation.md",
            },
            {
                "title": "Training a Larger Model",
                "youtube_id": "_QpDGJwFjYA",
                "contents": repository_root + "08-deep-learning/11-large-model.md",
            },
            {
                "title": "Using the Model",
                "youtube_id": "cM1WHKae1wo",
                "contents": repository_root + "08-deep-learning/12-using-model.md",
            },
            {
                "title": "Pytorch",
                "youtube_id": "Ne25VujHRLA",
                "contents": repository_root + "08-deep-learning/pytorch/README.md"
            },
            {
                "title": "Summary",
                "youtube_id": "mn0BcXJlRFM",
                "contents": repository_root + "08-deep-learning/13-summary.md",
            },
            {
                "title": "Explore More",
                "contents": repository_root + "08-deep-learning/14-explore-more.md",
            },
        ]
    )

    chapters.insert(
        loc=0,
        column="snapshot",
        value="https://img.youtube.com/vi/"
        + chapters.youtube_id.astype(str)
        + "/hqdefault.jpg",
    )
    chapters.insert(
        loc=2,
        column="youtube",
        value="https://youtube.com/watch?v=" + chapters.youtube_id.astype(str),
    )

    videos = chapters[chapters["youtube_id"].notnull()]
    videos[["snapshot", "title", "youtube"]]
    return (chapters,)


@app.cell(hide_code=True)
def _(chapters):
    contents = chapters[chapters["contents"].notnull()]
    contents[["title", "contents"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fashion Classification

    During this module we'll be working with a dataset that consists on a set of photos of clothes, and our goal will be to train a model that's capable of reading new images and determine based on the parameters that it learnt from those images, what type of clothing the new images are.

    ### Dataset

    > In this session, we'll be working on multiclass image classification with deep learning. Some deep learning frameworks like TensorFlow and Keras will be implemented on clothing dataset to classify images of clothes as t-shirts, hats, pants, etc.
    > 
    > The dataset has 5000 images of 20 different classes. However, we'll be using the subset which contains 10 of the most popular classes. The dataset can be downloaded from the above link.
    > 
    > Source: [01-fashion-classification.md](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/01-fashion-classification.md).

    * The dataset is available in Github: [alexeygrigorev/clothing-dataset-small](https://github.com/alexeygrigorev/clothing-dataset-small.git).
    * For convenience, the dataset has been cloned to the sibling [data](modules-8/data/) folder.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Pytorch

    * This module was initially recorded using Tensorflow but later rewriten using Pytorch. We'll follow here the updated [Pytorch](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/08-deep-learning/pytorch) version.
    * The module is pretty practique. For a deep dive into the theory of convolutional networks, check the [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu) Standford course.

    ### First Look at the Dataset
    """
    )
    return


@app.cell(hide_code=True)
def _(Image, mo):
    mo.hstack([
        Image.open("./module-8/data/train/dress/c968cba6-b6f6-4b59-820c-20535812609e.jpg"),
        Image.open("./module-8/data/train/shirt/bdaf259c-a7e1-4c45-a6ea-472ffe4116ad.jpg"),
        Image.open("./module-8/data/train/shoes/530c78c5-4fda-4abe-9f01-8d329524ab4f.jpg"),
        Image.open("./module-8/data/train/longsleeve/c73492a1-b5cd-40f0-9ca1-ae72635c544f.jpg"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Pretrained Models

    A quick way to get a image classifier working model is to start with a pretrained one. Here we use [MobileNetV2](https://docs.pytorch.org/vision/main/models/mobilenetv2.html) from **torchvision**.
    """
    )
    return


@app.cell
def _(models):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.eval()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Image Preprocess

    The [documentation of the model](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html#torchvision.models.mobilenet_v2) describes the preprocessing steps that are required for images to be understood by the model:

    The inference transforms are available at `MobileNet_V2_Weights.IMAGENET1K_V2.transforms` and perform the following preprocessing operations: Accepts `PIL.Image`, batched **(B, C, H, W)** and single **(C, H, W)** image `torch.Tensor` objects. The images are resized to `resize_size=[232]` using `interpolation=InterpolationMode.BILINEAR`, followed by a central crop of `crop_size=[224]`. Finally the values are first rescaled to `[0.0, 1.0]` and then normalized using `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`.
    """
    )
    return


@app.cell
def _(Image, transforms):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    single_image = Image.open("./module-8/data/train/shoes/530c78c5-4fda-4abe-9f01-8d329524ab4f.jpg")

    x = preprocess(single_image)
    x
    return single_image, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Predictions

    After preprocessing, we can use the model to make predictions. The output vector will contain logits corresponding to each of the 1,000 categories that the model can recognize.
    """
    )
    return


@app.cell
def _(model, torch, x):
    def single_predict(x: torch.tensor) -> torch.tensor:
        batch_t = torch.unsqueeze(x, 0)

        with torch.no_grad():
            output = model(batch_t)

        return output

    single_prediction = single_predict(x)

    {
        "prediction": single_prediction,
        "shape": single_prediction.shape,
    }
    return (single_prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can now get the list of identifiers of each of the classes ordering them so that the most likely classes are shown first:""")
    return


@app.cell
def _(mo, single_image, single_prediction, torch):
    _, indices = torch.sort(single_prediction, descending=True)

    with open("module-8/data/imagenet_classes.txt", "r") as f:
        imagenet_classes = f.read().split("\n")

    mo.hstack([
        [imagenet_classes[index] for index in indices[0]][: 5],
        single_image,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Transfer Learning

    With transfer learning we can take a pretrained model and adapt it to our specific needs.

    ### Create a Dataset Handler
    """
    )
    return


@app.cell
def _(Image):
    import os
    from torch.utils.data import Dataset

    class ClothingDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.classes = sorted(os.listdir(data_dir))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            for label_name in self.classes:
                label_dir = os.path.join(data_dir, label_name)
                for img_name in os.listdir(label_dir):
                    self.image_paths.append(os.path.join(label_dir, img_name))
                    self.labels.append(self.class_to_idx[label_name])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
    return ClothingDataset, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Define our Transformations

    We already showed the transformations that the model expects. We'll now create a function that prepares those transformations so that we can reuse them.
    """
    )
    return


@app.cell
def _(transforms):
    def get_transforms():
        input_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return (get_transforms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Define the Dataloaders""")
    return


@app.cell
def _(ClothingDataset, get_transforms):
    from torch.utils.data import DataLoader

    def get_dataloader(split: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        dataset = ClothingDataset(
            data_dir=f"./module-8/data/{split}",
            transform=get_transforms()
        )

        return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    train_loader = get_dataloader("train", batch_size=32, shuffle=True)
    val_loader = get_dataloader("validation", batch_size=32, shuffle=False)
    return train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create the Model""")
    return


@app.cell
def _(models, torch):
    import torch.nn as nn

    class ClothingClassifierMobileNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()

            # Add the original model (frozen)
            self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Remove original classifier
            self.base_model.classifier = nn.Identity()

            # Add custom layers
            self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.output_layer = nn.Linear(1280, num_classes)

        def forward(self, x):
            x = self.base_model.features(x)
            x = self.global_avg_pooling(x)
            x = torch.flatten(x, 1)
            x = self.output_layer(x)
            return x
    return ClothingClassifierMobileNet, nn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Train the model""")
    return


@app.cell
def _(nn, torch, train_loader, val_loader):
    import torch.optim as optim

    def get_device() -> str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        model: nn.Module,
        optimizer: optim.Adam,
        criterion: nn.CrossEntropyLoss,
        checkpoint_filename: str,
        num_epochs: int = 10,
    ):
        best_val_accuracy = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1} / {num_epochs}")

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(get_device()), labels.to(get_device())

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            val_acc = evaluate(model, criterion)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), checkpoint_filename)
                print(f"  Checkpoint saved: {checkpoint_filename}")

    def evaluate(
        model: nn.Module,
        criterion: nn.CrossEntropyLoss
    ) -> float:
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(get_device()), labels.to(get_device())

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return val_acc
    return evaluate, get_device, optim, train


@app.cell
def _(
    ClothingClassifierMobileNet,
    evaluate,
    get_device,
    nn,
    optim,
    torch,
    train,
):
    from pathlib import Path

    mobilenet_weights_filename = "./module-8/data/mobilenet-model.torch"

    def get_mobilenet_classifier(learning_rate: float = 0.01) -> tuple[nn.Module, optim.Adam, nn.CrossEntropyLoss]:
        device = get_device()

        model = ClothingClassifierMobileNet(num_classes=10)
        model.to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion

    def train_mobilenet_model():
        print("The model weights were not found, so the model will be trained")
        model, optimizer, criterion = get_mobilenet_classifier()
        model = train(model, optimizer, criterion, checkpoint_filename = mobilenet_weights_filename, num_epochs = 10)

    def evaluate_mobilenet_model():
        print("The model weights were found, so the model weights will be loaded and the model will be evaluated")
        model, optimizer, criterion = get_mobilenet_classifier()
        model.load_state_dict(torch.load(mobilenet_weights_filename, weights_only=True))
        evaluate(model, criterion)

    if not Path(mobilenet_weights_filename).exists():
        train_mobilenet_model()
    else:
        evaluate_mobilenet_model()
    return Path, mobilenet_weights_filename, train_mobilenet_model


@app.cell
def _(Path, mo, mobilenet_weights_filename, os, train_mobilenet_model):
    def delete_file(filename: str):
        if Path(filename).exists():
            os.remove(filename)

    def delete_mobilenet_weights():
        delete_file(mobilenet_weights_filename)
        train_mobilenet_model()

    mo.ui.button(
        label="Remove MobileNet weights and retrain",
        kind="danger",
        on_click=lambda v: delete_mobilenet_weights(),
    )
    return delete_file, delete_mobilenet_weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Improve the Model

    We'll not improve the model by adding an inner layer.
    """
    )
    return


@app.cell
def _(models, nn, torch):
    class UpgradedClothingClassifierMobileNet(nn.Module):
        def __init__(self, size_inner: int = 100, num_classes: int = 10):
            super().__init__()

            # Add the original model (frozen)
            self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Remove original classifier
            self.base_model.classifier = nn.Identity()

            # Add custom layers
            self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.inner = nn.Linear(1280, size_inner)
            self.relu = nn.ReLU()
            self.output_layer = nn.Linear(size_inner, num_classes)

        def forward(self, x):
            x = self.base_model.features(x)
            x = self.global_avg_pooling(x)
            x = torch.flatten(x, 1)
            x = self.inner(x)
            x = self.relu(x)
            x = self.output_layer(x)
            return x
    return (UpgradedClothingClassifierMobileNet,)


@app.cell
def _(
    Path,
    UpgradedClothingClassifierMobileNet,
    evaluate,
    get_device,
    nn,
    optim,
    torch,
    train,
):
    improved_mobilenet_weights_filename = "./module-8/data/improved_mobilenet-model.torch"

    def get_improved_mobilenet_classifier(learning_rate: float = 0.01) -> tuple[nn.Module, optim.Adam, nn.CrossEntropyLoss]:
        device = get_device()

        model = UpgradedClothingClassifierMobileNet(num_classes=10)
        model.to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion

    def train_improved_mobilenet_model():
        print("The model weights were not found, so the model will be trained")
        model, optimizer, criterion = get_improved_mobilenet_classifier()
        model = train(model, optimizer, criterion, checkpoint_filename = improved_mobilenet_weights_filename, num_epochs = 10)

    def evaluate_improved_mobilenet_model():
        print("The model weights were found, so the model weights will be loaded and the model will be evaluated")
        model, optimizer, criterion = get_improved_mobilenet_classifier()
        model.load_state_dict(torch.load(improved_mobilenet_weights_filename, weights_only=True))
        evaluate(model, criterion)

    if not Path(improved_mobilenet_weights_filename).exists():
        train_improved_mobilenet_model()
    else:
        evaluate_improved_mobilenet_model()
    return improved_mobilenet_weights_filename, train_improved_mobilenet_model


@app.cell
def _(
    delete_file,
    delete_mobilenet_weights,
    improved_mobilenet_weights_filename,
    mo,
    train_improved_mobilenet_model,
):
    def delete_improved_mobilenet_weights():
        delete_file(improved_mobilenet_weights_filename)
        train_improved_mobilenet_model()

    mo.ui.button(
        label="Remove the improved MobileNet weights and retrain",
        kind="danger",
        on_click=lambda v: delete_mobilenet_weights(),
    )
    return


if __name__ == "__main__":
    app.run()
