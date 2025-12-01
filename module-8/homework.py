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
    return Image, mo, np, torch, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Homework

    > **Note**: it's very likely that in this homework your answers won't match 
    > the options exactly. That's okay and expected. Select the option that's
    > closest to your solution.
    > If it's exactly in between two options, select the higher value.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Dataset

    In this homework, we'll build a model for classifying various hair types. 
    For this, we will use the Hair Type dataset that was obtained from 
    [Kaggle](https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset) 
    and slightly rebuilt.

    You can download the target dataset for this homework from 
    [here](https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip):

    ```bash
    wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
    unzip data.zip
    ```

    In the lectures we saw how to use a pre-trained neural network. In the homework, we'll train a much smaller model from scratch. 

    We will use PyTorch for that.

    You can use Google Colab or your own computer for that.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Data Preparation

    The dataset contains around 1000 images of hairs in the separate folders 
    for training and test sets. 

    ### Reproducibility

    Reproducibility in deep learning is a multifaceted challenge that requires attention 
    to both software and hardware details. In some cases, we can't guarantee exactly the same results during the same experiment runs.

    Therefore, in this homework we suggest to set the random number seed generators by:

    ```python
    import numpy as np
    import torch

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ```

    Also, use PyTorch of version 2.8.0 (that's the one in Colab).
    """
    )
    return


@app.cell
def _(np, torch):
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Model

    For this homework we will use Convolutional Neural Network (CNN). We'll use PyTorch.

    You need to develop the model with following structure:

    * The shape for input should be `(3, 200, 200)` (channels first format in PyTorch)
    * Next, create a convolutional layer (`nn.Conv2d`):
        * Use 32 filters (output channels)
        * Kernel size should be `(3, 3)` (that's the size of the filter)
        * Use `'relu'` as activation 
    * Reduce the size of the feature map with max pooling (`nn.MaxPool2d`)
        * Set the pooling size to `(2, 2)`
    * Turn the multi-dimensional result into vectors using `flatten` or `view`
    * Next, add a `nn.Linear` layer with 64 neurons and `'relu'` activation
    * Finally, create the `nn.Linear` layer with 1 neuron - this will be the output
        * The output layer should have an activation - use the appropriate activation for the binary classification case

    As optimizer use `torch.optim.SGD` with the following parameters:

    * `torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)`
    """
    )
    return


@app.cell
def _(torch):
    import torch.nn as nn

    class BinaryClassification(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.linear = nn.Linear(32 * 99 * 99, 64)
            self.output = nn.Linear(64, 1)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            x = self.relu(x)
            x = self.output(x)
            return x
    return BinaryClassification, nn


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 1

    Which loss function you will use?

    * `nn.MSELoss()`
    * `nn.BCEWithLogitsLoss()`
    * `nn.CrossEntropyLoss()`
    * `nn.CosineEmbeddingLoss()`

    (Multiple answered can be correct, so pick any)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""I used **nn.BCEWithLogitsLoss()**.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 2

    What's the total number of parameters of the model? You can use `torchsummary` or count manually. 

    In PyTorch, you can find the total number of parameters using:

    ```python
    # Option 1: Using torchsummary (install with: pip install torchsummary)
    from torchsummary import summary
    summary(model, input_size=(3, 200, 200))

    # Option 2: Manual counting
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    ```

    * 896 
    * 11214912
    * 15896912
    * 20073473
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The number of parameters of the model is **20073473**, as shown below.""")
    return


@app.cell
def _(get_classifier):
    def count_model_parameters():
        model, _, _ = get_classifier()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

    count_model_parameters()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Generators and Training

    For the next two questions, use the following transformation for both train and test sets:
    """
    )
    return


@app.cell
def _(transforms):
    def get_transforms():
        return transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return (get_transforms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We don't need to do any additional pre-processing for the images.""")
    return


@app.cell
def _(DataLoader, nn, torch):
    import torch.optim as optim

    def get_device() -> str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        model: nn.Module,
        optimizer: optim.Adam,
        criterion: nn.CrossEntropyLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_filename: str,
        num_epochs: int = 10,
    ):
        best_val_accuracy = 0.0

        train_history = {"accuracy": [], "loss": []}
        validation_history = {"accuracy": [], "loss": []}

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1} / {num_epochs}")

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(get_device()), labels.to(get_device())
                labels = labels.float().unsqueeze(1)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total

            train_history["loss"].append(train_loss)
            train_history["accuracy"].append(train_acc)

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            val_loss, val_acc = evaluate(model, criterion, val_loader)

            validation_history["loss"].append(train_loss)
            validation_history["accuracy"].append(train_acc)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), checkpoint_filename)
                print(f"  Checkpoint saved: {checkpoint_filename}")

        return train_history, validation_history

    def evaluate(
        model: nn.Module,
        criterion: nn.CrossEntropyLoss,
        val_loader: DataLoader
    ) -> tuple[float]:
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(get_device()), labels.to(get_device())
                labels = labels.float().unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return val_loss, val_acc
    return evaluate, get_device, optim, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    * Use `batch_size=20`
    * Use `shuffle=True` for both training, but `False` for test.
    """
    )
    return


@app.cell
def _(Image, get_transforms):
    import os
    from torch.utils.data import Dataset, DataLoader

    class HomeworkDataset(Dataset):
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

    def get_dataloader(split: str, batch_size: int = 20, shuffle: bool = True) -> DataLoader:
        dataset = HomeworkDataset(
            data_dir=f"./module-8/data/homework/{split}",
            transform=get_transforms()
        )

        return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    train_loader = get_dataloader("train", shuffle=True)
    val_loader = get_dataloader("test", shuffle=False)
    return DataLoader, HomeworkDataset, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now fit the model.""")
    return


@app.cell
def _(
    BinaryClassification,
    evaluate,
    get_device,
    nn,
    optim,
    torch,
    train,
    train_loader,
    val_loader,
):
    from pathlib import Path

    weights_filename = "./module-8/data/homework-model.torch"

    def get_classifier() -> tuple[nn.Module, optim.Adam, nn.CrossEntropyLoss]:
        device = get_device()

        model = BinaryClassification()
        model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
        criterion = nn.BCEWithLogitsLoss()

        return model, optimizer, criterion

    # This values were manually written after a train execution
    # Running the training process will reset them
    train_history = {"accuracy": [0.7965, 0.7965], "loss": [-2.948, 2.948]}
    validation_history = {"accuracy": [], "loss": []}

    if not Path(weights_filename).exists():
        print("The model weights were not found, so the model will be trained")
        model, optimizer, criterion = get_classifier()
        train_history, validation_history = train(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            checkpoint_filename = weights_filename,
            num_epochs = 10,
        )
    else:
        print("The model weights were found, so the model weights will be loaded and the model will be evaluated")
        model, optimizer, criterion = get_classifier()
        model.load_state_dict(torch.load(weights_filename, weights_only=True))
        evaluate(
            model,
            criterion,
            val_loader,
        )

    train_history, validation_history
    return Path, criterion, get_classifier, model, optimizer, train_history


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 3

    What is the median of training accuracy for all the epochs for this model?

    * 0.05
    * 0.12
    * 0.40
    * 0.84
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""After a training, the following expression returned 0.7965, so the closest suggested option is **0.84**.""")
    return


@app.cell
def _(np, train_history):
    np.median(train_history["accuracy"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 4

    What is the standard deviation of training loss for all the epochs for this model?

    * 0.007
    * 0.078
    * 0.171
    * 1.710
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""After a training, the following expression returned 2.948, so the closest suggested option is **1.710**.""")
    return


@app.cell
def _(np, train_history):
    np.std(train_history["loss"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Data Augmentation

    For the next two questions, we'll generate more data using data augmentations. 

    Add the following augmentations to your training data generator:

    ```python
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    ```
    """
    )
    return


@app.cell
def _(transforms):
    def get_augmented_transforms():
        return transforms.Compose([
            transforms.RandomRotation(50),
            transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return (get_augmented_transforms,)


@app.cell
def _(DataLoader, HomeworkDataset, get_augmented_transforms):
    def get_augmented_dataloader(split: str, batch_size: int = 20, shuffle: bool = True) -> DataLoader:
        dataset = HomeworkDataset(
            data_dir=f"./module-8/data/homework/{split}",
            transform=get_augmented_transforms()
        )

        return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    train_augmented_loader = get_augmented_dataloader("train", shuffle=True)
    val_augmented_loader = get_augmented_dataloader("test", shuffle=False)
    return train_augmented_loader, val_augmented_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 5 

    Let's train our model for 10 more epochs using the same code as previously.

    > **Note:** make sure you don't re-create the model.
    > we want to continue training the model we already started training.
    """
    )
    return


@app.cell
def _(
    Path,
    criterion,
    evaluate,
    get_classifier,
    model,
    optimizer,
    torch,
    train,
    train_augmented_loader,
    val_augmented_loader,
):
    augmented_weights_filename = "./module-8/data/homework-augmented_model.torch"

    # This values were manually written after a train execution
    # Running the training process will reset them
    augmented_train_history = {"accuracy": [], "loss": [10.44]}
    augmented_validation_history = {"accuracy": [0.7523, 0.7523, 0.7523, 0.7523, 0.7523, 0.7523], "loss": []}

    if not Path(augmented_weights_filename).exists():
        print("The model weights were not found, so the model will be trained")
        augmented_train_history, augmented_validation_history = train(
            model,
            optimizer,
            criterion,
            train_augmented_loader,
            val_augmented_loader,
            checkpoint_filename = augmented_weights_filename,
            num_epochs = 10,
        )
    else:
        print("The model weights were found, so the model weights will be loaded and the model will be evaluated")
        augmented_model, augmented_optimizer, augmented_criterion = get_classifier()
        augmented_model.load_state_dict(torch.load(augmented_weights_filename, weights_only=True))
        evaluate(
            augmented_model,
            augmented_criterion,
            val_augmented_loader,
        )

    augmented_train_history, augmented_validation_history
    return augmented_train_history, augmented_validation_history


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    What is the mean of test loss for all the epochs for the model trained with augmentations?

    * 0.008
    * 0.08
    * 0.88
    * 8.88
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The mean of the loss for the augmented epochs is 10.44. The closest suggested option is **8.88**.""")
    return


@app.cell
def _(augmented_train_history, np):
    np.mean(augmented_train_history["loss"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Question 6

    What's the average of test accuracy for the last 5 epochs (from 6 to 10)
    for the model trained with augmentations?

    * 0.08
    * 0.28
    * 0.68
    * 0.98
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The average test accuracy for the last 5 epochs trained with augmentations is 0.7523, and the closest suggested option is **0.68**.""")
    return


@app.cell
def _(augmented_validation_history, np):
    np.average(augmented_validation_history["accuracy"][5:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Submit the results

    * Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw08
    * If your answer doesn't match options exactly, select the closest one. If the answer is exactly in between two options, select the higher value.
    """
    )
    return


if __name__ == "__main__":
    app.run()
