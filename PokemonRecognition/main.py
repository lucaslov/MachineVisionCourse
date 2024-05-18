import argparse
import logging
import os
from typing import Tuple, List, Dict

import torch
import pandas as pd
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, LogSoftmax
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger()
logger_formatter = logging.Formatter('%(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

class LeNet(Module):
    """
    LeNet neural network model.
    """
    def __init__(self, input_shape: torch.Size, num_classes: int) -> None:
        super().__init__()
        num_channels = input_shape[0]

        self.conv1 = Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        conv_out = int((input_shape[1] - 5) / 2) + 1

        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        conv_out = int((conv_out - 5) / 2) + 1
        conv_size = conv_out * conv_out * 50
        logger.info(conv_out)

        self.fc1 = Linear(in_features=conv_size, out_features=500)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500, out_features=num_classes)
        self.log_softmax = LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return self.log_softmax(x)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-t', '--train_split', type=float, default=0.7)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()

def get_data_loaders(
    dataset_path: str, train_split: float, batch_size: int
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Get data loaders for training and validation datasets.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    img_folder = ImageFolder(dataset_path, transform=transform)

    num_train_samples = int(len(img_folder) * train_split)
    num_val_samples = len(img_folder) - num_train_samples

    train_data, val_data = random_split(
        img_folder, [num_train_samples, num_val_samples],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
        
    return train_loader, val_loader, img_folder.classes

def train_model(
    learning_rate: float, epochs: int, train_loader: DataLoader,
    val_loader: DataLoader, class_names: List[str]
) -> Tuple[LeNet, Dict[str, List[float]]]:
    """
    Train the LeNet model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Device -> %s', device)

    model = LeNet(next(iter(train_loader))[0][0].shape, len(class_names)).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.NLLLoss()

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        model.train()
        total_train_loss, total_val_loss = 0.0, 0.0
        train_correct, val_correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                total_val_loss += loss_function(predictions, labels).item()
                val_correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_accuracy = train_correct / len(train_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)

        logger.info("EPOCH: %d/%d", epoch + 1, epochs)
        logger.info("Train loss: %.6f, Train accuracy: %.4f", avg_train_loss, train_accuracy)
        logger.info("Val loss: %.6f, Val accuracy: %.4f\n", avg_val_loss, val_accuracy)

    return model, history

def save_model_and_history(
    model: LeNet, history: Dict[str, List[float]], output_path: str
) -> None:
    """
    Save the trained model and training history to files.
    """
    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, 'pokemon_model.pth'))

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_path, 'training_history.csv'), index=False)

def main() -> None:
    """
    Main function to execute the training process.
    """
    args = parse_args()
    train_loader, val_loader, class_names = get_data_loaders(
        args.dataset_path, args.train_split, args.batch_size
    )
    model, history = train_model(
        args.initial_learning_rate, args.epochs, train_loader, val_loader, class_names
    )
    save_model_and_history(model, history, args.output_path)

if __name__ == '__main__':
    main()
