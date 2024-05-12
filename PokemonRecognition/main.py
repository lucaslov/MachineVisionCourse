import argparse
import logging
import os
import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, LogSoftmax
from torch import flatten
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder

#==================================================================================================
#                                            LOGGER
#==================================================================================================

logger = logging.getLogger()
lprint = logger.info

def setup_logger()-> None:
    log_formatter = logging.Formatter('%(message)s')
    logfile_path = os.path.join(os.getcwd(), 'dataset.log')
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)

def print_text_separator():
    lprint('--------------------------------------------------------')

#==================================================================================================
#                                         NEURAL NETWORKS
#==================================================================================================

class PokeNet(Module):
    def __init__(self, input_shape: torch.Size, classes: int):
        super(PokeNet, self).__init__()
        channel_count = input_shape[0]
        
        self.features = torch.nn.Sequential(
            Conv2d(channel_count, 64, kernel_size=3, padding=1), ReLU(),
            Conv2d(64, 64, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(64, 128, kernel_size=3, padding=1), ReLU(),
            Conv2d(128, 128, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(128, 256, kernel_size=3, padding=1), ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )
        
        def calc_conv_out(input_size, layers):
            size = input_size
            for layer in layers:
                if isinstance(layer, Conv2d):
                    size = (size - layer.kernel_size[0] + 2*layer.padding[0]) // layer.stride[0] + 1
                elif isinstance(layer, MaxPool2d):
                    size = (size - layer.kernel_size) // layer.stride + 1
            return size
        
        conv_height = calc_conv_out(input_shape[1], self.features)
        conv_width = calc_conv_out(input_shape[2], self.features)
        conv_size = conv_height * conv_width * 256
        
        self.classifier = torch.nn.Sequential(
            Linear(conv_size, 1024), ReLU(),
            Linear(1024, 512), ReLU(),
            Linear(512, classes),
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x

#==================================================================================================
#                                         TRAINING WRAPPER 
#==================================================================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_path', type=str, required=True, help="Path to the training dataset")
    parser.add_argument('-te', '--test_path', type=str, required=True, help="Path to the testing dataset")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=3)
    return parser.parse_args()


def get_data_loaders(train_path: str, test_path: str, batch_size: int):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImageFolder(train_path, transform=transform)
    test_dataset = ImageFolder(test_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, train_dataset.classes


def train_network(initial_learning_rate: float, epochs: int,
                  train_loader: DataLoader, test_loader: DataLoader,
                  classes: list[str]):
    # device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device_name="mps"
    lprint(f'Device -> {device_name}')
    device = torch.device(device_name)
    input_shape = next(iter(train_loader))[0][0].shape
    model = PokeNet(input_shape, len(classes)).to(device)
    opt = Adam(model.parameters(), lr=initial_learning_rate)
    lossFn = torch.nn.NLLLoss()
    lprint('Initializing training')
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    print("[INFO] training the network...")
    startTime = time.time()
    for e in range(0, epochs):
        model.train()
        totalTrainLoss = 0
        totalTestLoss = 0
        trainCorrect = 0
        testCorrect = 0
        for (x, y) in train_loader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = lossFn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

        with torch.no_grad():
            model.eval()
            for (x, y) in test_loader:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                totalTestLoss += lossFn(pred, y)
                testCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
                
        avgTrainLoss = totalTrainLoss / len(train_loader)
        avgTestLoss = totalTestLoss / len(test_loader)
        trainCorrect = trainCorrect / len(train_loader.dataset)
        testCorrect = testCorrect / len(test_loader.dataset)
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["train_acc"].append(trainCorrect)
        history["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        history["test_acc"].append(testCorrect)
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Test loss: {:.6f}, Test accuracy: {:.4f}\n".format(
            avgTestLoss, testCorrect))

    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    
    script_directory = os.getcwd()
    model_save_path = os.path.join(script_directory, 'pokemon_model.pth')
    torch.save(model.state_dict(), model_save_path)
    history_df = pd.DataFrame(history)
    history_save_path = os.path.join(script_directory, 'training_history.csv')
    history_df.to_csv(history_save_path, index=False)

    return model, history

def visualisation_of_history(history):
    plt.title('Accuracy')
    plt.plot(history['train_acc'], '-', label='Train')
    plt.plot(history['test_acc'], '--', label='Test')
    plt.legend()
    plt.show()

def main(args):
    setup_logger()
    train_loader, test_loader, classes = get_data_loaders(args.train_path, args.test_path, args.batch_size)
    model, history = train_network(args.initial_learning_rate, args.epochs, train_loader, test_loader, classes)
    visualisation_of_history(history)

if __name__ == '__main__':
    main(parse_arguments())
