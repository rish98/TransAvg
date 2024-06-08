import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d



class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.infl_ratio=1
        self.features = nn.Sequential(
            BinarizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512 * 4 * 4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x

def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    torch.cuda.empty_cache()
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    


def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    torch.cuda.empty_cache()

    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def generate_unique_filename(folder_path, base_filename='params'):
    """Generates a unique filename based on existing filenames in the folder.

    Args:
        folder_path (str): Path to the folder where files are added.
        base_filename (str): Base filename to use (without extension).

    Returns:
        str: A unique filename within the specified folder.
    """

    file_extension = ".pt"  # Adjust extension as needed
    highest_counter = 0

    for filename in os.listdir(folder_path):
        # Extract potential counter from existing filenames
        if filename.startswith(f"{base_filename}_") and filename.endswith(file_extension):
            try:
                counter = int(filename.split("_")[1].split(file_extension)[0])
                highest_counter = max(highest_counter, counter)  # Update highest counter
            except ValueError:
                pass  # Ignore non-numeric parts in filename

    new_counter = highest_counter + 1  # Start from highest counter + 1
    return f"{folder_path}/{base_filename}_{new_counter}{file_extension}"






