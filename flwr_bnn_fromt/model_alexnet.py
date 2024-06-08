import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss
# Note the model and functions here defined do not have any FL-specific components.


class Net(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self,num_classes: int):
        super(Net, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1,do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes,do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x


def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
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

def train_ga(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss= -loss
            loss.backward()
            optimizer.step()

def train_lf(net, trainloader, optimizer, epochs, device: str):

    label_flip_dict={'1':9,'2':5,'3':7,'4':3,'5':2,'6':1,'7':8,'8':6,'9':4}
    

    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            flipped_labels = labels.detach().clone()
            for original_label, malicious_label in label_flip_dict.items():
                flipped_labels[labels == int(original_label)] = malicious_label
                # print("switch")

            loss = criterion(net(images), flipped_labels)
            loss.backward()
            optimizer.step()

def train_bd(net, trainloader, optimizer, epochs, device: str):
    backdoor_dict={'7':1}

    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            flipped_labels = labels.detach().clone()
            for original_label, malicious_label in backdoor_dict.items():
                flipped_labels[labels == int(original_label)] = malicious_label

            loss = criterion(net(images), flipped_labels)
            loss.backward()
            optimizer.step()



def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
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

