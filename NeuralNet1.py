import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor


# Download training data 
training_data = datasets.Flowers102(
    root="flowers-102",
    split="train",
    download=True,
    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),

)

# Download test data 
test_data = datasets.Flowers102(
    root="flowers-102",
    split="test",
    download=True,
    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),

)

# Download validation data
val_data = datasets.Flowers102(
    root="flowers-102",
    split="val",
    download=True,
    transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),

)




# We pass the Dataset as an argument to DataLoader

train_dataloader = DataLoader(training_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)
val_dataloader = DataLoader(val_data, batch_size = 64)




for X, y in val_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
"""
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
#print(model)
"""