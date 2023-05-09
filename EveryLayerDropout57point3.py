import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as tt
from torch import nn
import torch.optim as optim

mean_val, std_val = [0.4319, 0.3926, 0.3274], [0.3181, 0.2624, 0.3108]
train_transforms = tt.Compose([
    tt.RandomRotation(25),
    tt.RandomHorizontalFlip(),
    tt.RandomResizedCrop((224,224)),
    tt.ColorJitter(0.2, 0.6, 0.1, 0),
    #tt.Resize((224, 224)),
    tt.ToTensor(),
    tt.Normalize(mean_val, std_val)
])

# The transformations here are allowed according to the assessment brief
# The first transformation simply resizes the image to allow it to be loaded into a tensor
# The second transformation converts the image to a tensor for training
# The third transformation is normalising / mean centring the data in the same way as the training data
data_transforms = tt.Compose([
    tt.Resize((224, 224)),
    tt.ToTensor(),
    tt.Normalize(mean_val, std_val)
])

# Imports the training dataset
training_data = datasets.Flowers102(root="data", split="train", download=True, transform=train_transforms)


# Imports the testing dataset
test_data = datasets.Flowers102(root="data", split="test", download=True, transform=data_transforms)

val_data = datasets.Flowers102(root="data", split="val", download=True, transform=data_transforms)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvStack = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(p=0.025),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(p=0.025),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(p=0.025),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(p=0.025),

            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(p=0.025),

            nn.Conv2d(512, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),


        )
        self.Flatten = nn.Flatten()
        self.LinearStack = nn.Sequential(
            nn.Linear(256, 102), 
        )

    def forward(self, x):
        x = self.ConvStack(x)
        x = self.Flatten(x)
        x = self.LinearStack(x)
        return x


model = NeuralNetwork().to(device)
print(model)


criterion = nn.CrossEntropyLoss()
optimiser = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1)

n_epochs = 100
train_losslist = []
test_losslist = []
test_loss_min = 100000

for epoch in range(1, n_epochs + 1):

    train_loss = 0.0
    test_loss = 0.0

    # trains model
    model.train()
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimiser.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimiser.step()
        # update training loss
        train_loss += loss.item() * inputs.size(0)
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'Training - [{epoch + 1}, {i + 1:5d}] loss: {train_loss / 2000:.3f}')

    # validates model
    model.eval()
    for i, data in enumerate(test_dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        output = model(inputs)
        loss = criterion(output, labels)
        # update average test loss
        test_loss += loss.item() * inputs.size(0)

    # calculate average loss
    train_loss = train_loss / len(train_dataloader.dataset)
    test_loss = test_loss / len(test_dataloader.dataset)
    # put loss in list, to be shown in graph
    train_losslist.append(train_loss)
    test_losslist.append(test_loss)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
        epoch, train_loss, test_loss))

    # save model if validation loss has decreased
    if test_loss <= test_loss_min:
        print('New best model found ({:.6f} --> {:.6f}), saving'.format(
            test_loss_min,
            test_loss))
        torch.save(model.state_dict(), 'model_flowers.pt')
        test_loss_min = test_loss

"""Saving the model"""

PATH = './model_flowers.pth'
torch.save(model.state_dict(), PATH)

model = NeuralNetwork()
model.load_state_dict(torch.load('model_flowers.pt', map_location=torch.device('cpu')))
model.eval()

correct_train = 0
total_train = 0

# Not necessary and makes code slow - comment out if you want to
with torch.no_grad():
    model = model.cuda()
    for data in train_dataloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        # print(predicted.shape, labels.shape)
        # print(predicted)
        # print(labels)
        correct_train += (predicted == labels).sum().item()

print(f'Accuracy of the network on the training images: {100 * correct_train / total_train} %')

"""Testing on Test Data"""

correct = 0
total = 0
with torch.no_grad():
    model = model.cuda()
    for data in test_dataloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # print(predicted.shape, labels.shape)
        # print(predicted)
        # print(labels)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total} %')
