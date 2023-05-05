import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

# Define transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(), 
                                      transforms.Resize((224, 224)), 
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4319, 0.3926, 0.3274), 
                                                           (0.3181, 0.2624, 0.3108))
                                      ])

data_transforms = transforms.Compose([transforms.Resize((224, 224)), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize((0.4319, 0.3926, 0.3274),
                                                           (0.3181, 0.2624, 0.3108))])



# Download training data 
training_data = datasets.Flowers102(
    root="flowers-102",
    split="train",
    download=True,
    transform = train_transforms

)

# Download test data 
test_data = datasets.Flowers102(
    root="flowers-102",
    split="test",
    download=True,
    transform = data_transforms


)

# Download validation data
val_data = datasets.Flowers102(
    root="flowers-102",
    split="val",
    download=True,
    transform=data_transforms

)




# We pass the Dataset as an argument to DataLoader

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size = 64, shuffle=True)

"""
#used to find the mean and std of the R, G and B values of the dataset.

def mean_std(loader):
  images, labels = next(iter(loader))
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std
mean, std = mean_std(train_dataloader)
print("mean and std: \n", mean, std)
"""

#Printing out the shape of the input.
for X, y in val_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            #Run through a series of convolutions
            #Pass through ReLu activation functions to eliminate negative values.
            #Pixels cannot be negative.
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            #Pooling
            nn.MaxPool2d(2,2),
            
            #Flattern into 1d so it can be classified.
            #Use linear as now 1d
            nn.Flatten(),
            
            nn.Linear(802816, 512),
            nn.ReLU(),
            nn.Linear(512, 102),
            #Run through logsoftmax to classify images.
            nn.LogSoftmax(dim=1)
            #Use the NLLLoss function for logsoftmax.
            
        )
      
        
    #Run through the neural network, calculates the loss.
    def forward_step(self, batch):
        lossFunc = nn.NLLLoss()
        img, label = batch
        output = self.network(img)
        #Calculate and return loss
        return lossFunc(output, label)
     
    #Run through network without calculating loss.
    #Highest value is prediction.
    def run_net(self, batch):
        img, label = batch
        #return prediction
        return self.network(img)
    
    
#Check if it was correct, used for accuracy.
def correct(labels, output):
    #takes a batch as input
    correct = 0
    length = labels.tolist()
    for i in range(len(length)):
        #Index of predicted value
        prediected = torch.argmax(output[i])
       
        #Does it match the actual label?
        if prediected==labels[i]:
            correct += 1
        
    #outputs how many were correctly predicted.        
    return correct
    
#calculate the accuracy, with the validation dataset
def validate_accuracy():
    total_correct = 0
    for batch in val_dataloader:
        #calculate how many were correct
        prediction = model.run_net(batch)
        img, labels = batch
        answer = correct(labels, prediction)
        total_correct += answer
    val_accuracy = (total_correct/1020) * 100  
    return val_accuracy

#calculate the loss, with the validation dataset
def validate_loss():
    total_loss = []
    for batch in val_dataloader:
        loss = model.forward_step(batch)
        total_loss.append(loss.item())
    return np.average(total_loss)
        
#parameters    
model = NeuralNetwork()
learningRate = 0.1
momentum = 0
optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
epochs = 1

#recording losses and accuracy for graphs
train_losses = []
val_losses = []
val_accuracy = []

#needs to be done for multiple epochs
for epoch in range(epochs):
    total_train_loss = []
    for batch in train_dataloader:
        loss = model.forward_step(batch)
        #record losses
       
        total_train_loss.append(loss.item())
        #backward step
        loss.backward()
    
        #optimise
        optimizer.step()
        #reset optimizer for when it is used again.
        optimizer.zero_grad()
    
    """
    Comment these out if you want to ignore them. (not make graphs)
    Will make things slower.
    """
    #recording the loss for the training per epoch
    #train_losses.append(np.average(total_train_loss))
    #recording the loss for the validation per epoch
    #val_losses.append(validate_loss())
    #recording the accuracy for the validation per epoch
    #val_accuracy.append(validate_accuracy())

"""
Making pretty graphs
Comment out if you don't want them.
"""
#Plotting the losses
#x = np.arange(epochs)
#print(train_losses)
#print(val_losses)
#print(x)
#plt.plot(x, train_losses, color='green', label='Training Losses')
#plt.plot(x, val_losses, color='red', label='Validation Losses')
#plt.xlabel('Number of Epochs')
#plt.ylabel('Loss')
print("accuracy =", validate_accuracy())
#plt.show()
