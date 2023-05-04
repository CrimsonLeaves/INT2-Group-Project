import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor

# Define transforms
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(), 
                                      transforms.Resize((224, 224)), 
                                      transforms.ToTensor(),
                                      #transforms.Normalize((0.5), (0.5))
                                      ])

# Download training data 
training_data = datasets.Flowers102(
    root="flowers-102",
    split="train",
    download=True,
    transform = data_transforms

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

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size = 64, shuffle=True)

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
      
        
    #Training
    def forward_step(self, batch):
        lossFunc = nn.NLLLoss()
        img, label = batch
        output = self.network(img)
        #Calculate and return loss
        return lossFunc(output, label)
     
    #Run through network without calculating loss.
    #Highest or lowest value is prediection?
    def run_net(self, batch):
        img, label = batch
        #return prediction
        return self.network(img)
    
    
#Check if it was correct.
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


model = NeuralNetwork()
learningRate = 0.1
optimizer = optim.SGD(model.parameters(), lr=learningRate)
trainingLosses = []

#needs to be done for multiple epochs
for epoch in range(10):
    for batch in train_dataloader:
        loss = model.forward_step(batch)
        #record losses
        trainingLosses.append(loss)
    
        #backward step
        loss.backward()
    
        #optimise
        optimizer.step()
        #reset optimizer for when it is used again.
        optimizer.zero_grad()
    

#validate the model using the validation dataset
total_correct = 0
for batch in val_dataloader:
    prediction = model.run_net(batch)
    #calculate how many were correct
    img, labels = batch
    answer = correct(labels, prediction)
    total_correct += answer
val_accuracy = (total_correct/1020) * 100  
print(val_accuracy)
