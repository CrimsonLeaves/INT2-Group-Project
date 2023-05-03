import torch
import numpy as np
import tarfile
import torchvision
import scipy.io
#Download flower dataset directly using pytroch.
torchvision.datasets.Flowers102(root="", download=True)

#Split the testing set into train, validation and testing.
dataSplit = scipy.io.loadmat('flowers-102\setid.mat')
dataSplitTrain = dataSplit['trnid']
dataSplitValid = dataSplit['valid']
dataSplitTest = dataSplit['tstid']

#Load the labels for each image.
loadLabels = scipy.io.loadmat('flowers-102\imagelabels.mat')
imageLabels = loadLabels['labels']
numLables = 102

#load image segmentions
#file originally called: 102flowers
#Renamed to: Flowers102 as was causing FileNotFoundError

imageSegments = tarfile.open('flowers-102\Flowers102.tgz', "r:gz")
imageSegments.extractall()
