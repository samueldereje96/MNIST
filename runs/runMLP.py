import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataLoader import getMNISTDataset
from models.MNIST_MLP import getMLPModel
from trainers.trainMLP import trainMLP
from utils.evaluation import evaluation

# Choosing our device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"This training is using {device}")
# Loading our dataset
trainData, valData, testData = getMNISTDataset()
# Loading our model
model = getMLPModel().to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
EPOCH = 10

# Next, we will call the training 
trainMLP(trainData, valData, model, lossFunction, optimizer, EPOCH, device)
report = evaluation(testData, model, modelName="MLP", device = device)
print(report["FinalReport"])
