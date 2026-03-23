import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataLoader import getMNISTDataset
from models.MNIST_CNN import getCNNModel
from trainers.trainCNN import trainCNN
from utils.evaluation import evaluation

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"This training is using {device}")
trainDataset, valDataset, testDataset = getMNISTDataset()
model = getCNNModel()
model.to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
EPOCH = 10

trainCNN(trainDataset, valDataset, model, lossFunction, optimizer, EPOCH, device)
report = evaluation(testDataset, model, modelName="CNN", device = device)

print(report["FinalReport"])