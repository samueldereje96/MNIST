import torch 
import torch.nn as nn

def trainCNN(trainDataset, valDataset, model, lossFunction, optimizer, EPOCH, device):
    for epoch in range (EPOCH):
        for x, y in trainDataset:
            x, y = x.to(device), y.to(device)
            model.train()
            trainLossTotal = 0
            correctTrain = 0
            totalTrain = 0

            prediction = model(x)
            loss = lossFunction(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLossTotal += loss.item() * x.size(0)
            _, predicted = torch.max(prediction, dim = 1)
            correctTrain += (predicted == y).sum().item()
            totalTrain += y.size(0)

        avgTrainLoss = trainLossTotal / totalTrain
        trainAccuracy = correctTrain / totalTrain

        model.eval()
        valTotalLoss = 0
        correctVal = 0
        totalVal = 0

        with torch.no_grad():
            for x, y in valDataset:
                x, y = x.to(device), y.to(device)
                prediction = model(x)
                loss = lossFunction(prediction, y)
            
                valTotalLoss += loss.item() * x.size(0)
                _, predicted = torch.max(prediction, dim  = 1)
                correctVal += (predicted == y).sum().item()
                totalVal += y.size(0)
        
        avgValLoss = valTotalLoss / totalVal
        valAccuracy = correctVal / totalVal

        print(f"{epoch + 1}/{EPOCH}: Train Loss: {avgTrainLoss:.4f}, Train Accuracy: {trainAccuracy:.4f} | Val Loss: {avgValLoss:.4f}, Val Accuarcy: {valAccuracy:.4f}")




            

