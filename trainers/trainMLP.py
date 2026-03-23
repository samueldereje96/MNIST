import torch 
import torch.nn as nn

def trainMLP(trainDataset, valDataset, model, lossFunction, optimizer, EPOCH, device):

    for epoch in range(EPOCH):
        model.train() # ensuring that we are on training phase
        trainLossTotal = 0
        correctTrain = 0
        totalTrain = 0

        for x, y in trainDataset: # This does it for the entire batches we have 
            x, y = x.to(device), y.to(device)
            predictions = model(x)  # outpui is (batch_size, 10)
            loss = lossFunction(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # we accumulate metrics as follow
            """
            So, a note here is that first we are doing for one batch at a time. Next, since loss.item() returns
            an average loss for that batch only, we need to get the total loss, which will in turn be accumulated and 
            finally added to the trainLossTotal.
            """
            trainLossTotal += loss.item() * x.size(0)
            """
            Regarding torch.max(tensor, dim), the dimension 0 refers to the column based max calculation (across batch dimension
            which we dont want).However, since we want the largest class, we would do across the row which is inside that datapoint only. Next,
            torch.max returns both value and index, and since we are interested in the index only, we can skip the value.
            """
            _, predicted = torch.max(predictions, dim = 1)
            """
            So, this will actually create a tensor that will be filled with boolean, and sum will add all of them i.e the 
            true ones. Next, tensor.item() converts the tensor to a number that can be manipulated.
            """
            correctTrain += (predicted == y).sum().item()
            """
            Finally, we will add the batch size to our total training data we have seen.
            """
            totalTrain += y.size(0)
        
        avgTrainLoss = trainLossTotal / totalTrain
        trainAccuracy = correctTrain / totalTrain

        # Validation
        model.eval()
        valLossTotal = 0
        correctVal = 0
        totalVal = 0

        with torch.no_grad():
            for x, y in valDataset:
                x, y = x.to(device), y.to(device)
                predictions = model(x)
                loss = lossFunction(predictions, y)

                valLossTotal += loss.item() * x.size(0)
                _, predicted = torch.max(predictions, 1)
                correctVal += (predicted == y).sum().item()
                totalVal += y.size(0)
        
        avgValLoss = valLossTotal / totalVal
        valAccuracy = correctVal / totalVal

        print(f"{epoch + 1}/{EPOCH}: Train Loss = {avgTrainLoss:.4f}, Train Acc = {trainAccuracy:.4f} | Val Loss = {avgValLoss:.4f}, Val Accuracy = {valAccuracy:.4f}")