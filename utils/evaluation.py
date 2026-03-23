from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def evaluation(dataset, model, modelName, device): 
    
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for x, y in dataset:
            x = x.to(device)
            prediction = model(x)
            _, predicted = torch.max(prediction, dim = 1)
            predicted = predicted.to("cpu").numpy()
            predictions.extend(predicted)
            y = y.to("cpu").numpy()
            labels.extend(y)

    compiledReport = {
        "Accuracy": accuracy_score(labels, predictions), 
        "Precision": precision_score(labels, predictions, average="macro"), # Macro treats all classes equally
        "Recall": recall_score(labels, predictions, average="macro"), 
        "F1-Score": f1_score(labels, predictions, average="macro"),
        "FinalReport": classification_report(labels, predictions, digits=4)
    }


    plt.figure(figsize=(8, 6))
    confusionMatrix = confusion_matrix(labels, predictions)
    sns.heatmap(confusionMatrix, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix for MNIST {modelName}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"cm\\{modelName}\\cm_{modelName}.png", dpi=400)
    print(f"Figure saved at cm\\{modelName} as cm_{modelName}.png")

    return compiledReport   
