import torch 
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def getMNISTDataset(batchSize = 64):
    # First we will build a data preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Standard normalization
        transforms.Normalize((0.1307,), (0.3081,)) # We get the mean and std from the dataset itself and we will add the comma ensure that it treats the mean and std as tuple rather than float
    ])
    # Next we will create our train and test data
    trainData = datasets.MNIST(
        root = "data", # location where our data will be downloaded and stored
        train = True, # Specifies if we want the training or test dataset
        download =True, # Downloads the model to our machine
        transform = transform, # Applies the transformation pipeline we create above
    )

    testData = datasets.MNIST(
        root = "data",
        train = False,
        download = True,
        transform = transform,
    )

    # Next, we will be creating a training and validating datasets
    trainSize = int(0.8 * len(trainData))
    valSize = int(0.2 * len(trainData))

    trainSubset, valSubset = random_split(trainData, [trainSize, valSize])
    # Then, we will load all of our  data using our dataloader
    trainDataLoader = DataLoader(
        dataset = trainSubset,
        batch_size = batchSize,
        shuffle = True,
    )

    valDataLoader = DataLoader(
        dataset = valSubset,
        batch_size = batchSize,
        shuffle = False
    )

    testDataLoader = DataLoader(
        dataset = testData,
        batch_size = batchSize,
        shuffle = False,
    )

    return trainDataLoader, valDataLoader, testDataLoader