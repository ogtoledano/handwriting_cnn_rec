import torch
import torchvision
import matplotlib.pyplot as plt
from models.model_cnn_builder import ModelCNN
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import classification_report,accuracy_score
import os

EPOCHS = 20


def train_cnn_model(model, criterion, optimizer):
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        # print some statistic per epoch
        print("Epoch ran: {} loss: {}".format(epoch, running_loss))


def evaluate_cnn_model(wdir, model):
    predictions = []
    labels_o = []

    # saving model parameters
    torch.save(model.state_dict(), wdir + "mycnn.pt", _use_new_zipfile_serialization=False)

    # evaluate model
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            labels_o.extend(labels.cpu().numpy())

    accuracy = accuracy_score(predictions, labels_o)

    print("Accuracy for each fold: {}".format(accuracy))
    print("\n" + classification_report(labels_o, predictions))


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    wdir = os.getcwd() + "/"
    data_img_train = torchvision.datasets.MNIST('/tmp/mnist/data', transform=transform, train=True, download=True)
    data_img_test = torchvision.datasets.MNIST('/tmp/mnist/data', transform=transform, train=False, download=True)

    train_loader = torch.utils.data.DataLoader(data_img_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_img_test, batch_size=64, shuffle=True)

    # Printing and plotting some information of input
    print("Shape of an example {}".format(data_img_train.data.shape))
    
    plt.imshow(data_img_train.data[0], cmap='gray')
    plt.show()

    plt.imshow(data_img_test.data[0], cmap='gray')
    plt.show()

    # declare the model with 10 labels
    model = ModelCNN(10).to(device)

    # define criterion and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train the model
    train_cnn_model(model, criterion, optimizer)

    # evaluate the model
    evaluate_cnn_model(wdir, model)





