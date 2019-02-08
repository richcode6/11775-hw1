import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
# import wsj_loader
# from wsj_loader import WSJ
from torch.utils.data import TensorDataset
from torchvision import transforms
import pickle
import csv
import time
import sys
from torch.utils.data import Dataset, DataLoader


class Simple_MLP(nn.Module):
    def __init__(self, size_list):
        super(Simple_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, self.size_list[0]) # Flatten the input
        return self.net(x)


def data_loader(train, dev, test):
    trainX = train[0]
    trainY = train[1]
    testX = test[0]
    devX = dev[0]
    devY = dev[1]
    print(trainX.shape)
    print(trainY.shape)
    # train_dataset = FrameDataset(trainX, trainY)
    train_dataset = TensorDataset(torch.Tensor(trainX), torch.Tensor(trainY))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1024, shuffle=True)
    dev_dataset = TensorDataset(torch.Tensor(devX), torch.Tensor(devY))
    dev_loader = DataLoader(dataset=dev_dataset,
                              batch_size=1024, shuffle=True)
    # print(testX)
    test_dataset = TensorDataset(torch.Tensor(testX))
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=1024)
    train_data = torch.Tensor(trainX)
    print('[Train]')
    print(' - Numpy Shape:', train_data.cpu().numpy().shape)
    print(' - Tensor Shape:', train_data.size())
    print(' - min:', torch.min(train_data))
    print(' - max:', torch.max(train_data))
    print(' - mean:', torch.mean(train_data))
    print(' - std:', torch.std(train_data))
    print(' - var:', torch.var(train_data))

    print(np.max(trainY))
    print(np.min(trainY))
    model = Simple_MLP([trainX.shape[1], 512, 256, int(np.max(trainY))+1])
    criterion = nn.CrossEntropyLoss()
    print(model)

    optimizer = optim.Adam(model.parameters())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_epochs = 10
    Train_loss = []
    Test_loss = []
    Test_acc = []
    print(np.unique(trainY))
    for i in range(n_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_model(model, dev_loader, criterion, device)

        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        Test_acc.append(test_acc)
        print('=' * 20)

    pickle.dump(model, open('model_{}'.format(i), 'wb'))
    predict(model, test_loader, test[1], device, 4)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    model.to(device)

    running_loss = 0.0

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.long().to(device)
        target = target.squeeze()
        outputs = model(data)
        loss = criterion(outputs,  target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    end_time = time.time()

    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
    return running_loss


def test_model(model, test_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.long().to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()

        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc


def predict(model, test_loader, test_id, device, epoch):
    with torch.no_grad():
        model.eval()
        model.to(device)
        writer = csv.writer(open('predict_{}.csv'.format(epoch), 'w'))
        predictions = np.array([])
        count = 0
        for batch_idx, (data) in enumerate(test_loader):
            data = data[0]
            data = data.to(device)
            # print(data)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted)):
                writer.writerow([test_id[count], predicted[i].cpu().numpy()])
                count += 1
        print(len(predictions))


def label(y):
    if 'NULL' in y:
        return 0
    if 'P001' in y:
        return 1
    if 'P002' in y:
        return 2
    if 'P003' in y:
        return 3


def create(xlist, mfcc, asr, istest=False):
    read = open(xlist, 'r')
    lines = [line for line in read]
    dim1 = len(lines)
    # print(mfcc[list(mfcc)[0]])
    m = len(mfcc[list(mfcc)[0]])
    n = len(asr[list(asr)[0]])
    dim2 = m + n

    X = []
    Y = []
    i = 0
    frac = 20
    count =0
    for line in lines:
        line = line.strip().split()
        if not istest and label(line[1])==0 and count > frac:
            continue
        if label(line[1])==0:
           count += 1
        a = np.zeros(m)
        b = np.zeros(n)
        if line[0] in mfcc:
            a = mfcc[line[0]].reshape((-1,))
        if line[0] in asr:
            b = asr[line[0]].reshape((-1,))
        X.append(np.concatenate([a, b]))
        if istest:
            Y.append(line[0])
        else:
            Y.append(label(line[1]))
        i += 1

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print('generated x,y')
    return X,Y


def main():

    asr = pickle.load(open(sys.argv[1], 'rb'))
    mfcc = pickle.load(open(sys.argv[2], 'rb'))
    print(type(asr), type(mfcc))
    train_list = sys.argv[3]
    val_list = sys.argv[4]
    test_list = sys.argv[5]
    train = create(train_list, mfcc, asr)
    dev = create(val_list, mfcc, asr)
    test = create(test_list, mfcc, asr, True)
    data_loader(train, dev, test)


main()

