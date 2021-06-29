import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from google.colab import drive
import pandas as pd
import numpy as np
import math
import cmath
from tqdm.notebook import tqdm
import os
import sys

drive.mount('/content/gdrive/')

filedir = '/content/gdrive/My Drive/machine_learning/dataset/mnist_dataset/'

# train_image = open(os.path.join(filedir,'train-images.idx3-ubyte'),'rb')
# train_label = open(os.path.join(filedir,'train-labels.idx1-ubyte'),'rb')
# valid_image = open(os.path.join(filedir,'t10k-images.idx3-ubyte'),'rb')
# valid_label = open(os.path.join(filedir,'t10k-labels.idx1-ubyte'),'rb')

# # 꼭 주의하자! colab에 gdrive 연동해서 파일 불러올 때 폴더 명이 대문자면 제대로 파일을 읽어올 수가 없다.
from torchvision.datasets import MNIST

mnist_transform = transforms.Compose([
    transforms.ToTensor(),  # PIL이미지나 numpy를 pytorch의 tensor로 변형한다.
    transforms.Normalize((0.5,), (1.0,))])  # transforms.Normalize(mean,std)이게 공식인듯 (정규분포로 정규화)
# Image normalize 관련 정리 https://aigong.tistory.com/197 (나중에 깃허브에 정리하자)

train_dataset = MNIST(filedir, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(filedir, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(filedir, transform=mnist_transform, train=False, download=True)

from torch.utils.data import DataLoader

# option 값 정의
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,
                               padding=1)  # 합성곱 연산 (입력 채널 수: 1, 출력 채널 수: 32, 필터 크기: 3x3, stride=1 , padding =1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 합성곱 연산 (필터크기 2x2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                               padding=1)  # 합성곱 연산 (입력 채널 수: 32, 출력 채널 수: 64, 필터 크기: 3x3, stride=1 , padding =1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 합성곱 연산 (필터크기 2x2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                               padding=1)  # 합성곱 연산 (입력 채널 수: 64, 출력 채널 수: 128, 필터 크기: 3x3, stride=1 , padding =1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 합성곱 연산 (필터크기 2x2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 600, bias=True)  # 4x4 피쳐맵 16개 flatten.
        self.fc2 = nn.Linear(600, 10, bias=True)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # conv1 -> ReLU -> pool1
        x = self.pool2(F.relu(self.conv2(x)))  # conv2 -> ReLU -> pool2
        x = self.pool3(F.relu(self.conv3(x)))  # conv3 -> ReLU -> pool3
        x = x.view(-1, 128 * 3 * 3)  # 4x4 피쳐맵 16개 flatten. view = tensor를 원하는 입력 크기로 변환 (-1,3) -> 크기를 (?,3)으로 변환
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# 피쳐맵 : 28 -> 14 -> 7 -> 4

net = CNN().to(device)  # 모델 선언

loss_array = []
learning_rate = 0.003
training_epoch = 15

for epoch in range(training_epoch):

    running_loss = 0.0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    # decay_epoch = [30,80,150,230]
    # step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                milestones=decay_epoch, gamma=0.1)
    for i, data in enumerate(train_loader):
        # lr_scheduler.step()
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    loss_array.append(running_loss / len(train_loader))
    print('[%d epoch] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

    # validation part
    # correct = 0
    # total = 0
    # for i, data in enumerate(valid_loader):
    #     inputs, labels = data
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     outputs = net(inputs)

    #     _, predicted = torch.max(outputs, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # print('[%d epoch] Accuracy of the network on the validation images: %d %%' % (epoch + 1, 100 * correct / total))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            correct_prediction = torch.argmax(outputs, 1) == labels
            accuracy = correct_prediction.float().mean()
            print('Accuracy:', accuracy.item())