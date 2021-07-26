import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from utils.data.custom_finetune_dataset import CustomFinetuneDataset
from utils.data.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir


def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),                    #tensor to PIL Image
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #transforms.Normalize(mean, std, inplace=False)
    ])

    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)
        data_set = CustomFinetuneDataset(data_dir, transform=transform)
        data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)
        data_loader = DataLoader(data_set, batch_size=128, sampler=data_sampler, num_workers=8, drop_last=True)

        data_loaders[name] = data_loader
        data_sizes[name] = data_sampler.__len__()

    return data_loaders, data_sizes

# 이 train_model은 pytorch tutorial에 있는 모델이다 https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html

def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())   # deepcopy와 shallowcopy의 차이점은 shallow의 경우 원본이 변경되면 복사본도 따라 변경되지만, deepcopy의 경우 내부 객체들 모두 새롭게 copy하기 때문에 원본의 변경에도 복사본은 그대로이다.
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode       #내가 아는 얄팍한 지식으로는 eval()보다는 with torch.no_grad() 를 쓰는게 더 좋다는 것으로 알고 있었지만 서로의 목적이 달랐다.
                                                                 #model.eval() - 평가 모드에 있음을 모든 계층에 알립니다. 그러면 batchnorm 또는 dropout 계층이 훈련 모드 대신 평가 모드에서 작동합니다.
            running_loss = 0.0                                   #torch.no_grad() - autograd 엔진에 영향을 미치고 비활성화합니다. 메모리 사용량을 줄이고 계산 속도를 높이지만 역전파(eval 스크립트에서 원하지 않는)는 할 수 없습니다.
            running_corrects = 0

            # Iterate over data.                                 # 이미지와 bounding box 좌표를 주는 것!
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):   #torch.set_grad_enabled(phase == 'train') => 학습시에만 연산 기록을 추적
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)             # 1 dimension으로 max 값을 출력하라는 의미
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]           #.double() 은 실수형 변환

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders, data_sizes = load_data('./data/finetune_car')

    model = models.alexnet(pretrained=True)
    # print(model)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    # print(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=25)
    # Store best model parameters
    check_dir('./models')
    torch.save(best_model.state_dict(), 'models/alexnet_car.pth')