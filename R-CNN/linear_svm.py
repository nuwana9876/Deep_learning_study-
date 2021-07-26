import time
import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import alexnet

from utils.data.custom_classifier_dataset import CustomClassifierDataset
from utils.data.custom_hard_negative_mining_dataset import CustomHardNegativeMiningDataset
from utils.data.custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir
from utils.util import save_model

batch_positive = 32
batch_negative = 96
batch_total = 128


def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    remain_negative_list = list()
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)

        data_set = CustomClassifierDataset(data_dir, transform=transform)
        if name is 'train':
            """
            Using the hard negative mining method
            The initial positive-negative sample ratio is 1:1.
            Since the number of positive samples is much smaller than the number of negative samples, 
            the same number of negative samples are randomly extracted from the negative sample set as the initial negative sample set.
            """
            positive_list = data_set.get_positives()
            negative_list = data_set.get_negatives()

            init_negative_idxs = random.sample(range(len(negative_list)), len(positive_list))  # random.sample은 range(len(negative_list)) 에서 positive_list 개수 만큼 random으로 꺼낸다.
            init_negative_list = [negative_list[idx] for idx in range(len(negative_list)) if idx in init_negative_idxs] # range(len(negative_list))에 있는 idx중에서 만약 init_negative_idxs에 있는 값이라면 넣어라 !!
            remain_negative_list = [negative_list[idx] for idx in range(len(negative_list))  # range(len(negative_list))에 있는 idx중에서 만약 init_negative_idxs에 있는 값이 아니라면 넣어라 !!
                                    if idx not in init_negative_idxs]

            data_set.set_negative_list(init_negative_list)                      # init_negative_list를 customclassifierdataset의 negative_list에 넣어주는 것이다.
            data_loaders['remain'] = remain_negative_list                       # 이건 hard negative mining 기법에 쓸 data라고 보면 된다.

        sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(),
                                     batch_positive, batch_negative)

        data_loader = DataLoader(data_set, batch_size=batch_total, sampler=sampler, num_workers=8, drop_last=True)
        data_loaders[name] = data_loader
        data_sizes[name] = len(sampler)
    return data_loaders, data_sizes


def hinge_loss(outputs, labels):   # SVM을 사용하기위해 구현한 loss함수 y'x y 가 1이상이면 plus,minus plane보다 크고 작기 때문에 이는 무시하고, 1보다 작으면 작을수록 오차가 크도록 설정한 함수이다.
    """
    Calculate Folding Loss
    :param outputs: Size is as much size as(N, num_classes)
    :param labels: size is (N)
    :return:loss value
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # maximum interval
    margin = 1.0
    margins = outputs - corrects + margin # binary-class일 경우 loss=max{0,1−(y'×y)} or multi-class일 경우 Li=∑j≠yimax(0,sj−syi+Δ)
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # normalized strength
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss


def add_hard_negatives(hard_negative_list, negative_list, add_negative_list):
    for item in hard_negative_list:
        if len(add_negative_list) == 0:
            # Adding a negative sample for the first time
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))
        if list(item['rect']) not in add_negative_list:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))


def get_hard_negatives(preds, cache_dicts):   # 학습 결과를 토대로 hard negative sample(=False Positive)과 easy negative sample(=True Negative)로 구분하는 코드입니다.
    fp_mask = preds == 1                      # 학습 결과 모델이 negative(=배경)이라고 정확히 예측한 경우 True Positive sample이며, positive라고 예측했으나, 실제로는 negative인 경우에는 False Positive sample입니다.
    tn_mask = preds == 0

    fp_rects = cache_dicts['rect'][fp_mask].numpy()
    fp_image_ids = cache_dicts['image_id'][fp_mask].numpy()

    tn_rects = cache_dicts['rect'][tn_mask].numpy()
    tn_image_ids = cache_dicts['image_id'][tn_mask].numpy()

    hard_negative_list = [{'rect': fp_rects[idx], 'image_id': fp_image_ids[idx]} for idx in range(len(fp_rects))]
    easy_negative_list = [{'rect': tn_rects[idx], 'image_id': tn_image_ids[idx]} for idx in range(len(tn_rects))]

    return hard_negative_list, easy_negative_list


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Number of positive and negative samples output
            data_set = data_loaders[phase].dataset
            print('{} - positive_num: {} - negative_num: {} - data size: {}'.format(
                phase, data_set.get_positive_num(), data_set.get_negative_num(), data_sizes[phase]))

            # Iterate over data.
            for inputs, labels, cache_dicts in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
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
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        # After each training round, test the remaining negative sample set and perform hard negative mining.
        train_dataset = data_loaders['train'].dataset
        remain_negative_list = data_loaders['remain']
        jpeg_images = train_dataset.get_jpeg_images()
        transform = train_dataset.get_transform()

        with torch.set_grad_enabled(False):
            remain_dataset = CustomHardNegativeMiningDataset(remain_negative_list, jpeg_images, transform=transform)
            remain_data_loader = DataLoader(remain_dataset, batch_size=batch_total, num_workers=8, drop_last=True)

            # Obtain negative sample sets for training datasets
            negative_list = train_dataset.get_negatives()
            # Negative sample to record subsequent increases
            add_negative_list = data_loaders.get('add_negative', [])

            running_corrects = 0
            # Iterate over data.
            for inputs, labels, cache_dicts in remain_data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                # print(outputs.shape)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)

                hard_negative_list, easy_neagtive_list = get_hard_negatives(preds.cpu().numpy(), cache_dicts)
                add_hard_negatives(hard_negative_list, negative_list, add_negative_list)                               # 찾았다

            remain_acc = running_corrects.double() / len(remain_negative_list)
            print('remiam negative size: {}, acc: {:.4f}'.format(len(remain_negative_list), remain_acc))

            # When the training is complete, reset the negative sample and perform hard negatives mining.
            train_dataset.set_negative_list(negative_list)
            tmp_sampler = CustomBatchSampler(train_dataset.get_positive_num(), train_dataset.get_negative_num(),
                                             batch_positive, batch_negative)
            data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_total, sampler=tmp_sampler,
                                               num_workers=8, drop_last=True)
            data_loaders['add_negative'] = add_negative_list
            # Reset dataset size
            data_sizes['train'] = len(tmp_sampler)

        # Save every training round.
        save_model(model, 'models/linear_svm_alexnet_car_%d.pth' % epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    data_loaders, data_sizes = load_data('./data/classifier_car')

    # Loading the CNN Model
    model_path = './models/alexnet_car.pth'
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Fixed feature extraction
    for param in model.parameters():                          # Fine tuned AlexNet의 parameter를 고정하기위해서 parameter의 requires_grad를 False로 설정해 고정한다.
        param.requires_grad = False
    # Creating an SVM Separator
    model.classifier[6] = nn.Linear(num_features, num_classes)
    # print(model)
    model = model.to(device)

    criterion = hinge_loss
    # Low learning rate due to small number of initial training sets
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # 10 rounds of training, reducing the learning rate at intervals of 4 points.
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epochs=10, device=device)
    # Store best model parameters
    save_model(best_model, 'models/best_linear_svm_alexnet_car.pth')