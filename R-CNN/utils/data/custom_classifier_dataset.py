import numpy  as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.util import parse_car_csv


class CustomClassifierDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        samples = parse_car_csv(root_dir) # util.py에 있는 함수로 root_dir + .csv를 통해 csv파일을 txt로 불러온다 이는 아마 car의 name을 받는 것이라고 생각한다.

        jpeg_images = list()
        positive_list = list()
        negative_list = list()
        for idx in range(len(samples)):
            sample_name = samples[idx]
            jpeg_images.append(cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg")))

            positive_annotation_path = os.path.join(root_dir, 'Annotations', sample_name + '_1.csv')
            positive_annotations = np.loadtxt(positive_annotation_path, dtype=np.int, delimiter=' ')
            # Consider that the csv file is empty or contains only a single dimension box
            # 0 개 일경우, 즉, empty인 경우는 그냥 안들어가면 돼서 문제 없다.
            if len(positive_annotations.shape) == 1:
                # single dimension box coordinates
                if positive_annotations.shape[0] == 4:
                    positive_dict = dict()

                    positive_dict['rect'] = positive_annotations
                    positive_dict['image_id'] = idx
                    # positive_dict['image_name'] = sample_name

                    positive_list.append(positive_dict)
            else: # multi dimension box coordinates
                for positive_annotation in positive_annotations:
                    positive_dict = dict()

                    positive_dict['rect'] = positive_annotation
                    positive_dict['image_id'] = idx
                    # positive_dict['image_name'] = sample_name

                    positive_list.append(positive_dict)

            negative_annotation_path = os.path.join(root_dir, 'Annotations', sample_name + '_0.csv')
            negative_annotations = np.loadtxt(negative_annotation_path, dtype=np.int, delimiter=' ')
            # Consider that the csv file is empty or contains only a single dimension box
            if len(negative_annotations.shape) == 1:
                # single dimension box coordinates
                if negative_annotations.shape[0] == 4:
                    negative_dict = dict()

                    negative_dict['rect'] = negative_annotations
                    negative_dict['image_id'] = idx
                    # negative_dict['image_name'] = sample_name

                    negative_list.append(negative_dict)
            else:
                for negative_annotation in negative_annotations:
                    negative_dict = dict()

                    negative_dict['rect'] = negative_annotation
                    negative_dict['image_id'] = idx
                    # negative_dict['image_name'] = sample_name

                    negative_list.append(negative_dict)

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_list = positive_list
        self.negative_list = negative_list

# 그냥 드는 생각이지만 custom_finetune_dataset에서 정의한 방식과 지금 방식을 비교하면 보기 편하게 정의된 것은 후자의 방식이라고 생각한다. 전자의 방식으로 생각한 이유가 뭘까?

    def __getitem__(self, index: int):
        # Position the image to which the subscript belongs
        if index < len(self.positive_list):
            # a positive sample
            target = 1
            positive_dict = self.positive_list[index]

            xmin, ymin, xmax, ymax = positive_dict['rect']
            image_id = positive_dict['image_id']

            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
            cache_dict = positive_dict                        #Hard negative mining
        else:
            # negative sample
            target = 0
            idx = index - len(self.positive_list)                               # 솔직히 이 부분 이해가 잘 안간다. index가 positive 부터 차례대로 있어서 그 이후 negative sample이 존재하는 데이터라면 이해가 가지만
            negative_dict = self.negative_list[idx]                             # 그렇지 않고 index가 positive와 negative sample의 무작위 조합이라면 이 말은 이해가 되지 않는다.

            xmin, ymin, xmax, ymax = negative_dict['rect']
            image_id = negative_dict['image_id']

            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
            cache_dict = negative_dict                        #Hard negative mining

        # print('index: %d image_id: %d target: %d image.shape: %s [xmin, ymin, xmax, ymax]: [%d, %d, %d, %d]' %
        #       (index, image_id, target, str(image.shape), xmin, ymin, xmax, ymax))
        if self.transform:
            image = self.transform(image)

        return image, target, cache_dict

    def __len__(self) -> int:
        return len(self.positive_list) + len(self.negative_list)

    def get_transform(self):
        return self.transform

    def get_jpeg_images(self) -> list:
        return self.jpeg_images

    def get_positive_num(self) -> int:
        return len(self.positive_list)

    def get_negative_num(self) -> int:
        return len(self.negative_list)

    def get_positives(self) -> list:
        return self.positive_list

    def get_negatives(self) -> list:
        return self.negative_list

    # apply to hard negative mining
    # replacement negative sample
    def set_negative_list(self, negative_list):
        self.negative_list = negative_list


def test(idx):
    root_dir = '../../data/classifier_car/val'
    train_data_set = CustomClassifierDataset(root_dir)

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    # test id=3/66516/66517/530856
    image, target, cache_dict = train_data_set.__getitem__(idx)
    print('target: %d' % target)
    print('dict: ' + str(cache_dict))

    image = Image.fromarray(image) #이미지를 numpy 배열로 부터 읽어 들여서 받는다.
    print(image)
    print(type(image))

    # cv2.imshow('image', image)
    # cv2.waitKey(0)


def test2():
    root_dir = '../../data/classifier_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomClassifierDataset(root_dir, transform=transform)
    image, target, cache_dict = train_data_set.__getitem__(230856)
    print('target: %d' % target)
    print('dict: ' + str(cache_dict))
    print('image.shape: ' + str(image.shape))


def test3():
    root_dir = '../../data/classifier_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomClassifierDataset(root_dir, transform=transform)
    data_loader = DataLoader(train_data_set, batch_size=128, num_workers=8, drop_last=True)
    # drop_last hyperparameter는 batch_size가 5일 때, 마지막 batch_size가 2라면, batch_size 의존도가 높은 함수의 경우 last_batch를 생략할 수 있다.
    inputs, targets, cache_dicts = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    # test(159622)
    # test(4051)
    test(24768)
    # test2()
    # test3()