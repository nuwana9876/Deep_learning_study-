import numpy  as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.util import parse_car_csv


class CustomFinetuneDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        samples = parse_car_csv(root_dir)

        # load all car images
        jpeg_images = [cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg")) for sample_name in
                       samples]

        # positive : iou >= 0.5
        positive_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_1.csv') for sample_name in
                                samples]
        # negative : iou < 0.5
        negative_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_0.csv') for sample_name in
                                samples]

        # Bounding box size
        positive_sizes = list()
        negative_sizes = list()
        # bounding box coordinates
        positive_rects = list()
        negative_rects = list()

        # positive_rects = [(x, y, w, h), ....]
        # positive_sizes = [1, .....]
        for annotation_path in positive_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            # Existence file is empty or only one line of data exists in the file
            if len(rects.shape) == 1:
                # Single line
                if rects.shape[0] == 4:
                    positive_rects.append(rects)
                    positive_sizes.append(
                        1)  # 여기가 문제구만. positive_size에 1을 넣어주는 것. negative_annotations의 경우에는 negative_size에 1을 추가하는 것을 볼 수 있다.
                else:  # 코드를 봤을 때 , positive_annotations와 negative_annotation 모두 rects.shape가 1일 때의 else문이 전부 positive_sizes.append(0)이라는 것을 볼 수 있는데
                    positive_sizes.append(
                        0)  # index를 고려하면 negative_annotations의 경우는 negative_sizes.append(0)가 맞다고 생각하는데 무슨 이유일까?
            else:
                # 오 새로운 발견! 즉 rects.shape가 1이 아니란 것은 이미지 1개에서 찾은 1종류의 객체가 여러개라는 의미로 볼 수 있다는 걸까?
                # 만약 3개면, bounding box도 3개가 들어왔을 것이고, size도 3개가 될 것이다.
                positive_rects.extend(rects)
                positive_sizes.append(len(rects))
        for annotation_path in negative_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            # It's the same as the regular sample rule.
            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    negative_rects.append(rects)
                    negative_sizes.append(1)
                else:
                    positive_sizes.append(0)
            else:
                negative_rects.extend(rects)
                negative_sizes.append(len(rects))

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_sizes = positive_sizes
        self.negative_sizes = negative_sizes
        self.positive_rects = positive_rects
        self.negative_rects = negative_rects
        self.total_positive_num = int(np.sum(positive_sizes))
        self.total_negative_num = int(np.sum(negative_sizes))

    def __getitem__(self, index: int):
        # Position the image to which the subscript belongs
        image_id = len(self.jpeg_images) - 1  # 0 ~ number of images - 1
        if index < self.total_positive_num:  # 사실 말하자면 index < self.total_positive_num 의 조건이 이해가 되지 않는다. 이 조건이 positive sample이라는 것을 나타내는 이유가 뭘까?
            # a positive sample                                                          # 이해한대로 설명하면, init에서 positive_annotations를 먼저 positive_sizes에 넣어주고 이후 negative_annotations를 나중에 positive_sizes에 넣어주기 때문에
            target = 1  # index를 total_positive_num과 비교했을 때, 작으면 positive 영역에 있는 것이고 , 크면 negative_sample에서 len(rects.shape)가 1일 때의 else 부분을 의미한다고 생각
            xmin, ymin, xmax, ymax = self.positive_rects[index]
            # Find an image to belong to
            for i in range(len(self.positive_sizes) - 1):
                if np.sum(self.positive_sizes[:i]) <= index < np.sum(self.positive_sizes[:(i + 1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        else:
            # negative sample
            target = 0
            idx = index - self.total_positive_num
            xmin, ymin, xmax, ymax = self.negative_rects[idx]
            # Find an image to belong to
            for i in range(len(self.negative_sizes) - 1):
                if np.sum(self.negative_sizes[:i]) <= idx < np.sum(self.negative_sizes[:(i + 1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]

        # print('index: %d image_id: %d target: %d image.shape: %s [xmin, ymin, xmax, ymax]: [%d, %d, %d, %d]' %
        #       (index, image_id, target, str(image.shape), xmin, ymin, xmax, ymax))
        if self.transform:
            image = self.transform(image)

        return image, target  # index에 맞는 image와 target(0인지 1인지)룰 출력한다

    def __len__(self) -> int:
        return self.total_positive_num + self.total_negative_num  # 전체 길이

    def get_positive_num(self) -> int:
        return self.total_positive_num

    def get_negative_num(self) -> int:
        return self.total_negative_num


def test(idx):
    root_dir = '../../data/finetune_car/train'
    train_data_set = CustomFinetuneDataset(root_dir)

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    # test id=3/66516/66517/530856
    image, target = train_data_set.__getitem__(idx)
    print('target: %d' % target)

    image = Image.fromarray(image)
    print(image)
    print(type(image))

    # cv2.imshow('image', image)
    # cv2.waitKey(0)


def test2():
    root_dir = '../../data/finetune_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    image, target = train_data_set.__getitem__(530856)
    print('target: %d' % target)
    print('image.shape: ' + str(image.shape))


def test3():
    root_dir = '../../data/finetune_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    data_loader = DataLoader(train_data_set, batch_size=128, num_workers=8, drop_last=True)

    inputs, targets = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    # test(159622)
    # test(4051)
    test(24768)