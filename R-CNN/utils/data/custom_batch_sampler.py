import numpy  as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.data.custom_finetune_dataset import CustomFinetuneDataset


class CustomBatchSampler(Sampler):

    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:
        """
        two-category data set
        For each batch, there are positive samples of batch_positive and negative samples of batch_negative.
        @param num_positive: Number of positive samples
        @param num_negative : Number of negative samples
        @param batch_positive: Number of positive samples in a single order
        @param batch_negative: Number of negative samples in a single order
        """
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        length = num_positive + num_negative
        self.idx_list = list(range(length))

        self.batch = batch_negative + batch_positive
        self.num_iter = length // self.batch

    def __iter__(self):
        sampler_list = list()
        for i in range(self.num_iter):
            tmp = np.concatenate(                                                           # np.concatenate axis 초기 값 axis = 0, 행 방향(위->아래)으로 붙이는 것이라고 생각해도 좋다.
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),     # random.sample(seq, num) -> num만큼의 무작위 개수를 seq에서 뽑는 것이라고 생각한다.
                 random.sample(self.idx_list[self.num_positive:], self.batch_negative))     # 이 코드들을 봐서 이야기 하건데, 0~num_positive 까지의 index는 positive_sample / num_positive ~ end 까지의 index는 negative_sample에 속한다
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)                                               # iter(sample_list)로 쓴 이유가 아마 iter하면서 데이터를 뽑아내는 것이라고 볼 수 있다.

    def __len__(self) -> int:
        return self.num_iter * self.batch                                       # 배치 개수 * 배치 크기

    def get_num_batch(self) -> int:
        return self.num_iter


def test():
    root_dir = '../../data/finetune_car/train'
    train_data_set = CustomFinetuneDataset(root_dir)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96) #positive sample 32개 negative sample 64개

    print('sampler len: %d' % train_sampler.__len__())
    print('sampler batch num: %d' % train_sampler.get_num_batch())

    first_idx_list = list(train_sampler.__iter__())[:128]
    print(first_idx_list)
    # Number of positive samples in a single batch
    print('positive batch: %d' % np.sum(np.array(first_idx_list) < 66517))


def test2():
    root_dir = '../../data/finetune_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96)
    data_loader = DataLoader(train_data_set, batch_size=128, sampler=train_sampler, num_workers=8, drop_last=True)

    inputs, targets = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    test()