import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.function as F
from torch.untils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 1000
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 1000
DEVICE_IDS = [0, 1, 2, 3]

# data directory 지정하기
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# checkpoint 경로 directory 만들기
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1).
                nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x


# def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
#   """
#   Args:
#       pretrained (bool): If True, returns a model pre-trained on ImageNet
#       progress (bool): If True, displays a progress bar of the download to stderr
#   """
#   model = AlexNet(**kwargs)
#   if pretrained:
#       state_dict = load_state_dict_from_url(model_urls['alexnet'],
#                                             progress=progress)
#       model.load_state_dict(state_dict)
#   return model
if __name__ == '__main__':
    # seed value 출력하기
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    tbwriter = SummaryWriter(log_dir=LOG_DIR)  # 주소지정을 통해 해당주소에 저장 후 생성
    print('TensorboardX summary writer created')
    # TensorboardX를 통해서 이미지나 음성 또는 비디오 또는 그래프를 그릴수 있다. 이런 걸 도와주는 툴이라고 생각한다.

    # model 생성하기
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # 다수의 GPU에서 train
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    print('AlexNet created')

    # dataset과 data loader 생성하기
    # Imagefolder를 쓸 때는 규칙이 있다. 데이터가 폴더에 담겨있는데 path -- cat -- data / path -- dog -- data 이런식으로
    # 구성된 (해당 주소에 data가 바로 있는 경우가 아닌 경우) 거라면 Imagefolder를 쓰면 한번에 dataset이 바로 만들어진다.
    # 이거 말고도 Datasetfolder가 있는데 이는 파일 확장자를 통해서 파일들을 다 불러온다.
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset created')
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,  # drop_last란 만약 배치의 길이가 다른 경우 (데이터 개수 27개중 batch_size가 5라면 마지막 batch 크기는 2)
        # 배치의 길이가 다른 경우 손실을 귀하기 귀찮고, 배치 의존도가 높은 경우 문제가 있으므로 마지막 배치를 사용하지 않도록 해주는 것
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    # optimizer 생성하기
    optimizer = optim.SGD(
        params=alexnet.parameters(),
        lr=LR_INIT,
        momentum=MOMENTUM,
        weight_decay=LR_DECAY)
    print('Optimizer created')

    # lr_scheduler로 LR 감소시키기 : 30epochs 마다 1/10
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created')

    # train 시작
    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            # loss 계산
            output = alexnet(imgs)
            loss = F.cross_entropy(output, classes)

            # parameter 갱신
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            # 정보를 기록하고 tensorboard에 추가하기
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                          .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

            # gradient values와 parameter average values 추력하기
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # parameters의 grad 출력하고 저장하기
                    # parameters values 출력하고 저장하기
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                                   parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                                   parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

            total_steps += 1

        # checkpoints 저장하기
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)
