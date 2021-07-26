import cv2
import numpy as np
from torchvision.datasets import VOCDetection

if __name__ == '__main__':
    """
    Download Pascal VOC Dataset
    """
    dataset = VOCDetection('../../data', year='2007', image_set='trainval', download=True)

    img, target = dataset.__getitem__(1000)
    img = np.array(img)

    print(target)
    print(img.shape)

    cv2.imshow('img', img)
    cv2.waitKey(0)