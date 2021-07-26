import time
import shutil
import numpy as np
import cv2
import os
import selectivesearch
from utils.util import check_dir
from utils.util import parse_car_csv
from utils.util import parse_xml
from utils.util import compute_ious


# train
# positive num: 66517
# negatie num: 464340
# val
# positive num: 64712
# negative num: 415134

def parse_annotation_jpeg(annotation_path, jped_path, gs):
    # read the image
    img = cv2.imread(jpeg_path)
    # get_rects => return x,y,x+w,y+h
    selectivesearch.get_rects(gs)  # region proposals
    # annotation_path는 PASCAL VOC 계층구조 하이퍼링크를 참조해주세요
    bndboxes = parse_xml(annotation_path)  # ground truth boxes

    # get size of the biggest bounding box
    maximum_bndbox_size = 0
    for bndbox in bndboxes:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    # Comparing all region proposals and ground truth
    # return a list of iou results for each region proposals
    iou_list = compute_ious(rects, bndboxs)

    positive_list = list()
    negative_list = list()

    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]

        # When fine-tuning the pre-trained CNN model
        # if iou_score >= 0.5 -> consider as positive example
        if iou_score >= 0.5:
            positive_list.append(rects[i])
        # if iou_score < 0.5 and rectangle size is 20% of maximum_bounding box size -> consider as negative example
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            negative_list.append(rects[i])
        else:
            pass
    return positive_list, negative_list


if __name__ == '__main__':
    car_root_dir = '../../data/voc_car/'  # car data in pascal voc data directory
    finetune_root_dir = '../../data/finetune_car/'  # fine_tuned car data in pascal voc data directory
    check_dir(finetune_root_dir)  # if directory is not exist -> make directory

    gs = selectivesearch.get_selective_search()  # create selective search segmentation object
    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)  # => '../../data/voc_car/train' or '../../data/voc_car/val'
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')  # => '../../data/voc_car/train/Annotations'
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')  # => '../../data/voc_car/train/JPEGImages'

        dst_root_dir = os.path.join(finetune_root_dir,
                                    name)  # => '../../data/finetune_car/train' or '../../data/finetune_car/val'
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')  # => '../../data/finetune_car/train/Annotations'
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')  # => '../../data/finetune_car/train/JPEGImages'

        # if directory is not exist -> make directory
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0
        # After '../../data/voc_car/train' + 'car.csv', execute np.loadtxt('../../data/voc_car/train/car.csv')
        # (해당 directory에서 파일을 불러와 이를 numpy 배열 형태로 저장하는 기능이다)
        samples = parse_car_csv(src_root_dir)

        src_csv_path = os.path.join(src_root_dir, 'car.csv')  # '../../data/voc_car/train' + 'car.csv'
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')  # '../../data/finetune_car/train' + 'car.csv'
        shutil.copyfile(src_csv_path, dst_csv_path)  # (dst_csv_path에 담긴 파일명을 가진 src_csv_path안의 데이터를 복사한 복사본을 만든다.)

        for sample_name in samples:
            since = time.time()  # a function that tells how long has elapsed

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')

            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')

            shutil.copyfile(src_jpeg_path,
                            dst_jpeg_path)  # (dst_jpeg_path에 담긴 파일명을 가진 src_jpeg_path안의 데이터를 복사한 복사본을 만든다.)

            # save the numpy.array file to the corresponding path
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')

            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        print('%s positive num: %d' % (name, total_num_positive))
        print('%s negative num: %d' % (name, total_num_negative))
    print('done')