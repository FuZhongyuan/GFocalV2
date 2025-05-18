import os
import cv2
import numpy as np
import jittor as jt
import jittor.dataset as jt_dataset
from pycocotools.coco import COCO
from jittor.dataset.utils import get_image_from_path, get_item_array
from utils.boxs_utils import xywh2xyxy
from PIL import Image


COCO_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
colors = [(67, 68, 113), (130, 45, 169), (2, 202, 130), (127, 111, 90), (92, 136, 113),
          (33, 250, 7), (238, 92, 104), (0, 151, 197), (134, 9, 145), (253, 181, 88),
          (246, 11, 137), (55, 72, 220), (136, 8, 253), (56, 73, 180), (85, 241, 53),
          (153, 207, 15), (187, 183, 180), (149, 32, 71), (92, 113, 184), (131, 7, 201),
          (56, 20, 219), (243, 201, 77), (13, 74, 96), (79, 14, 44), (195, 150, 66),
          (2, 249, 42), (195, 135, 43), (105, 70, 66), (120, 107, 116), (122, 241, 22),
          (17, 19, 179), (162, 185, 124), (31, 65, 117), (88, 200, 80), (232, 49, 154),
          (72, 1, 46), (59, 144, 187), (200, 193, 118), (123, 165, 219), (166, 42, 210),
          (203, 70, 77), (164, 25, 156), (87, 226, 159), (146, 110, 198), (178, 16, 239)]

default_aug_cfg = {
    'hsv_h': 0.014,
    'hsv_s': 0.68,
    'hsv_v': 0.36,
    'degree': 0.,
    'translate': 0.,
    'shear': 0.,
    'beta': (8, 8),
    'pad_val': 114,
}


class COCODataSets(jt_dataset.Dataset):
    def __init__(self, img_root, annotation_path,
                 img_size=640,
                 imageset='train2017',
                 augments=True,
                 aug_cfg=None,
                 use_mosaic=True,
                 mosaic_prob=0.5,
                 mixup_prob=0.15,
                 transform=None,
                 keep_ratio=True,
                 load_labels=True,
                 cache_images=False,
                 cache_labels=False,
                 ):
        super(COCODataSets, self).__init__()
        self.img_size = img_size
        self.imageset = imageset
        self.coco = COCO(annotation_path)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.annotations = self._load_coco_annotations()
        self.img_root = img_root
        self.augments = augments
        self.transform = transform
        self.aug_cfg = aug_cfg if aug_cfg is not None else default_aug_cfg
        self.use_mosaic = use_mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.keep_ratio = keep_ratio
        self.load_labels = load_labels
        self.cache_images = cache_images
        self.cache_labels = cache_labels
        self.batch_size = None

        if self.cache_labels:
            self.labels = self._cache_labels()

    def set_transform(self, transform):
        if self.augments:
            from utils.augmentations import SSDAugmentationTrain
            from utils.augmentations import YOLOV5Augmentation, MOSAIC
            if self.use_mosaic and self.imageset.find('train') != -1 and self.transform is None:
                self.transform = MOSAIC(img_size=self.img_size,
                                        usage_prob=self.mosaic_prob,
                                        Aug=YOLOV5Augmentation(degrees=self.aug_cfg['degree'],
                                                              translate=self.aug_cfg['translate'],
                                                              scale=[0.1, 2.0],
                                                              shear=self.aug_cfg['shear'],
                                                              perspective=0.0,
                                                              fill_color=self.aug_cfg['pad_val'],
                                                              hsv_h=self.aug_cfg['hsv_h'],
                                                              hsv_s=self.aug_cfg['hsv_s'],
                                                              hsv_v=self.aug_cfg['hsv_v']),
                                        img_root=self.img_root,
                                        imgsz=self.img_size,
                                        keep_ratio=self.keep_ratio,
                                        annos=self.annotations,
                                        mixup_prob=self.mixup_prob)
            elif self.imageset.find('train') != -1 and transform is None:
                self.transform = SSDAugmentationTrain(size=self.img_size,
                                                     mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225),
                                                     variance=(0.1, 0.2),
                                                     use_base=False)
        else:
            from utils.augmentations import SSDBaseTransform
            self.transform = SSDBaseTransform(self.img_size, mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        target = []
        for anno in annotations:
            if anno.get("ignore", False):
                continue
            x1 = anno["bbox"][0]
            y1 = anno["bbox"][1]
            width_box = anno["bbox"][2]
            height_box = anno["bbox"][3]
            if anno["area"] > 0 and width_box > 0 and height_box > 0:
                label_id = anno["category_id"]
                label_id_index = self.class_ids.index(label_id)
                target.append([x1, y1, x1 + width_box, y1 + height_box, label_id_index])
        return {'img_id': id_,
                'file_name': im_ann["file_name"],
                'width': width,
                'height': height,
                'annos': target}

    def _cache_labels(self):
        labels = []
        for i, data in enumerate(self.annotations):
            targets = data['annos']
            labels.append(targets)
        return labels

    def pull_item(self, index):
        id_ = self.ids[index]

        im_ann = self.coco.loadImgs(id_)[0]
        img_path = os.path.join(self.img_root, im_ann["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.annotations[index]
        width = im_ann["width"]
        height = im_ann["height"]
        target["orig_size"] = [width, height]
        target["img_size"] = [self.img_size, self.img_size]
        if self.transform is not None:
            if self.use_mosaic and self.imageset.find('train') != -1:
                img, target = self.transform(img, target, self.annotations, index)
            elif self.augments and self.imageset.find('train') != -1:
                img, boxes, labels = self.transform(img, target["annos"])
                target["annos"] = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, boxes, labels = self.transform(img, target["annos"])
                boxes = boxes * img.shape[0]
                target["annos"] = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        scale_target = np.zeros((len(target["annos"]), 5))
        for i, anno in enumerate(target["annos"]):
            scale_target[i, :4] = anno[:4] / self.img_size
            scale_target[i, 4] = anno[4]
        target["annos"] = scale_target
        return img, target

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img, target = self.pull_item(index)
        return img, target


def coco_collate(batch):
    """
    Args:
        batch: list of (image, target)
    """
    imgs, targets = [], []
    # 过滤空的数据
    for sample in batch:
        img, anno = sample
        if len(anno["annos"]) == 0:
            continue
        imgs.append(img)
        targets.append(anno)
    if not imgs:
        return [], []
    imgs = np.stack(imgs, axis=0)
    imgs = np.ascontiguousarray(imgs)
    imgs = jt.array(imgs).float32()
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs, targets 