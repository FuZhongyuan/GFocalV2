import os
import time
import jittor as jt
import yaml
import json

import cv2 as cv
import numpy as np
from tqdm import tqdm
from nets.retinanet import GFocal
from datasets.coco import coco_ids, rgb_mean, rgb_std
from utils.augmentations import RandScaleToMax
from utils.model_utils import AverageLogger


def coco_eavl(anno_path="/home/huffman/data/annotations/instances_val2017.json", pred_path="predicts.json"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoGt = COCO(anno_path)  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(pred_path)  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


@jt.no_grad()
def eval_model(weight_path="weights/gfocalV2_resnet50_best_map.pth"):
    from pycocotools.coco import COCO
    with open("config/gfocal.yaml", 'r') as rf:
        cfg = yaml.safe_load(rf)
    net = GFocal(**{**cfg['model'], 'pretrained': False, "nms_iou_thresh": 0.6})
    net.load_state_dict(jt.load(weight_path, map_location="cpu")['ema'])
    net.eval().half()
    data_cfg = cfg['data']
    basic_transform = RandScaleToMax(max_threshes=[data_cfg['max_thresh']])
    coco = COCO(data_cfg['val_annotation_path'])
    coco_predict_list = list()
    time_logger = AverageLogger()
    pbar = tqdm(coco.imgs.keys())
    for img_id in pbar:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(data_cfg['val_img_root'], file_name)
        img = cv.imread(img_path)
        h, w, _ = img.shape
        img, ratio, (left, top) = basic_transform.make_border(img,
                                                              max_thresh=data_cfg['max_thresh'],
                                                              border_val=(103, 116, 123))
        img_inp = (img[:, :, ::-1] / 255.0 - np.array(rgb_mean)) / np.array(rgb_std)
        img_inp = jt.from_numpy(img_inp).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float().half()
        tic = time.time()
        predict = net(img_inp)["predicts"][0]
        duration = time.time() - tic
        time_logger.update(duration)
        pbar.set_description("fps:{:4.2f}".format(1 / time_logger.avg()))
        if predict is None:
            continue
        predict[:, [0, 2]] = ((predict[:, [0, 2]] - left) / ratio).clamp(min=0, max=w)
        predict[:, [1, 3]] = ((predict[:, [1, 3]] - top) / ratio).clamp(min=0, max=h)
        box = predict.cpu().numpy()
        coco_box = box[:, :4]
        coco_box[:, 2:] = coco_box[:, 2:] - coco_box[:, :2]
        for p, b in zip(box.tolist(), coco_box.tolist()):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': coco_ids[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})
    with open("predicts.json", 'w') as file:
        json.dump(coco_predict_list, file)
    coco_eavl(anno_path=data_cfg['val_annotation_path'], pred_path="predicts.json")


if __name__ == '__main__':
    eval_model()
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.398
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.426
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.213
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.439
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.507
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.549
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.344
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.606
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
