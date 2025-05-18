import math
import random
import cv2 as cv
import numpy as np
from copy import deepcopy

cv.setNumThreads(0)


class BoxInfo(object):
    def __init__(self, img_path, boxes=None, labels=None, weights=None, padding_val=(103, 116, 123)):
        super(BoxInfo, self).__init__()
        self.img_path = img_path
        self.img = None
        self.boxes = boxes
        self.labels = labels
        self.weights = weights
        self.padding_val = padding_val

    def revise(self):
        if self.boxes is None:
            self.boxes = np.zeros(shape=(0, 4))
        if self.labels is None:
            self.labels = np.zeros(shape=(0,))
        if self.weights is None:
            self.weights = np.ones(shape=(0,))

    def clone(self):
        return deepcopy(self)

    def load_img(self):
        self.img = cv.imread(self.img_path)
        return self

    def draw_box(self, colors, names):
        if self.img is None:
            return None
        ret_img = self.img.copy()
        if self.boxes is None or len(self.boxes) == 0:
            return ret_img

        if self.weights is None:
            self.weights = np.ones_like(self.labels)

        for label_idx, weight, (x1, y1, x2, y2) in zip(self.labels, self.weights, self.boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(ret_img, (x1, y1), (x2, y2), color=colors[int(label_idx)], thickness=2)
            cv.putText(ret_img, "{:s}".format(names[int(label_idx)]),
                       (x1, y1 + 5),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       colors[int(label_idx)], 2)
            cv.putText(ret_img, "{:.2f}".format(weight),
                       (x1, y1 + 15),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       colors[int(label_idx)], 1)
        return ret_img


class BasicTransform(object):
    def __init__(self, p=0.5):
        self.p = p

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        pass

    def __call__(self, box_info: BoxInfo) -> BoxInfo:
        assert box_info.img is not None, "please load in img first"
        aug_p = np.random.uniform()
        if aug_p <= self.p:
            box_info = self.aug(box_info)
        return box_info

    def reset(self, **settings):
        p = settings.get('p', None)
        if p is not None:
            self.p = p
        return self


class Identity(BasicTransform):
    def __init__(self, **kwargs):
        kwargs['p'] = 1.0
        super(Identity, self).__init__(**kwargs)

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        return box_info


class RandNoise(BasicTransform):
    def __init__(self, **kwargs):
        kwargs['p'] = 1.0
        super(RandNoise, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img):
        mu = 0
        pre_type = img.dtype
        sigma = np.random.uniform(1, 15)
        ret_img = img + np.random.normal(mu, sigma, img.shape)
        ret_img = ret_img.clip(0., 255.).astype(pre_type)
        return ret_img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class RandBlur(BasicTransform):
    """
    随机进行模糊
    """

    def __init__(self, **kwargs):
        kwargs['p'] = 1.0
        super(RandBlur, self).__init__(**kwargs)

    @staticmethod
    def gaussian_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img

    @staticmethod
    def median_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.medianBlur(img, kernel_size, 0)
        return img

    @staticmethod
    def blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.blur(img, (kernel_size, kernel_size))
        return img

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        aug_blur = np.random.choice([self.gaussian_blur, self.median_blur, self.blur])
        img = aug_blur(img)
        return img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class RandHSV(BasicTransform):
    """
    color jitter
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, **kwargs):
        kwargs['p'] = 1.0
        super(RandHSV, self).__init__(**kwargs)
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
        ret_img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        return ret_img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class RandScaleToMax(BasicTransform):
    def __init__(self, max_threshes,
                 pad_to_square=True,
                 minimum_rectangle=False,
                 scale_up=True,
                 division=64,
                 **kwargs):
        kwargs['p'] = 1.0
        super(RandScaleToMax, self).__init__(**kwargs)
        assert isinstance(max_threshes, list)
        self.max_threshes = max_threshes
        self.pad_to_square = pad_to_square
        self.minimum_rectangle = minimum_rectangle
        self.scale_up = scale_up
        self.division = division

    def make_border(self, img: np.ndarray, max_thresh, border_val):
        h, w = img.shape[:2]
        r = min(max_thresh / h, max_thresh / w)
        if not self.scale_up:
            r = min(r, 1.0)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        if r != 1.0:
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        if not self.pad_to_square:
            return img, r, (0, 0)
        dw, dh = int(math.ceil((max_thresh - new_w) / self.division) * self.division), \
            int(math.ceil((max_thresh - new_h) / self.division) * self.division)
        if self.minimum_rectangle:
            dw, dh = 0, 0
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = cv.copyMakeBorder(img, top, bottom, left, right, borderType=cv.BORDER_CONSTANT, value=border_val)
        return img, r, (left, top)

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        max_thresh = np.random.choice(self.max_threshes)
        img, r, (left, top) = self.make_border(box_info.img, max_thresh, box_info.padding_val)
        new_boxes = box_info.boxes.copy()
        if len(new_boxes) > 0:
            new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]] * r + left
            new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]] * r + top
        box_info.img = img
        box_info.boxes = new_boxes
        return box_info


class RandScaleMinMax(BasicTransform):
    def __init__(self, min_threshes, max_thresh=1024, **kwargs):
        kwargs['p'] = 1.0
        super(RandScaleMinMax, self).__init__(**kwargs)
        assert isinstance(min_threshes, list)
        self.min_threshes = min_threshes
        self.max_thresh = max_thresh

    def scale_img(self, img: np.ndarray, min_thresh):
        h, w = img.shape[:2]
        r = min(min_thresh / min(h, w), self.max_thresh / max(h, w))
        if r != 1:
            new_w, new_h = int(round(w * r)), int(round(h * r))
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        return img, r

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        min_thresh = np.random.choice(self.min_threshes)
        img, r = self.scale_img(box_info.img, min_thresh)
        box_info.img = img
        if len(box_info.boxes) > 0:
            box_info.boxes[:, :] = box_info.boxes[:, :] * r
        return box_info


class LRFlip(BasicTransform):
    """
    左右翻转
    """

    def __init__(self, **kwargs):
        kwargs['p'] = 0.5
        super(LRFlip, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img: np.ndarray) -> np.ndarray:
        return np.fliplr(img)

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        h, w = box_info.img.shape[:2]
        box_info.img = self.img_aug(box_info.img)
        if len(box_info.boxes) > 0:
            box_info.boxes[:, [0, 2]] = w - box_info.boxes[:, [2, 0]]
        return box_info


class UDFlip(BasicTransform):
    """
    上下翻转
    """

    def __init__(self, **kwargs):
        kwargs['p'] = 0.5
        super(UDFlip, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img: np.ndarray) -> np.ndarray:
        return np.flipud(img)

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        h, w = box_info.img.shape[:2]
        box_info.img = self.img_aug(box_info.img)
        if len(box_info.boxes) > 0:
            box_info.boxes[:, [1, 3]] = h - box_info.boxes[:, [3, 1]]
        return box_info


class RandPerspective(BasicTransform):
    def __init__(self,
                 target_size=None,
                 degree=(0, 0),
                 translate=0,
                 scale=(1.0, 1.0),
                 shear=0,
                 perspective=0.0,
                 **kwargs):
        kwargs['p'] = 1.0
        super(RandPerspective, self).__init__(**kwargs)
        self.target_size = target_size
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def reset(self, **settings):
        p = settings.get('p', None)
        degree = settings.get('degree', None)
        scale = settings.get('scale', None)
        perspective = settings.get('perspective', None)
        shear = settings.get('shear', None)
        translate = settings.get('translate', None)
        target_size = settings.get('target_size', None)
        if p is not None:
            self.p = p
        if degree is not None:
            self.degree = degree
        if scale is not None:
            self.scale = scale
        if perspective is not None:
            self.perspective = perspective
        if shear is not None:
            self.shear = shear
        if translate is not None:
            self.translate = translate
        if target_size is not None:
            self.target_size = target_size
        return self

    def get_transform_matrix(self, img):
        img_h, img_w = img.shape[:2]

        # Center
        C = np.eye(3)
        C[0, 2] = -img_w / 2  # x translation (pixels)
        C[1, 2] = -img_h / 2  # y translation (pixels)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degree[0], self.degree[1])
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(self.scale[0], self.scale[1])
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * img_w  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * img_h  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
        return M

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        M = self.get_transform_matrix(box_info.img)
        h, w = box_info.img.shape[:2]
        if self.target_size is not None:
            new_h, new_w = self.target_size, self.target_size
        else:
            new_h, new_w = h, w
        img = cv.warpAffine(box_info.img, M[:2], dsize=(new_w, new_h), borderValue=box_info.padding_val)
        nb = len(box_info.boxes)
        box_info.img = img
        if nb > 0:
            xy = np.ones((nb * 4, 3))
            xy[:, :2] = box_info.boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(nb * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = xy[:, :2].reshape(nb, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, nb).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, new_w)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, new_h)
            box_info.boxes = new
        return box_info
        
class Mosaic(BasicTransform):
    def __init__(self,
                 candidate_box_info,
                 color_gitter=None,
                 target_size=640,
                 rand_center=True,
                 translate=0.1,
                 scale=(0.5, 1.5),
                 degree=(0, 0),
                 **kwargs):
        kwargs['p'] = 1.0
        super(Mosaic, self).__init__(**kwargs)
        self.candidate_box_info = candidate_box_info
        self.target_size = target_size
        self.color_gitter = color_gitter
        self.rand_center = rand_center
        self.translate = translate
        self.scale = scale
        self.degree = degree
        self.rand_perspective = RandPerspective(
            target_size=None,
            degree=degree,
            translate=translate,
            scale=scale,
            shear=0,
            perspective=0.0,
        )

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        assert box_info.img is not None, 'please load img first'
        assert len(self.candidate_box_info) >= 3, 'please add enough image to self.candidate_box_info'
        idx = np.random.choice(list(range(len(self.candidate_box_info))), 3, replace=False)
        mosaic_imgs = []
        for idx_i in idx:
            mosaic_imgs.append(self.candidate_box_info[idx_i].clone().load_img())
        mosaic_imgs.append(box_info)
        if self.color_gitter is not None:
            mosaic_imgs = [self.color_gitter(item) for item in mosaic_imgs]
        img_h, img_w = box_info.img.shape[:2]
        target_h, target_w = self.target_size, self.target_size
        if self.rand_center:
            xc = int(random.uniform(img_w // 2, target_w - img_w // 2))
            yc = int(random.uniform(img_h // 2, target_h - img_h // 2))
        else:
            xc, yc = [self.target_size // 2] * 2
        result_img = np.full([self.target_size, self.target_size, 3], 114, dtype=np.uint8)
        result_boxes = []
        result_labels = []
        result_weights = []
        for i, item in enumerate(mosaic_imgs):
            boxes = item.boxes
            labels = item.labels
            weights = item.weights
            img = item.img
            h, w = img.shape[:2]
            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.target_size), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.target_size, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.target_size), min(self.target_size, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            else:
                assert False
            result_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            offset_w = x1a - x1b
            offset_h = y1a - y1b
            if len(boxes) > 0:
                tmp_boxes = boxes.copy()
                tmp_boxes[:, [0, 2]] = tmp_boxes[:, [0, 2]] + offset_w
                tmp_boxes[:, [1, 3]] = tmp_boxes[:, [1, 3]] + offset_h
                result_boxes.append(tmp_boxes)
                result_labels.append(labels)
                result_weights.append(weights)
        box_info.img = result_img
        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, axis=0)
            result_labels = np.concatenate(result_labels, axis=0)
            result_weights = np.concatenate(result_weights, axis=0)
            new_boxes = []
            new_labels = []
            new_weights = []
            for box, label, weight in zip(result_boxes, result_labels, result_weights):
                xmin, ymin, xmax, ymax = box.tolist()
                if xmin > xmax or ymin > ymax or xmax <= 0 or ymax <= 0 or xmin >= self.target_size or ymin >= self.target_size:
                    continue
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(self.target_size, xmax)
                ymax = min(self.target_size, ymax)
                if (xmax - xmin) < 2 or (ymax - ymin) < 2:
                    continue
                new_boxes.append([xmin, ymin, xmax, ymax])
                new_labels.append(label)
                new_weights.append(weight)
            if len(new_boxes) > 0:
                box_info.boxes = np.array(new_boxes, dtype=np.float32)
                box_info.labels = np.array(new_labels, dtype=np.float32)
                box_info.weights = np.array(new_weights, dtype=np.float32)
            else:
                box_info.boxes = np.zeros((0, 4), dtype=np.float32)
                box_info.labels = np.zeros((0,), dtype=np.float32)
                box_info.weights = np.ones((0,), dtype=np.float32)
        if self.target_size != self.target_size:
            box_info = self.rand_perspective(box_info)
        return box_info


class MosaicWrapper(Mosaic):
    def __init__(self, sizes, **kwargs):
        super(MosaicWrapper, self).__init__(**kwargs)
        self.sizes = sizes

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        target_size = np.random.choice(self.sizes)
        self.target_size = target_size
        return super(MosaicWrapper, self).aug(box_info)


class MixUp(BasicTransform):
    def __init__(self,
                 candidate_box_info,
                 color_gitter=None,
                 mix_ratio=0.5, **kwargs):
        kwargs['p'] = 1.0
        super(MixUp, self).__init__(**kwargs)
        self.candidate_box_info = candidate_box_info
        self.color_gitter = color_gitter
        self.mix_ratio = mix_ratio

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        assert box_info.img is not None, 'please load img first'
        assert len(self.candidate_box_info) >= 1, 'please add enough image to self.candidate_box_info'

        idx = np.random.choice(list(range(len(self.candidate_box_info))), 1)[0]
        tmp_box_info = self.candidate_box_info[idx].clone().load_img()
        # 确保目标图像的尺寸一致
        target_h, target_w = box_info.img.shape[:2]
        tmp_box_info.img = cv.resize(tmp_box_info.img, dsize=(target_w, target_h))

        if len(tmp_box_info.boxes) > 0:
            img_h, img_w = tmp_box_info.img.shape[:2]
            r_w, r_h = target_w / img_w, target_h / img_h
            tmp_box_info.boxes[:, [0, 2]] = tmp_box_info.boxes[:, [0, 2]] * r_w
            tmp_box_info.boxes[:, [1, 3]] = tmp_box_info.boxes[:, [1, 3]] * r_h

        r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        box_info.img = (box_info.img * r + tmp_box_info.img * (1 - r)).astype(np.uint8)
        c1_boxes = box_info.boxes.copy()
        c1_labels = box_info.labels.copy()
        c1_weights = box_info.weights.copy() * r
        if len(tmp_box_info.boxes) > 0:
            c2_boxes = tmp_box_info.boxes.copy()
            c2_labels = tmp_box_info.labels.copy()
            c2_weights = tmp_box_info.weights.copy() * (1 - r)
            box_info.boxes = np.concatenate([c1_boxes, c2_boxes], axis=0)
            box_info.labels = np.concatenate([c1_labels, c2_labels], axis=0)
            box_info.weights = np.concatenate([c1_weights, c2_weights], axis=0)
        return box_info


class MixUpWrapper(MixUp):
    def __init__(self, beta=(8, 8), **kwargs):
        super(MixUpWrapper, self).__init__(**kwargs)
        self.beta = beta

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        if self.color_gitter is not None:
            box_info = self.color_gitter(box_info)
        return super(MixUpWrapper, self).aug(box_info)


class RandCrop(BasicTransform):
    def __init__(self, min_thresh=0.5, max_thresh=0.8, **kwargs):
        kwargs['p'] = 0.5
        super(RandCrop, self).__init__(**kwargs)
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh

    def get_crop_area(self, h, w):
        crop_h = int(h * np.random.uniform(self.min_thresh, self.max_thresh))
        crop_w = int(w * np.random.uniform(self.min_thresh, self.max_thresh))
        start_h = int(np.random.uniform(0, h - crop_h))
        start_w = int(np.random.uniform(0, w - crop_w))
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        return start_h, start_w, end_h, end_w

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        img = box_info.img
        boxes = box_info.boxes
        labels = box_info.labels
        weights = box_info.weights
        h, w = img.shape[:2]
        start_h, start_w, end_h, end_w = self.get_crop_area(h, w)
        img = img[start_h:end_h, start_w:end_w, :]
        # conform the boxes
        # x1y1x2y2
        new_boxes = []
        new_labels = []
        new_weights = []
        for box, label, weight in zip(boxes, labels, weights):
            xmin, ymin, xmax, ymax = box.tolist()
            xmin -= start_w
            xmax -= start_w
            ymin -= start_h
            ymax -= start_h
            if xmin > xmax or ymin > ymax or xmax <= 0 or ymax <= 0 or xmin >= (end_w - start_w) or ymin >= (
                    end_h - start_h):
                continue
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(end_w - start_w, xmax)
            ymax = min(end_h - start_h, ymax)
            new_boxes.append([xmin, ymin, xmax, ymax])
            new_labels.append(label)
            new_weights.append(weight)
        if len(new_boxes) > 0:
            box_info.boxes = np.array(new_boxes, dtype=np.float32)
            box_info.labels = np.array(new_labels, dtype=np.float32)
            box_info.weights = np.array(new_weights, dtype=np.float32)
        box_info.img = img
        return box_info


class OneOf(BasicTransform):
    def __init__(self, transforms, **kwargs):
        kwargs['p'] = 1.0
        super(OneOf, self).__init__(**kwargs)
        if isinstance(transforms[0], tuple):
            # 套路1
            transform_probs = [t[0] for t in transforms]
            transforms_sum = sum(transform_probs)
            self.transforms_probs = [t / transforms_sum for t in transform_probs]
            self.transforms = [t[1] for t in transforms]
        else:
            # 套路2
            self.transforms = transforms
            self.transforms_probs = [1 / len(transforms) for _ in range(len(transforms))]

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        select_ind = np.random.choice(list(range(len(self.transforms))),
                                      p=self.transforms_probs)
        select_transform = self.transforms[select_ind]
        box_info = select_transform(box_info)
        return box_info


class Compose(BasicTransform):
    def __init__(self, transforms, **kwargs):
        kwargs['p'] = 1.0
        super(Compose, self).__init__(**kwargs)
        self.transforms = transforms

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        for t in self.transforms:
            box_info = t(box_info)
        return box_info 