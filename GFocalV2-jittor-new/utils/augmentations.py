import math
import random
import cv2 as cv
import numpy as np
import jittor as jt
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
        dw, dh = int(math.ceil(max_thresh - new_w)), int(math.ceil(max_thresh - new_h))
        if self.minimum_rectangle:
            dw, dh = 0, 0
            max_thresh = max(new_w, new_h)
        if self.division and not self.minimum_rectangle:
            dw = int(dw)
            dh = int(dh)
            max_thresh = math.ceil(max_thresh / self.division) * self.division
        top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
        left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=border_val)  # add border
        dxy = (left, top)
        return img, r, dxy

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        max_thresh = np.random.choice(self.max_threshes)
        ret_box_info = box_info.clone()
        ret_box_info.revise()
        if ret_box_info.img is None:
            raise ValueError("img is None, plz call load_img first!")
        img, r, (dx, dy) = self.make_border(ret_box_info.img, max_thresh, ret_box_info.padding_val)
        ret_box_info.img = img
        if len(box_info.boxes) == 0:
            return ret_box_info
        boxes = ret_box_info.boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + dy
        ret_box_info.boxes = boxes
        return ret_box_info

    def reset(self, **settings):
        p = settings.get('p', None)
        max_threshes = settings.get('max_threshes', None)
        if p is not None:
            self.p = p
        if max_threshes is not None:
            self.max_threshes = max_threshes
        return self


class RandScaleMinMax(BasicTransform):
    def __init__(self, min_threshes, max_thresh=1024, **kwargs):
        kwargs['p'] = 1.0
        super(RandScaleMinMax, self).__init__(**kwargs)
        self.min_threshes = min_threshes
        self.max_thresh = max_thresh

    def scale_img(self, img: np.ndarray, min_thresh):
        h, w = img.shape[:2]
        r = min_thresh / min(h, w)
        if h < w:
            h_new, w_new = min_thresh, int(w * r)
        else:
            h_new, w_new = int(h * r), min_thresh
        if w_new > self.max_thresh:
            w_new = self.max_thresh
        if h_new > self.max_thresh:
            h_new = self.max_thresh
        img = cv.resize(img, (w_new, h_new), interpolation=cv.INTER_LINEAR)
        return img, (h_new, w_new), r

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        min_thresh = np.random.choice(self.min_threshes)
        ret_box_info = box_info.clone()
        ret_box_info.revise()
        if ret_box_info.img is None:
            raise ValueError("img is None, plz call load_img first!")
        img, (h_new, w_new), r = self.scale_img(ret_box_info.img, min_thresh)
        ret_box_info.img = img
        if len(ret_box_info.boxes) == 0:
            return ret_box_info
        boxes = ret_box_info.boxes.copy()
        boxes = boxes * r
        ret_box_info.boxes = boxes
        return ret_box_info


class LRFlip(BasicTransform):
    """
    对图片进行左右翻转
    """

    def __init__(self, **kwargs):
        super(LRFlip, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img: np.ndarray) -> np.ndarray:
        img = np.ascontiguousarray(img[:, ::-1, :])
        return img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        ret_box_info = box_info.clone()
        ret_box_info.revise()
        img = ret_box_info.img
        img = self.img_aug(img)
        ret_box_info.img = img
        if len(ret_box_info.boxes) == 0:
            return ret_box_info
        boxes = ret_box_info.boxes.copy()
        h, w = ret_box_info.img.shape[:2]
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        ret_box_info.boxes = boxes
        return ret_box_info


class UDFlip(BasicTransform):
    """
    对图片进行上下翻转
    """

    def __init__(self, **kwargs):
        super(UDFlip, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img: np.ndarray) -> np.ndarray:
        img = np.ascontiguousarray(img[::-1, :, :])
        return img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        ret_box_info = box_info.clone()
        ret_box_info.revise()
        img = ret_box_info.img
        img = self.img_aug(img)
        ret_box_info.img = img
        if len(ret_box_info.boxes) == 0:
            return ret_box_info
        boxes = ret_box_info.boxes.copy()
        h, w = ret_box_info.img.shape[:2]
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        ret_box_info.boxes = boxes
        return ret_box_info


class RandPerspective(BasicTransform):
    def __init__(self,
                 target_size=None,
                 degree=(0, 0),
                 translate=0,
                 scale=(1.0, 1.0),
                 shear=0,
                 perspective=0.0,
                 **kwargs):
        super(RandPerspective, self).__init__(**kwargs)
        self.target_size = target_size
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def reset(self, **settings):
        p = settings.get('p', None)
        target_size = settings.get('target_size', None)
        degree = settings.get('degree', None)
        translate = settings.get('translate', None)
        scale = settings.get('scale', None)
        shear = settings.get('shear', None)
        perspective = settings.get('perspective', None)
        if p is not None:
            self.p = p
        if target_size is not None:
            self.target_size = target_size
        if degree is not None:
            self.degree = degree
        if translate is not None:
            self.translate = translate
        if scale is not None:
            self.scale = scale
        if shear is not None:
            self.shear = shear
        if perspective is not None:
            self.perspective = perspective
        return self

    def get_transform_matrix(self, img):
        h, w = img.shape[:2]
        if self.target_size is not None:
            h, w = self.target_size

        # Center
        C = np.eye(3)
        C[0, 2] = -w / 2  # x translation (pixels)
        C[1, 2] = -h / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degree, self.degree)
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
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * w  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * h  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        return M, (h, w)

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        ret_box_info = box_info.clone()
        ret_box_info.revise()
        if ret_box_info.img is None:
            raise ValueError("img is None, plz call load_img first!")
        img = ret_box_info.img
        M, (h, w) = self.get_transform_matrix(img)
        img = cv.warpPerspective(img, M, dsize=(w, h), flags=cv.INTER_LINEAR, borderValue=ret_box_info.padding_val)
        ret_box_info.img = img
        n = len(ret_box_info.boxes)
        if n == 0:
            return ret_box_info

        xy = np.ones((n * 4, 3))
        xy[:, :2] = ret_box_info.boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new_boxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, w)
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, h)

        # filter candidates
        i = self.box_candidates(box1=ret_box_info.boxes.T, box2=new_boxes.T)
        ret_box_info.boxes = new_boxes[i]
        ret_box_info.labels = ret_box_info.labels[i]
        ret_box_info.weights = ret_box_info.weights[i]

        return ret_box_info

    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        """
        计算box的长宽比，面积比,保留合适的框
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


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
        """
        :param candidate_box_info:
        :param target_size:
        :param rand_center: 中心点是否随机
        :param translate:
        :param scale:
        :param degree:
        :param kwargs:
        """
        super(Mosaic, self).__init__(**kwargs)
        self.candidate_box_info = candidate_box_info
        self.color_gitter = color_gitter
        self.target_size = target_size
        self.rand_center = rand_center
        self.translate = translate
        self.scale = scale
        self.degree = degree

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        ret_box_info = box_info.clone()
        ret_box_info.revise()
        if ret_box_info.img is None:
            raise ValueError("img is None, plz call load_img first!")
        target_size = self.target_size
        if self.rand_center:
            yc, xc = [int(random.uniform(0.25 * target_size, 0.75 * target_size)) for _ in range(2)]  # mosaic center
        else:
            yc, xc = [int(round(0.5 * target_size)) for _ in range(2)]
        indices = [random.randint(0, len(self.candidate_box_info) - 1) for _ in range(3)]
        boxes_hw_list = [ret_box_info.clone()]
        for index in indices:
            boxes_hw_list.append(self.candidate_box_info[index].clone().load_img())

        random.shuffle(boxes_hw_list)
        result_img = np.zeros([target_size, target_size, 3], dtype=np.uint8)
        result_img[:, :, :] = ret_box_info.padding_val
        result_boxes, result_labels, result_weights = [], [], []
        half_w, half_h = int(0.5 * target_size), int(0.5 * target_size)
        random_perspective_transform = RandPerspective(
            target_size=(target_size, target_size),
            degree=self.degree,
            translate=self.translate,
            scale=self.scale,
            shear=0,
            perspective=0.0,
            border=0
        )

        for i, box_hw in enumerate(boxes_hw_list):
            color_gitter = self.color_gitter
            if color_gitter is not None:
                box_hw = color_gitter(box_hw)
            h, w = box_hw.img.shape[:2]
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_size), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_size, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_size), min(target_size, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_img[y1a:y2a, x1a:x2a] = box_hw.img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            if box_hw.boxes.shape[0] > 0:
                boxes = box_hw.boxes.copy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] + padw
                boxes[:, [1, 3]] = boxes[:, [1, 3]] + padh
                result_boxes.append(boxes)
                result_labels.append(box_hw.labels)
                result_weights.append(box_hw.weights)
        if len(result_boxes) < 1:
            result_boxes = np.zeros((0, 4), dtype=np.float32)
            result_labels = np.zeros((0,), dtype=np.float32)
            result_weights = np.zeros((0,), dtype=np.float32)
        else:
            result_boxes = np.concatenate(result_boxes, axis=0)
            result_labels = np.concatenate(result_labels, axis=0)
            result_weights = np.concatenate(result_weights, axis=0)

        ret_box_info.img = result_img
        ret_box_info.boxes = result_boxes
        ret_box_info.labels = result_labels
        ret_box_info.weights = result_weights

        ret_box_info = random_perspective_transform.aug(ret_box_info)
        return ret_box_info


class MosaicWrapper(Mosaic):
    def __init__(self, sizes, **kwargs):
        super(MosaicWrapper, self).__init__(**kwargs)
        self.sizes = sizes

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        target_size = np.random.choice(self.sizes)
        self.target_size = target_size
        return super().aug(box_info)


class MixUp(BasicTransform):
    def __init__(self,
                 candidate_box_info,
                 color_gitter=None,
                 mix_ratio=0.5, **kwargs):
        """
        :param candidate_box_info:
        :param mix_ratio: 混合比例
        :param kwargs:
        """
        super(MixUp, self).__init__(**kwargs)
        self.candidate_box_info = candidate_box_info
        self.color_gitter = color_gitter
        self.mix_ratio = mix_ratio

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        if r <= 0.1 or r >= 0.9:
            return box_info
        ret_box_info = box_info.clone()
        ret_box_info.revise()
        if ret_box_info.img is None:
            raise ValueError("img is None, plz call load_img first!")
        index = random.randint(0, len(self.candidate_box_info) - 1)
        mix_box_info = self.candidate_box_info[index].clone().load_img()
        color_gitter = self.color_gitter
        if color_gitter is not None:
            mix_box_info = color_gitter(mix_box_info)
        ratio = r
        h1, w1 = ret_box_info.img.shape[:2]
        h2, w2 = mix_box_info.img.shape[:2]
        if h1 != h2 or w1 != w2:
            mix_box_info.img = cv.resize(mix_box_info.img, (w1, h1))
            if mix_box_info.boxes.shape[0] > 0:
                mix_box_info.boxes[:, [0, 2]] = mix_box_info.boxes[:, [0, 2]] * (w1 / w2)
                mix_box_info.boxes[:, [1, 3]] = mix_box_info.boxes[:, [1, 3]] * (h1 / h2)
        ret_box_info.img = (ret_box_info.img * ratio + mix_box_info.img * (1 - ratio)).astype(np.uint8)

        if len(ret_box_info.boxes) == 0:
            ret_box_info.boxes = mix_box_info.boxes
            ret_box_info.labels = mix_box_info.labels
            ret_box_info.weights = mix_box_info.weights * (1 - ratio)
        elif len(mix_box_info.boxes) > 0:
            ret_box_info.boxes = np.concatenate((ret_box_info.boxes, mix_box_info.boxes), 0)
            ret_box_info.labels = np.concatenate((ret_box_info.labels, mix_box_info.labels), 0)
            ret_box_info.weights = np.concatenate((ret_box_info.weights * ratio, mix_box_info.weights * (1 - ratio)), 0)
        return ret_box_info


class MixUpWrapper(MixUp):
    def __init__(self, beta=(8, 8), **kwargs):
        super(MixUpWrapper, self).__init__(**kwargs)
        self.beta = beta

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        r = np.random.beta(*self.beta)  # mixup ratio, alpha=beta=8.0
        if r <= 0.1 or r >= 0.9:
            return box_info
        self.mix_ratio = r
        return super().aug(box_info)


class RandCrop(BasicTransform):
    def __init__(self, min_thresh=0.5, max_thresh=0.8, **kwargs):
        super(RandCrop, self).__init__(**kwargs)
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh

    def get_crop_area(self, h, w):
        assert h > 0 and w > 0
        # 中心点
        max_marginh = int(h * (1 - self.min_thresh) / 2)
        max_marginw = int(w * (1 - self.min_thresh) / 2)
        min_marginh = int(h * (1 - self.max_thresh) / 2)
        min_marginw = int(w * (1 - self.max_thresh) / 2)
        marginvh = int(np.random.randint(min_marginh, max_marginh + 1))
        marginvw = int(np.random.randint(min_marginw, max_marginw + 1))
        # top = int(np.random.randint(0, marginvh + 1))
        # bottom = int(np.random.randint(0, marginvh + 1))
        # left = int(np.random.randint(0, marginvw + 1))
        # right = int(np.random.randint(0, marginvw + 1))
        top = marginvh
        bottom = marginvh
        left = marginvw
        right = marginvw
        return h - top - bottom, w - left - right, top, left

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        h, w = box_info.img.shape[:2]
        ret_box_info = box_info.clone()
        ret_box_info.revise()
        if ret_box_info.img is None:
            raise ValueError("img is None, plz call load_img first!")
        crop_h, crop_w, crop_top, crop_left = self.get_crop_area(h, w)
        crop_bottom, crop_right = h - crop_h - crop_top, w - crop_w - crop_left
        # crop image
        ret_box_info.img = ret_box_info.img[crop_top:h - crop_bottom, crop_left:w - crop_right, :]
        if len(ret_box_info.boxes) == 0:
            return ret_box_info
        # crop box
        crop_box = [crop_left, crop_top, w - crop_right, h - crop_bottom]
        boxes = ret_box_info.boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(crop_box[0], crop_box[2])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(crop_box[1], crop_box[3])
        # crop lose some boxes
        boxes_w, boxes_h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep_inds = (boxes_w > 1) & (boxes_h > 1)
        if not keep_inds.any():
            ret_box_info.boxes = np.zeros((0, 4), dtype=np.float32)
            ret_box_info.labels = np.zeros((0,), dtype=np.int64)
            ret_box_info.weights = np.ones((0,), dtype=np.float32)
            return ret_box_info
        # discard bad boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - crop_left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - crop_top
        # update boxes
        ret_box_info.boxes = boxes[keep_inds]
        ret_box_info.labels = ret_box_info.labels[keep_inds]
        ret_box_info.weights = ret_box_info.weights[keep_inds]
        return ret_box_info


class OneOf(BasicTransform):
    def __init__(self, transforms, **kwargs):
        """
        kwargs
        p: float 默认0.5
        transforms: 变换列表
        """
        super(OneOf, self).__init__(**kwargs)
        self.transforms = transforms
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        t = np.random.choice(self.transforms, p=self.transforms_ps)
        box_info = t(box_info)
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


class RandomSelect(BasicTransform):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms, p=0.5):
        super(RandomSelect, self).__init__(p=p)
        self.transforms = transforms

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        t = np.random.choice(self.transforms)
        box_info = t(box_info)
        return box_info 