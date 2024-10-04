from __future__ import division
import sys
import random
from PIL import Image
#数据增强
try:
    import accimage
except ImportError:
    accimage = None
import numbers
import collections

import torchvision.transforms.functional as F

__all__ = ["Compose",
           "Resize",  # 尺寸缩减到对应size, 如果给定size为int,尺寸缩减到(size * height / width, size)
           "RandomScale",  # 尺寸随机缩放
           "RandomCrop",  # 随机裁剪,必要时可以进行padding
           "RandomHorizontalFlip",  # 随机水平翻转
           "ColorJitter",  # 亮度,对比度,饱和度,色调
           ]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',   #最邻近
    Image.BILINEAR: 'PIL.Image.BILINEAR', #双线性
    Image.BICUBIC: 'PIL.Image.BICUBIC',   #三次样条插值
    Image.LANCZOS: 'PIL.Image.LANCZOS',   #多相位图像插值
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
  #查看版本信息
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """
 #callable:检查一个对象是否可以调用
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize(object):
    def __init__(self, size):
        #isinstance:判断两个类型是否相同
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, sample):
        assert 'image' in sample.keys()
        assert 'label' in sample.keys()


        sample['image'] = F.resize(sample['image'], self.size, Image.BILINEAR)
        # NEAREST
        if 'depth' in sample.keys():
            sample['depth'] = F.resize(sample['depth'], self.size, Image.NEAREST)
        sample['label'] = F.resize(sample['label'], self.size, Image.NEAREST)

        return sample


class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
       #random.randint:函数返回参数1和参数2之间的任意整数
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        img = sample['image']
        if self.padding is not None:
            for key in sample.keys():
                #pad:参数1：四维或者五维的tensor Variabe;参数2：不同tensor的填充方式(上，下，左右等,数值代表填充次数)；参数3:填充的数值
                #在"contant"模式下默认填充0，mode="reflect" or "replicate"时没有value参数
                #参数4：constant‘, ‘reflect’ or ‘replicate’三种模式，指的是常量，反射，复制三种模式
                sample[key] = F.pad(sample[key], self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            for key in sample.keys():
                sample[key] = F.pad(sample[key], (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            for key in sample.keys():
                sample[key] = F.pad(sample[key], (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(sample['image'], self.size)
        for key in sample.keys():
            sample[key] = F.crop(sample[key], i, j, h, w)

        return sample


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            for key in sample.keys():
                sample[key] = F.hflip(sample[key])

        return sample


class RandomScale(object):
    def __init__(self, scale):
        assert isinstance(scale, Iterable) and len(scale) == 2
        assert 0 < scale[0] <= scale[1]
        self.scale = scale

    def __call__(self, sample):
        assert 'image' in sample.keys()
        assert 'label' in sample.keys()

        w, h = sample['image'].size
      #random.uniform:随机生成下一个实数，它在 [x, y] 范围内。
        scale = random.uniform(self.scale[0], self.scale[1])
        size = (int(round(h * scale)), int(round(w * scale)))

        # BILINEAR
        sample['image'] = F.resize(sample['image'], size, Image.BILINEAR)

        # NEAREST
        if 'depth' in sample.keys():
            sample['depth'] = F.resize(sample['depth'], size, Image.NEAREST)
        sample['label'] = F.resize(sample['label'], size, Image.NEAREST)

        return sample


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):

        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        assert 'image' in sample.keys()
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        sample['image'] = transform(sample['image'])
        return sample



