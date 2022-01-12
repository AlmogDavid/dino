import math
import numpy as np
import numbers
import random
import warnings
from collections import Sequence
from typing import List, Tuple
import torch.nn.functional as F
import torch
from PIL import ImageFilter, ImageOps
from PIL import Image
from torch import Tensor
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GaussianBlur:
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class MinimalSizeResize(torch.nn.Module):
    def __init__(self, minimal_size: int, patch_size: int, interpolation: TF.InterpolationMode = TF.InterpolationMode.BICUBIC):
        super().__init__()
        assert minimal_size % patch_size == 0
        self.minimal_size = minimal_size
        self.interpolation = interpolation
        self.patch_size = patch_size

    @staticmethod
    def _expand2square(pil_img):
        width, height = pil_img.size
        pad_width = pad_height = (0, 0)
        if width == height:
            return pil_img
        elif width > height:
            pad_height = ((width - height) // 2, (width - height + 1) // 2)
        else:  # height > width
            pad_width = ((height - width) // 2, (height - width + 1) // 2)

        img = np.pad(pil_img, pad_width=(pad_height, pad_width, (0, 0)), mode='symmetric')
        img = Image.fromarray(img)
        return img

    def forward(self, img: Tensor) -> Tensor:
        w, h = TF.get_image_size(img)
        if w < self.minimal_size or h < self.minimal_size:
            if w < h:
                new_w = self.minimal_size
                new_h = int((new_w / w) * h)
            else:  # h>=w
                new_h = self.minimal_size
                new_w = int((new_h / h) * w)

            img = TF.resize(img, size=[new_h, new_w], interpolation=self.interpolation)
            w, h = TF.get_image_size(img)

        if w % self.patch_size or h % self.patch_size:  # The image size is not a multiplication of the patch size
            if w < h:
                new_h = h - h % self.patch_size + self.patch_size
                new_w = int(new_h / h) * w
            else:  # w>=h
                new_w = w - w % self.patch_size + self.patch_size
                new_h = int(new_w / w) * h

            img = TF.resize(img, size=[new_h, new_w], interpolation=self.interpolation)

        # Pad the image in order to get to 1 aspect ratio
        img = self._expand2square(img)

        img_w, img_h = TF.get_image_size(img)
        assert img_w % self.patch_size == 0
        assert img_h % self.patch_size == 0

        return img

    def __repr__(self):
        return self.__class__.__name__ + "(minimal_size={0}, interpolation={1})".format(self.minimal_size, self.interpolation)


class RandomCropFlip(torch.nn.Module):

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int], flip_prob: float, patch_size: int) -> Tuple[int, int, int, int, bool]:
        do_flip = random.random() < flip_prob

        w, h = TF.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w, do_flip

        start_y = torch.randint(0, (h - th + 1)//patch_size, size=(1,)).item() * patch_size  # Make sure the crop is a multiplication of the patch size
        start_x = torch.randint(0, (w - tw + 1)//patch_size, size=(1,)).item() * patch_size

        return start_y, start_x, th, tw, do_flip

    @staticmethod
    def _setup_size(size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size

    def __init__(self, size, patch_size: int, flip_prob: float = 0.5):
        super().__init__()

        self.size = tuple(self._setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.patch_size = patch_size
        self.flip_prob = flip_prob

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image and flipped.
        """
        top, left, h, w, do_flip = self.get_params(img, self.size, self.flip_prob, self.patch_size)
        orig_w, orig_h = TF.get_image_size(img)

        cropped_img = TF.crop(img, top, left, h, w)
        if do_flip:
            cropped_img = TF.hflip(cropped_img)

        crop_w, crop_h = TF.get_image_size(cropped_img)
        assert left % self.patch_size == 0
        assert top % self.patch_size == 0
        assert w % self.patch_size == 0
        assert h % self.patch_size == 0
        assert crop_w % self.patch_size == 0
        assert crop_h % self.patch_size == 0
        assert crop_w == w
        assert crop_h == h

        bbox = (left, top, left+w, top+h)
        return cropped_img, bbox, do_flip, (orig_w, orig_h)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1}, flip={2:.2f})".format(self.size, self.padding, self.flip_prob)


class DataAugmentationDINOCPC:

    def __init__(self, local_crops_number, global_crop_size, local_crop_size, maximal_patch_size: int):
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.global_resize_and_flip = RandomCropFlip(global_crop_size, maximal_patch_size)
        # first global crop
        self.global_transfo1 = transforms.Compose([
            color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_resize_and_flip = RandomCropFlip(local_crop_size, maximal_patch_size)
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops, bboxes, flips, orig_img_size = [], [], [], []

        resized_image1, box1, flip1, img_size1 = self.global_resize_and_flip(image)
        crops.append(self.global_transfo1(resized_image1))
        bboxes.append(box1)
        flips.append(flip1)
        orig_img_size.append(img_size1)

        resized_image2, box2, flip2, img_size2 = self.global_resize_and_flip(image)
        crops.append(self.global_transfo2(resized_image2))
        bboxes.append(box2)
        flips.append(flip2)
        orig_img_size.append(img_size2)

        for _ in range(self.local_crops_number):
            resized_image, box, flip, img_size = self.local_resize_and_flip(image)
            crops.append(self.local_transfo(resized_image))
            bboxes.append(box)
            flips.append(flip)
            orig_img_size.append(img_size)

        bboxes = torch.from_numpy(np.asarray(bboxes, dtype=np.float32))
        return crops, bboxes, torch.from_numpy(np.asarray(flips, dtype=np.bool)), torch.from_numpy(np.asarray(orig_img_size, dtype=np.int32))