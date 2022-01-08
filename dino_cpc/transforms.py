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


# class RandomResizedFlippedCrop(torch.nn.Module):
#     """Crop the given image to random size and aspect ratio.
#     If the image is torch Tensor, it is expected
#     to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
#
#     A crop of random size (default: of 0.08 to 1.0) of the original size and a random
#     aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
#     is finally resized to given size.
#     This is popularly used to train the Inception networks.
#
#     Args:
#         size (int or sequence): expected output size of each edge. If size is an
#             int instead of sequence like (h, w), a square output size ``(size, size)`` is
#             made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
#             In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
#         scale (tuple of float): scale range of the cropped image before resizing, relatively to the origin image.
#         ratio (tuple of float): aspect ratio range of the cropped image before resizing.
#         interpolation (InterpolationMode): Desired interpolation enum defined by
#             :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
#             If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
#             ``InterpolationMode.BICUBIC`` are supported.
#             For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
#
#     """
#
#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=TF.InterpolationMode.BILINEAR,
#                  p=0.5):
#         super().__init__()
#         self.size = self._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
#
#         if not isinstance(scale, Sequence):
#             raise TypeError("Scale should be a sequence")
#         if not isinstance(ratio, Sequence):
#             raise TypeError("Ratio should be a sequence")
#         if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
#             warnings.warn("Scale and ratio should be of kind (min, max)")
#
#         self.interpolation = interpolation
#         self.scale = scale
#         self.ratio = ratio
#         self.p = p
#
#     @staticmethod
#     def _setup_size(size, error_msg):
#         if isinstance(size, numbers.Number):
#             return int(size), int(size)
#
#         if isinstance(size, Sequence) and len(size) == 1:
#             return size[0], size[0]
#
#         if len(size) != 2:
#             raise ValueError(error_msg)
#
#         return size
#
#     @staticmethod
#     def get_params(
#             img: Tensor, scale: List[float], ratio: List[float], p: float
#     ) -> Tuple[int, int, int, int, int, int, bool]:
#         """Get parameters for ``crop`` for a random sized crop.
#
#         Args:
#             img (PIL Image or Tensor): Input image.
#             scale (list): range of scale of the origin size cropped
#             ratio (list): range of aspect ratio of the origin aspect ratio cropped
#
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for a random
#                 sized crop.
#         """
#         flip = torch.rand(1) < p
#         width, height = TF.get_image_size(img)
#         area = height * width
#
#         log_ratio = torch.log(torch.tensor(ratio))
#         for _ in range(10):
#             target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
#             aspect_ratio = torch.exp(
#                 torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
#             ).item()
#
#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))
#
#             if 0 < w <= width and 0 < h <= height:
#                 i = torch.randint(0, height - h + 1, size=(1,)).item()
#                 j = torch.randint(0, width - w + 1, size=(1,)).item()
#                 return i, j, h, w, width, height, flip
#
#         # Fallback to central crop
#         in_ratio = float(width) / float(height)
#         if in_ratio < min(ratio):
#             w = width
#             h = int(round(w / min(ratio)))
#         elif in_ratio > max(ratio):
#             h = height
#             w = int(round(h * max(ratio)))
#         else:  # whole image
#             w = width
#             h = height
#         i = (height - h) // 2
#         j = (width - w) // 2
#         return i, j, h, w, width, height, flip
#
#     def forward(self, img):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be cropped and resized.
#
#         Returns:
#             PIL Image or Tensor: Randomly cropped and resized image.
#         """
#         i, j, h, w, W, H, flip = self.get_params(img, self.scale, self.ratio, self.p)
#         crop, pos, dim = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation), [i / H, j / W], [h / H, w / W]
#         if flip:
#             crop = TF.hflip(crop)
#         return crop, pos, dim, flip
#
#     def __repr__(self):
#         interpolate_str = self.interpolation.value
#         format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
#         format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
#         format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
#         format_string += ', interpolation={0})'.format(interpolate_str)
#         return format_string

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

        img = TF.crop(img, top, left, h, w)
        if do_flip:
            img = TF.hflip(img)

        crop_w, crop_h = TF.get_image_size(img)
        assert left % self.patch_size == 0
        assert top % self.patch_size == 0
        assert w % self.patch_size == 0
        assert h % self.patch_size == 0
        assert crop_w % self.patch_size == 0
        assert crop_h % self.patch_size == 0
        assert crop_w == w
        assert crop_h == h

        bbox = (left, top, left+w, top+h)
        return img, bbox, do_flip, (orig_w, orig_h)

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