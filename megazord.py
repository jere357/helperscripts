import random
from typing import Union

from PIL import Image
import math
import logging
import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations.utils import MAX_VALUES_BY_DTYPE


logger = logging.getLogger("base")

class Megazord:
    """
        Megazord is a composition of transformations that are applied to the input images.
        The transformations and their parameters are non-deterministic and are randomly sampled from a predefined range.
    """

    # first, second, third stage probs are the probabilities of the first, second and third stage of the pipeline
    # number_of_cycles is the number of times the pipeline is applied to the input images
    # stage_probabilites is a list of probabilities for each stage of the pipeline [color, blur, noise, compression]
    def __init__(self, number_of_cycles = 2, stage_probabilites = [0.8, 0.8, 0.8, 0.8], augment_gt = False,
                 sharpen_gt = True, convert_to_torch_tensor = True):
        self.number_of_cycles = number_of_cycles
        self.augment_gt = augment_gt
        #TODO: sharpen gt kao esrgan++
        self.sharpen_gt = sharpen_gt
        #TODO: convert to torch tensor at the end of the pipeline
        self.convert_to_torch_tensor = convert_to_torch_tensor
        #Assertions
        assert len(stage_probabilites) == 4, "There must be exactly 4 stage probabilities"

        #TODO: sinc filter na kraju svega da/ne procitat u paperu kako sta
        #9 different but all very slight color stage transformations
        self.color_stage_transforms = A.OneOf([
            A.ColorJitter(brightness=0.15, contrast=0.05, saturation=0.05, hue=0.05, p=1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=1),
            A.RGBShift(r_shift_limit=7, g_shift_limit=7, b_shift_limit=7, p=1),
            A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), p=1),
            A.RandomBrightness(limit=0.1, p=1),
            A.RandomToneCurve(p=1),
        ], p=stage_probabilites[0])
        #9 different but all very slight blur stage transformations
        self.blur_stage_transforms = A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.GaussianBlur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=5, p=1),
            A.MedianBlur(blur_limit=3, p=1),
            A.GlassBlur(sigma=0.15, max_delta=1, iterations=1, mode='fast',p=1),
            A.ZoomBlur(max_factor=1.05, step_factor=(0.01, 0.03), p=1),
            A.OpticalDistortion(p=1),
            A.Defocus(radius=(2, 4), alias_blur=(0.05, 0.1), p=1),
            A.AdvancedBlur(p=1),
            #A.Downscale(scale_min=0.5, scale_max=0.5, interpolation=0, always_apply=False, p=1),
        ], p=stage_probabilites[1])
        #4 different noise augmentations 
        self.noise_stage_transforms = A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), mean=5, p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.15, alpha_coef=0.04, p=1),
            A.Spatter(intensity=0.1, p=1),
        ], p=stage_probabilites[2])
        #2 different compression algorithms and the sinc filter
        self.compression_stage_transforms = A.OneOf([
            A.ImageCompression(quality_lower=70, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=1),
            A.ImageCompression(quality_lower=70, quality_upper=95, compression_type=A.ImageCompression.ImageCompressionType.WEBP, p=1),
            A.RingingOvershoot(p=1)
        ], p=stage_probabilites[3])
        self.transforms = [self.blur_stage_transforms, self.color_stage_transforms, self.noise_stage_transforms, self.compression_stage_transforms]

    def __call__(self, image):
        #images_lq, images_gt= input["lq"], input["gt"]
        #the order of transformation is random
        #TODO: add a random upscale + downsclae to the pipeline
        #scale_interpolations = [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        pipeline = A.Compose([random.shuffle(self.transforms)
        ])
        image_transformed = pipeline(image=image)

        return image_transformed, pipeline
    
    """
    imported from albumentations source code
    """
    @staticmethod
    def to_float(img, max_value=None):
        if max_value is None:
            try:
                max_value = MAX_VALUES_BY_DTYPE[img.dtype]
            except KeyError:
                raise RuntimeError(
                    "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                    "passing the max_value argument".format(img.dtype)
                )
        return img.astype("float32") / max_value

    """
    imported from albumentations source code
    """
    @staticmethod
    def from_float(img, dtype, max_value=None):
        if max_value is None:
            try:
                max_value = MAX_VALUES_BY_DTYPE[dtype]
            except KeyError:
                raise RuntimeError(
                    "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                    "passing the max_value argument".format(dtype)
                )
        return (img * max_value).astype(dtype)
    def rescale(self, img, scale, interpolation=cv2.INTER_AREA):
        h, w = img.shape[:2]

        need_cast = interpolation != cv2.INTER_NEAREST and img.dtype == np.uint8
        
        if need_cast:
            img = self.to_float(img)
        scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
        
        if need_cast:
            scaled = self.from_float(np.clip(scaled, 0, 1), dtype=np.dtype("uint8"))

        return scaled
