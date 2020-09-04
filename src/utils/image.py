from functools import partial
from PIL import Image

import numpy as np
import torch

from utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir, get_files_from_dir
from utils.logger import print_info, print_warning


IMG_EXTENSIONS = ['jpeg', 'jpg', 'JPG', 'png']


def resize(img, size, keep_aspect_ratio=True, resample=Image.ANTIALIAS, fit_inside=True):
    if isinstance(size, int):
        return resize(img, (size, size), keep_aspect_ratio=keep_aspect_ratio, resample=resample, fit_inside=fit_inside)
    elif keep_aspect_ratio:
        if fit_inside:
            ratio = float(min([s1 / s2 for s1, s2 in zip(size, img.size)]))  # XXX bug with np.float64 and round
        else:
            ratio = float(max([s1 / s2 for s1, s2 in zip(size, img.size)]))  # XXX bug with np.float64 and round
        size = round(ratio * img.size[0]), round(ratio * img.size[1])
        return img.resize(size, resample=resample)


def convert_to_img(arr):
    if isinstance(arr, torch.Tensor):
        if len(arr.shape) == 4:
            arr = arr.squeeze(0)
        elif len(arr.shape) == 2:
            arr = arr.unsqueeze(0)
        arr = arr.permute(1, 2, 0).detach().cpu().numpy()

    assert isinstance(arr, np.ndarray)
    if len(arr.shape) == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr.clip(0, 1) * 255)
    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')


def save_gif(path, name, in_ext='jpg', size=None, total_sec=10):
    files = sorted(get_files_from_dir(path, in_ext), key=lambda p: int(p.stem))
    try:
        # XXX images MUST be converted to adaptive color palette otherwise the gif has very bad quality
        imgs = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in files]
    except OSError as e:
        print_warning(e)
        return None

    if len(imgs) > 0:
        if size is not None and size != imgs[0].size:
            imgs = list(map(lambda i: resize(i, size=size), imgs))
        tpf = int(total_sec * 1000 / len(files))
        imgs[0].save(path.parent / name, optimize=False, save_all=True, append_images=imgs[1:], duration=tpf, loop=0)


class ImageResizer:
    """Resize images from a given input directory, keeping aspect ratio or not."""
    def __init__(self, input_dir, output_dir, size, in_ext=IMG_EXTENSIONS, out_ext='jpg', keep_aspect_ratio=True,
                 resample=Image.ANTIALIAS, fit_inside=True, rename=False, verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir)
        self.files = get_files_from_dir(input_dir, valid_extensions=in_ext, recursive=True, sort=True)
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.out_extension = out_ext
        self.resize_func = partial(resize, size=size, keep_aspect_ratio=keep_aspect_ratio, resample=resample,
                                   fit_inside=fit_inside)
        self.rename = rename
        self.name_size = int(np.ceil(np.log10(len(self.files))))
        self.verbose = verbose

    def run(self):
        for k, filename in enumerate(self.files):
            if self.verbose:
                print_info('Resizing and saving {}'.format(filename))
            img = Image.open(filename).convert('RGB')
            img = self.resize_func(img)
            name = str(k).zfill(self.name_size) if self.rename else filename.stem
            img.save(self.output_dir / '{}.{}'.format(name, self.out_extension))
