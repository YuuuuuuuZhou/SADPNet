
from typing import List, Tuple, Union

import mmcv
import numpy as np
from mmengine.utils import is_str


def palette_val(palette: List[tuple]) -> List[tuple]:

    new_palette = []
    for color in palette:
        color = [c / 255 for c in color]
        new_palette.append(tuple(color))
    return new_palette


def get_palette(palette: Union[List[tuple], str, tuple],
                num_classes: int) -> List[Tuple[int]]:
    assert isinstance(num_classes, int)

    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == 'random' or palette is None:
        state = np.random.get_state()
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif palette == 'coco':
        from mmdet.datasets import CocoDataset, CocoPanopticDataset
        dataset_palette = CocoDataset.METAINFO['palette']
        if len(dataset_palette) < num_classes:
            dataset_palette = CocoPanopticDataset.METAINFO['palette']
    elif palette == 'citys':
        from mmdet.datasets import CityscapesDataset
        dataset_palette = CityscapesDataset.METAINFO['palette']
    elif palette == 'voc':
        from mmdet.datasets import VOCDataset
        dataset_palette = VOCDataset.METAINFO['palette']
    elif is_str(palette):
        dataset_palette = [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')

    assert len(dataset_palette) >= num_classes, \
        'The length of palette should not be less than `num_classes`.'
    return dataset_palette


def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    scales = 0.5 + (areas - min_area) // (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def jitter_color(color: tuple) -> tuple:
    return color
