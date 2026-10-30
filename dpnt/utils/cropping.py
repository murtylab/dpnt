import os
from operator import itemgetter
from typing import List, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage as ndi
from torchvision import transforms


imagenet_transform_with_resize = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

imagenet_eval_transform_with_centercrop = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def crop_images(
    image: Union[Image.Image, List[Image.Image]],
    type: Union[str, List[str]],
    dark_thresh: int = 45,
    light_thresh: int = 253,
    min_frac: float = 0.02,
) -> Union[Image.Image, List[Image.Image]]:

    if isinstance(image, list):
        return [
            crop_images(img, type, dark_thresh, light_thresh, min_frac) for img in image
        ]

    if isinstance(type, list):
        for stage in type:
            image = crop_images(image, stage, dark_thresh, light_thresh, min_frac)
        return image

    valid_crops = [
        "trim-blackspace",
        "trim-whitespace",
        "trim-object",
        "center-crop",
        "pad-square",
        "pad-square-white",
    ]
    assert type in valid_crops, f"Invalid crop type: {type}"

    if type == "trim-blackspace":
        # Trim near-black margins. getbbox() alone fails because the
        # background is dark gray (~19), not zero, so threshold first.
        gray = np.array(image.convert("L"))
        coords = np.argwhere(gray > dark_thresh)
        if coords.size:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            image = image.crop((int(x0), int(y0), int(x1), int(y1)))

    elif type == "trim-whitespace":
        gray = np.array(image.convert("L"))
        content = gray < light_thresh
        rows = content.mean(axis=1) > min_frac
        cols = content.mean(axis=0) > min_frac
        if rows.any() and cols.any():
            y0, y1 = np.flatnonzero(rows)[[0, -1]]
            x0, x1 = np.flatnonzero(cols)[[0, -1]]
            image = image.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))

    elif type == "trim-object":
        # crop to the largest non-white blob
        box = largest_object_box(np.array(image.convert("L")), light_thresh=250)
        if box is not None:
            image = image.crop(box)

    elif type == "pad-square-white":
        # pad to square with the page's own white
        w, h = image.size
        if w != h:
            side = max(w, h)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(image.convert("RGB"), ((side - w) // 2, (side - h) // 2))
            image = canvas

    elif type == "pad-square":
        w, h = image.size
        if w != h:
            gray = np.array(image.convert("L"))
            fill = int(
                np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]))
            )
            side = max(w, h)
            canvas = Image.new(
                image.mode,
                (side, side),
                color=fill if image.mode == "L" else (fill,) * len(image.getbands()),
            )
            canvas.paste(image, ((side - w) // 2, (side - h) // 2))
            image = canvas

    elif type == "center-crop":
        width, height = image.size
        new_edge = min(width, height)
        left = (width - new_edge) // 2
        top = (height - new_edge) // 2
        image = image.crop((left, top, left + new_edge, top + new_edge))

    return image


valid_manipulations = ["invert_color", "blur", "flipped"]


def handle_image_splicing_routing(
    trials_data_row,
    data_dir: str,
) -> List[Image.Image]:
    image_name = f"{trials_data_row['stim1']}.jpg"
    image_filepath = os.path.join(data_dir, image_name)

    trial_text = trials_data_row["trialText"]

    # CFMT test trials present 3 faces in a single 1x3 row. "one of the six
    # target faces" refers to the six memorized identities, not six on-screen
    # options, so these trials are spliced the same way as triplets.
    if trial_text in (
        "Which face did you just view?",
        "Which face is one of the six target faces?",
    ):
        return splice_triplets(image_filepath=image_filepath)


def splice_triplets(image_filepath: str) -> List[Image.Image]:
    # splice in 1 row and 3 columns
    image = Image.open(image_filepath)
    width, height = image.size
    segment_width = width // 3

    spliced_image = Image.new("RGB", (segment_width * 3, height))

    for i in range(3):
        left = i * segment_width
        right = left + segment_width
        segment = image.crop((left, 0, right, height))
        spliced_image.paste(segment, (i * segment_width, 0))

    return [
        spliced_image.crop((0, 0, segment_width, height)),
        spliced_image.crop((segment_width, 0, segment_width * 2, height)),
        spliced_image.crop((segment_width * 2, 0, segment_width * 3, height)),
    ]


def content_mask(
    gray: np.ndarray, light_thresh: int = 250, median: int = 5
) -> np.ndarray:
    m = ndi.median_filter(gray, size=median) < light_thresh
    m = ndi.binary_closing(m, structure=np.ones((5, 5)))
    return ndi.binary_fill_holes(m)


def largest_object_box(gray: np.ndarray, light_thresh: int = 250):
    m = content_mask(gray, light_thresh, median=3)
    lab, k = ndi.label(m)
    if k == 0:
        return None
    biggest = int(np.argmax(ndi.sum(m, lab, range(1, k + 1)))) + 1
    rows, cols = ndi.find_objects(lab)[biggest - 1]
    return cols.start, rows.start, cols.stop, rows.stop
