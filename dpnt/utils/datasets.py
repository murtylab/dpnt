import json
import os
import shutil

import torch
from PIL import Image
from tqdm import tqdm

from ..utils.csv import read_csv
from ..utils.mapping import normalize_stimuli_name
from ..utils.cropping import (
    crop_images,
    handle_image_splicing_routing,
)


class CFMTDataset:
    def __init__(
        self,
        cfmt_csv_filepath: str,
        cfmt_stimuli_dir: str,
        transform: callable,
        crop: str = "trim-blackspace",
    ):
        self.transform = transform
        self.cfmt_stimuli_dir = cfmt_stimuli_dir
        self.crop = crop

        data = read_csv(cfmt_csv_filepath)
        data = data[data["type"] == "test"]

        self.cache(data)

    def cache(self, data: list):
        self.stimuli = []
        for _, row in data.iterrows():
            images = handle_image_splicing_routing(
                trials_data_row=row,
                data_dir=self.cfmt_stimuli_dir,
            )
            images = crop_images(images, type=self.crop)

            self.stimuli.append(
                {
                    "images": images,
                    "key": int(row["key"]),
                    "label": row["stim1"],
                }
            )

    def __len__(self):
        return len(self.stimuli)

    def __getitem__(self, idx):
        entry = self.stimuli[idx]
        images = entry["images"]
        if self.transform is not None:
            images = [self.transform(image) for image in images]
            images = torch.stack(images, dim=0)
        return {
            "images": images,
            "key": entry["key"],
            "label": entry["label"],
        }

    def get_dataloader(self, shuffle: bool = True):
        return torch.utils.data.DataLoader(
            self,
            batch_size=1,  # must be 1: each trial has its own label/key and a variable number of ground-truth facebank images
            shuffle=shuffle,
        )


class FacebankDataset:
    def __init__(
        self,
        cfmt_stimuli_dir: str,
        cfmt_csv_filepath: str,
        facebank_dir: str,
        transform: callable,
        crop: str = "trim-blackspace",
        device: str = "cuda:0",
        dtype: str = torch.float32,
        facebank_config_filepath: str = "facebank_config.json",
        name_normalizer: callable = normalize_stimuli_name,
    ):
        assert os.path.exists(
            cfmt_stimuli_dir
        ), f"cfmt_stimuli_dir does not exist: {cfmt_stimuli_dir}"
        assert os.path.exists(
            cfmt_csv_filepath
        ), f"cfmt_csv_filepath does not exist: {cfmt_csv_filepath}"

        self.cfmt_stimuli_dir = cfmt_stimuli_dir
        self.cfmt_csv_filepath = cfmt_csv_filepath
        self.name_normalizer = name_normalizer
        self.facebank_config_filepath = facebank_config_filepath
        self.crop = crop
        self.device = device
        self.dtype = dtype
        self.facebank_dir = self.generate_facebank(
            cfmt_csv_filepath, cfmt_stimuli_dir, facebank_dir
        )

        self.cache()
        self.transform = transform

    def generate_facebank(self, csv_filepath, dataset_dir, facebank_dir):
        config = {"crop": self.crop, "format": "png"}
        marker_filepath = os.path.join(facebank_dir, self.facebank_config_filepath)

        if os.path.exists(facebank_dir):
            existing = None
            if os.path.exists(marker_filepath):
                with open(marker_filepath) as f:
                    existing = json.load(f)
            if existing == config:
                print(
                    f"\033[92mFacebank directory already exists: {facebank_dir}\033[0m"
                )
                return facebank_dir
            print(
                f"\033[93mRebuilding {facebank_dir}: built under {existing}, "
                f"this run needs {config}\033[0m"
            )
            # Wipe the old bank first
            shutil.rmtree(facebank_dir)

        print(f"\033[92mGenerating facebank in {facebank_dir}\033[0m")
        data = read_csv(csv_filepath)
        data.columns = data.columns.str.strip()
        os.makedirs(facebank_dir, exist_ok=True)

        saved = 0
        for _, row in tqdm(
            data.iterrows(), total=len(data), desc="Generating facebank"
        ):
            if str(row["trialText"]).strip().lower() != "memorize":
                continue
            new_image_filepath = self.name_normalizer(str(row["stim1"]))
            if new_image_filepath is None:  # normalizer opted this row out of the bank
                continue
            image_filepath = (
                f"{str(row['stim1']).strip()}{str(row['stimFormat']).strip()}"
            )
            image = Image.open(os.path.join(dataset_dir, image_filepath))
            image = crop_images(image, type=self.crop)
            image.save(
                os.path.join(
                    facebank_dir, os.path.splitext(new_image_filepath)[0] + ".png"
                )
            )
            saved += 1

        with open(marker_filepath, "w") as f:
            json.dump(config, f)

        print(f"\033[92mSaved {saved} images\033[0m")

        return facebank_dir

    def cache(self):
        self.images = []
        self.labels = []
        self.views = []
        self.files = []

        for images in sorted(os.listdir(self.facebank_dir)):
            if not images.endswith(".png"):
                continue
            image = Image.open(
                fp=os.path.join(self.facebank_dir, images),
            ).convert("RGB")

            self.images.append(image)
            self.files.append(images)
            full_label = images.split(".")[0]

            self.labels.append(full_label.split("_")[0])
            self.views.append(full_label.split("_")[1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = (
            self.transform(self.images[idx])
            if self.transform is not None
            else self.images[idx]
        )

        return {
            "image": image,
            "label": self.labels[idx],
            "view": self.views[idx],
        }

    def facebank_indices(self, name: str, heldout_view: str) -> list:
        assert (
            heldout_view in ("all", "other") or heldout_view in self.views
        ), f"heldout_view must be 'all', 'other', or a facebank view {sorted(set(self.views))}, got {heldout_view}"
        keep_all = heldout_view in ("all", "other")
        return [
            i
            for i, label in enumerate(self.labels)
            if label == name and (keep_all or self.views[i] != heldout_view)
        ]

    def get_all_images(self) -> torch.Tensor:
        return torch.stack(
            [
                self.transform(image) if self.transform is not None else image
                for image in self.images
            ]
        ).to(device=self.device, dtype=self.dtype)

    def get_all_image_for_name(
        self, name: str, heldout_view: str, verbose: bool = False
    ) -> torch.Tensor:
        """
        Get all images for a given name, optionally excluding a heldout view.

        Args:
            name (str):         The name of the person to retrieve images for.
            heldout_view (str): The view to exclude from the returned images. Can
                                be "left", "right", "front", "all", or "other"
                                (both "all" and "other" -- a CFMT+ long-form probe
                                with no facebank counterpart -- exclude nothing).
            verbose (bool):     Whether to print verbose output.

        Returns:
            torch.Tensor:       A tensor containing all images for the given name, excluding
                                the optional heldout view

        """

        selected = [
            (
                self.transform(self.images[i])
                if self.transform is not None
                else self.images[i]
            )
            for i in self.facebank_indices(name, heldout_view)
        ]

        if verbose:
            print(
                f"name={name} heldout={heldout_view} \
                facebank_views={[self.views[i] for i,l in enumerate(self.labels) if l==name]} \
                kept={len(selected)}"
            )

        return torch.stack(selected, dim=0).to(device=self.device, dtype=self.dtype)
