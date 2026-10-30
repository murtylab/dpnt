from torchvision.models import resnet50
from torchvision import transforms
from .common import freeze_backbone, load_timm_encoder
from ..utils.cropping import (
    imagenet_transform_with_resize,
    imagenet_eval_transform_with_centercrop,
)
import os
import torch

try:
    import toponets
except ImportError:
    raise ImportError(
        "Toponets is not installed. Please install it from"
        "https://github.com/murtylab/topoloss or remove from the model bank"
    )


def load_resnet50_checkpoint(
    state_dict: dict,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
    preprocessor: transforms.Compose = imagenet_transform_with_resize,
) -> tuple[transforms.Compose, torch.nn.Module]:

    model = resnet50(weights=None)

    model.load_state_dict(state_dict)
    model.fc = torch.nn.Identity()
    model = model.to(device=device, dtype=dtype).eval()

    return preprocessor, freeze_backbone(model, freeze)


def load_resnet50_robust_l2_eps3(
    local_checkpoint_path: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    assert os.path.exists(
        local_checkpoint_path
    ), f"robust ResNet-50 checkpoint not found: {local_checkpoint_path}"

    ckpt = torch.load(local_checkpoint_path, map_location="cpu", weights_only=False)[
        "model"
    ]
    prefix = "module.model."
    state_dict = {k[len(prefix) :]: v for k, v in ckpt.items() if k.startswith(prefix)}

    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py#L58-L60 (Resize 256, CenterCrop 224) + datasets.py#L195-L196 (ImageNet mean/std)
    preprocessor = imagenet_eval_transform_with_centercrop

    return load_resnet50_checkpoint(
        state_dict,
        dtype=dtype,
        device=device,
        freeze=freeze,
        preprocessor=preprocessor,
    )


def load_resnet50_toponet_tau30(
    local_checkpoint_path: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:

    model = toponets.resnet50(tau=30.0, checkpoint_path=local_checkpoint_path)
    model.fc = (
        torch.nn.Identity()
    )  # penultimate (pooled) features, not the 1000-way ImageNet logits
    model = model.to(device=device, dtype=dtype).eval()
    preprocessor = imagenet_eval_transform_with_centercrop

    return preprocessor, freeze_backbone(model, freeze)


def load_resnet18_toponets_tau10(
    local_checkpoint_path: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:

    model = toponets.resnet18(tau=10.0, checkpoint_path=local_checkpoint_path)
    model.fc = (
        torch.nn.Identity()
    )  # penultimate (pooled) features, not the 1000-way ImageNet logits
    model = model.to(device=device, dtype=dtype).eval()
    preprocessor = imagenet_eval_transform_with_centercrop

    return preprocessor, freeze_backbone(model, freeze)


def load_resnet50_ssl_yfcc100m(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "resnet50.fb_ssl_yfcc100m_ft_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_resnet50_swsl_ig1b(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "resnet50.fb_swsl_ig1b_ft_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_resnet50_clip_openai(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "resnet50_clip.openai", dtype=dtype, device=device, freeze=freeze
    )


def load_resnet50_clip_yfcc15m(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "resnet50_clip.yfcc15m", dtype=dtype, device=device, freeze=freeze
    )


def load_resnet50_clip_cc12m(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "resnet50_clip.cc12m", dtype=dtype, device=device, freeze=freeze
    )
