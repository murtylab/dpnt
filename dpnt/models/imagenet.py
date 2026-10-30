import timm
import torch
from torchvision import transforms
from .common import freeze_backbone, load_timm_encoder


def load_vit_b_16_sup_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_224.augreg_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_vit_b_16_sam_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_224.sam_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_vit_b_16_dino_v1(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_224.dino", dtype=dtype, device=device, freeze=freeze
    )


def load_vit_b_16_mae(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    # MAE's CLS token isn't trained, so average patch tokens
    # fc_norm=False keeps the checkpoint's final norm
    return load_timm_encoder(
        "vit_base_patch16_224.mae",
        dtype=dtype,
        device=device,
        freeze=freeze,
        global_pool="avg",
        fc_norm=False,
    )


def load_vit_b_16_sup_in21k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_224.augreg_in21k", dtype=dtype, device=device, freeze=freeze
    )


def load_resnet50_sup_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "resnet50.tv_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_convnext_base_sup_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "convnext_base.fb_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_swin_base_sup_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "swin_base_patch4_window7_224.ms_in1k",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


def load_regnety_032_sup_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "regnety_032.pycls_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_vit_b_16_sup_in21k_ft_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


# untrained, same architectures
def load_vit_b_16_random(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    torch.manual_seed(0)
    preprocessor, _ = load_timm_encoder(
        "vit_base_patch16_224",
        dtype=dtype,
        device=device,
        freeze=freeze,
        img_size=224,
    )
    model = (
        timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        .to(device=device, dtype=dtype)
        .eval()
    )

    return preprocessor, freeze_backbone(model, freeze)


def load_rn50_random(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    torch.manual_seed(0)
    preprocessor, _ = load_timm_encoder(
        "resnet50",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )
    model = (
        timm.create_model("resnet50", pretrained=False, num_classes=0)
        .to(device=device, dtype=dtype)
        .eval()
    )

    return preprocessor, freeze_backbone(model, freeze)
