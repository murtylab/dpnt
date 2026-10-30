import torch
from torchvision import transforms
from .common import load_timm_encoder


def load_convnext_base_sup_in21k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "convnext_base.fb_in22k", dtype=dtype, device=device, freeze=freeze
    )


def load_convnextv2_base_fcmae(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    """ConvNeXt-V2-B pretrained with FCMAE -- a fully-convolutional masked
    autoencoder, the CNN analogue of MAE / EVA-02's MIM stage."""
    return load_timm_encoder(
        "convnextv2_base.fcmae", dtype=dtype, device=device, freeze=freeze
    )


def load_convnext_base_clip_laiona(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    """ConvNeXt-B CLIP on LAION-Aesthetic -- a differently-filtered LAION subset
    than the benchmark-filtered DFN / DataComp sets."""
    return load_timm_encoder(
        "convnext_base.clip_laiona", dtype=dtype, device=device, freeze=freeze
    )


def load_convnext_base_clip_laion2b(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "convnext_base.clip_laion2b", dtype=dtype, device=device, freeze=freeze
    )


def load_convnext_large_clip_laion2b(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "convnext_large_mlp.clip_laion2b_augreg",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


def load_regnety_320_seer(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "regnety_320.seer", dtype=dtype, device=device, freeze=freeze
    )


def load_regnety_160_swag(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "regnety_160.swag_lc_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_resnet50x4_clip_openai(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "resnet50x4_clip.openai", dtype=dtype, device=device, freeze=freeze
    )


def load_inception_resnet_v2_sup_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "inception_resnet_v2.tf_in1k", dtype=dtype, device=device, freeze=freeze
    )


def load_inception_resnet_v2_adv_in1k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "inception_resnet_v2.tf_ens_adv_in1k", dtype=dtype, device=device, freeze=freeze
    )
