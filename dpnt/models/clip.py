import torch
from torchvision import transforms
from .common import load_timm_encoder


def load_vit_b_16_clip_openai(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_clip_224.openai", dtype=dtype, device=device, freeze=freeze
    )


def load_vit_b_16_clip_laion400m(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_clip_224.laion400m_e32",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


def load_vit_b_16_clip_laion2b(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_clip_224.laion2b", dtype=dtype, device=device, freeze=freeze
    )


def load_vit_b_16_clip_datacomp(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_clip_224.datacompxl",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


def load_vit_b_16_clip_dfn2b(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_clip_224.dfn2b", dtype=dtype, device=device, freeze=freeze
    )


def load_vit_b_16_clip_metaclip_400m(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_clip_224.metaclip_400m",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


def load_vit_b_16_clip_metaclip_2p5b(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_clip_224.metaclip_2pt5b",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


def load_vit_b_16_siglip_webli(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_siglip_224.webli", dtype=dtype, device=device, freeze=freeze
    )


def load_vit_b_16_siglip2_webli(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch16_siglip_224.v2_webli",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


def load_vit_b_16_eva02_clip_merged2b(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "eva02_base_patch16_clip_224.merged2b",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )
