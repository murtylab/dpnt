import torch
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor
from .common import load_timm_encoder, freeze_backbone, HFViTEmbedder


def load_vit_b_14_dino_v2(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "vit_base_patch14_dinov2.lvd142m",
        dtype=dtype,
        device=device,
        freeze=freeze,
        img_size=224,
        crop_pct=224
        / 256,  # https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L77-L87 (resize 256 bicubic, center crop 224)
    )


def load_webssl_dino300m(dtype=torch.float32, device="cuda:0", freeze=True):
    hf_model = AutoModel.from_pretrained(
        "facebook/webssl-dino300m-full2b-224", dtype=dtype
    )
    model = HFViTEmbedder(hf_model, "cls").to(device).eval()

    # https://huggingface.co/facebook/webssl-dino300m-full2b-224/blob/main/preprocessor_config.json (shortest edge 224 bicubic, center crop 224, ImageNet mean/std)
    processor = AutoImageProcessor.from_pretrained(
        "facebook/webssl-dino300m-full2b-224", backend="pil" 
    )
    preprocessor = transforms.Lambda(
        lambda im: processor(images=im.convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ][0]
    )
    return preprocessor, freeze_backbone(model, freeze)


def load_webssl_mae300m(dtype=torch.float32, device="cuda:0", freeze=True):
    hf_model = AutoModel.from_pretrained(
        "facebook/webssl-mae300m-full2b-224", dtype=dtype, mask_ratio=0.0
    )
    model = HFViTEmbedder(hf_model, "mean").to(device).eval()

    # https://huggingface.co/facebook/webssl-mae300m-full2b-224/blob/main/preprocessor_config.json (shortest edge 224 bicubic, center crop 224, ImageNet mean/std)
    processor = AutoImageProcessor.from_pretrained(
        "facebook/webssl-mae300m-full2b-224", backend="pil" 
    )
    preprocessor = transforms.Lambda(
        lambda im: processor(images=im.convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ][0]
    )
    return preprocessor, freeze_backbone(model, freeze)


def load_aimv2_large(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "aimv2_large_patch14_224", dtype=dtype, device=device, freeze=freeze
    )


def load_beit_base_in21k(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "beit_base_patch16_224.in22k_ft_in22k",
        dtype=dtype,
        device=device,
        freeze=freeze,
    )


# segmentation (SA-1B)
def load_samvit_base_sa1b(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:
    return load_timm_encoder(
        "samvit_base_patch16.sa1b", dtype=dtype, device=device, freeze=freeze
    )
