import torch
from torchvision import transforms
import os
import pickle
from huggingface_hub import hf_hub_download
from .common import ONNXFaceEmbedder
from torchvision.models import resnet50
from .common import freeze_backbone, CLIPVisionCLS
import numpy as np
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from transformers import CLIPModel


def load_arcface_r50_webface600k(
    local_checkpoint_path: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, ONNXFaceEmbedder]:

    assert os.path.exists(local_checkpoint_path), (
        f"ArcFace ONNX weights not found at {local_checkpoint_path}. Download them "
        "once with `from insightface.app import FaceAnalysis; "
        "FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)`."
    )

    # https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py#L41-L42 (127.5/127.5 unless the ONNX embeds Sub/Mul)
    preprocessor = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize((112, 112)),
            transforms.PILToTensor(),
            transforms.Lambda(
                lambda t: (t.float() if embedded_norm else (t.float() - 127.5) / 127.5)
            ),
        ]
    )

    # readout: the flattened 512 x 7 x 7 map (after bn2), the input of the fc projection
    encoder = ONNXFaceEmbedder(local_checkpoint_path, tap="681")
    embedded_norm = encoder.has_embedded_normalization

    return preprocessor, encoder


def load_arcface_r50_ms1mv3(
    local_checkpoint_path: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, ONNXFaceEmbedder]:

    assert os.path.exists(local_checkpoint_path), (
        f"ArcFace R50 MS1MV3 ONNX weights not found at {local_checkpoint_path}. "
        "Download ms1mv3_arcface_r50 from the InsightFace model zoo."
    )

    # https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py#L41-L42 (127.5/127.5 unless the ONNX embeds Sub/Mul)
    preprocessor = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize((112, 112)),
            transforms.PILToTensor(),
            transforms.Lambda(
                lambda t: (t.float() if embedded_norm else (t.float() - 127.5) / 127.5)
            ),
        ]
    )

    # readout: the flattened 512 x 7 x 7 map (after bn2), the input of the fc projection
    encoder = ONNXFaceEmbedder(local_checkpoint_path, tap="681")
    embedded_norm = encoder.has_embedded_normalization

    return preprocessor, encoder


def load_vit_l_face_cosface_webface42m(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, ONNXFaceEmbedder]:

    ckpt = hf_hub_download(
        "kunkunlin1221/face-recognition_vit-l-pfc0.3-cosface-web42m",
        "wf42m_pfc03_vit-l_fp32.onnx",
    )
    # https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py#L41-L42 (same 112px (x - 127.5) / 127.5 convention)
    preprocessor = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize((112, 112)),
            transforms.PILToTensor(),  # uint8 RGB
            transforms.Lambda(lambda t: (t.float() - 127.5) / 127.5),
        ]
    )
    # readout: the flattened tokens after the final norm, the input of the MLP projection
    return preprocessor, ONNXFaceEmbedder(ckpt, tap="/Reshape_output_0")


def load_resnet50_vggface2_official(
    local_checkpoint_path: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:

    assert os.path.exists(local_checkpoint_path), f"not found: {local_checkpoint_path}"
    with open(local_checkpoint_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    state_dict = {
        k: torch.from_numpy(v) for k, v in raw.items() if not k.startswith("fc.")
    }  # drop 8631-way classifier

    model = resnet50(weights=None)
    for layer in (model.layer2, model.layer3, model.layer4):
        layer[0].conv1.stride, layer[0].conv2.stride = (2, 2), (1, 1)
    model.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
    model.avgpool = torch.nn.AvgPool2d(7, stride=1)
    model.fc = torch.nn.Identity()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    assert not missing and not unexpected, f"{missing=} {unexpected=}"

    # https://github.com/ox-vgg/vgg_face2#trained-models (shorter side 256, center 224 crop; mean is BGR per #compatibility; values: standard_evaluation/pytorch_feature_extractor.py#L12)
    preprocessor = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.Lambda(lambda t: t.float().flip(0)),  # RGB -> BGR
            transforms.Normalize(mean=[91.4953, 103.8827, 131.0912], std=[1, 1, 1]),
        ]
    )

    model = model.to(device=device, dtype=dtype).eval()
    return preprocessor, freeze_backbone(model, freeze)


def load_farl_vitb16_laionface(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
) -> tuple[transforms.Compose, torch.nn.Module]:

    # https://github.com/openai/CLIP/blob/main/clip/clip.py#L85 (FaRL reuses CLIP's preprocess: https://github.com/FacePerceiver/FaRL#use-farl-as-faceclip)
    preprocessor = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.PILToTensor(),
            transforms.Lambda(lambda t: t.float() / 255.0),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip = CLIPModel.from_pretrained("Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64")
    vision = clip.vision_model.to(device=device, dtype=dtype).eval()

    return preprocessor, freeze_backbone(CLIPVisionCLS(vision).eval(), freeze)


def load_facenet_vggface2(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
    image_size: int = 160,
) -> tuple[transforms.Compose, torch.nn.Module]:

    # https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py#L508-L509 (fixed_image_standardization: (x - 127.5) / 128)
    preprocessor = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization,
        ]
    )

    encoder = (
        InceptionResnetV1(pretrained="vggface2").to(device=device, dtype=dtype).eval()
    )
    # the pooled 1792-d feature, before the 512-d embedding projection (linear + batchnorm)
    encoder.last_linear = torch.nn.Identity()
    encoder.last_bn = torch.nn.Identity()

    return preprocessor, freeze_backbone(encoder, freeze)


def load_facenet_casia_webface(
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
    image_size: int = 160,
) -> tuple[transforms.Compose, torch.nn.Module]:

    # https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py#L508-L509 (fixed_image_standardization: (x - 127.5) / 128)
    preprocessor = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization,
        ]
    )

    encoder = (
        InceptionResnetV1(pretrained="casia-webface")
        .to(device=device, dtype=dtype)
        .eval()
    )
    # the pooled 1792-d feature, before the 512-d embedding projection (linear + batchnorm)
    encoder.last_linear = torch.nn.Identity()
    encoder.last_bn = torch.nn.Identity()

    return preprocessor, freeze_backbone(encoder, freeze)
