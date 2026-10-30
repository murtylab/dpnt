import torch
from torchvision import transforms
import timm
import onnxruntime as ort
import onnx


def freeze_backbone(module: torch.nn.Module, freeze: bool) -> torch.nn.Module:
    if freeze:
        for p in module.parameters():
            p.requires_grad_(False)
    return module


def load_timm_encoder(
    tag: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    freeze: bool = True,
    squash: bool = False,
    crop_pct: float = None,
    **model_kwargs,  # passed to timm.create_model, e.g. img_size=224, global_pool="avg", fc_norm=False
) -> tuple[transforms.Compose, torch.nn.Module]:

    if tag.split(".")[0] in {"resnet50_clip", "resnet50x4_clip"}:
        # AttentionPool2d in older timm versions creates a random projection
        # when constructed with num_classes=0. Load the native pretrained head
        # first, then remove only its output projection, retaining learned pooling.
        encoder = timm.create_model(tag, pretrained=True, **model_kwargs)
        encoder.reset_classifier(0)
    else:
        encoder = timm.create_model(tag, pretrained=True, num_classes=0, **model_kwargs)
    encoder = encoder.to(device=device, dtype=dtype).eval()

    cfg = dict(timm.data.resolve_model_data_config(encoder))
    img_size = model_kwargs.get("img_size")
    if img_size is not None:
        # resolve_model_data_config reports the checkpoint's native size, which
        # is no longer what the model was built for
        cfg["input_size"] = (cfg["input_size"][0], img_size, img_size)
    if crop_pct is not None:
        cfg["crop_pct"] = crop_pct
    if squash:
        cfg["crop_pct"] = 1.0
        cfg["crop_mode"] = "squash"
    preprocessor = timm.data.create_transform(**cfg, is_training=False)

    return preprocessor, freeze_backbone(encoder, freeze)


class HFViTEmbedder(torch.nn.Module):
    """transformers ViT-style backbone -> [B, D] embedding, pooled from
    last_hidden_state via the CLS token or the mean of patch tokens."""

    def __init__(self, hf_model: torch.nn.Module, pooling: str = "cls"):
        super().__init__()
        assert pooling in ("cls", "mean")
        self.hf_model, self.pooling = hf_model, pooling

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.hf_model(pixel_values=images).last_hidden_state
        return h[:, 0] if self.pooling == "cls" else h[:, 1:].mean(dim=1)


class ONNXFaceEmbedder:
    """Wraps an ONNX face-recognition backbone as a callable matching the
    torch interface the rest of this file expects: takes a preprocessed
    [N, C, H, W] tensor, returns an [N, D] embedding tensor.
    """

    def __init__(self, onnx_path: str, providers: list = None, tap: str = None):
        """tap: name of an intermediate [N, D] tensor to return instead of the graph's output (the
        flattened features going into the head's projection)"""
        self.onnx_path = onnx_path
        self.tap = tap

        options = ort.SessionOptions()
        options.log_severity_level = 3

        proto = onnx.load(onnx_path)
        graph = proto.graph
        early_ops = {node.op_type for node in graph.node[:8]}

        # https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py#L25-L42
        self.has_embedded_normalization = {"Sub", "Mul"} <= early_ops

        source = onnx_path
        if tap is not None:
            produced = {out for node in graph.node for out in node.output}
            assert tap in produced, f"{tap!r} is not a tensor of {onnx_path}"
            del graph.output[:]
            graph.output.append(onnx.helper.make_empty_tensor_value_info(tap))
            source = proto.SerializeToString()

        self.session = ort.InferenceSession(
            source,
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        device, dtype = images.device, images.dtype
        x = images.detach().float().cpu().numpy()
        (out,) = self.session.run(None, {self.input_name: x})
        return torch.from_numpy(out).to(device=device, dtype=dtype)


class CLIPVisionCLS(torch.nn.Module):
    """transformers CLIP vision tower -> [N, D] pooled embedding."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values=images).pooler_output
