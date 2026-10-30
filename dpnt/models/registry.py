import torch

from .imagenet import (
    load_vit_b_16_sup_in1k,
    load_vit_b_16_sam_in1k,
    load_vit_b_16_dino_v1,
    load_vit_b_16_mae,
    load_vit_b_16_sup_in21k,
    load_resnet50_sup_in1k,
    load_convnext_base_sup_in1k,
    load_swin_base_sup_in1k,
    load_regnety_032_sup_in1k,
    load_vit_b_16_sup_in21k_ft_in1k,
)

from .clip import (
    load_vit_b_16_clip_openai,
    load_vit_b_16_clip_laion400m,
    load_vit_b_16_clip_laion2b,
    load_vit_b_16_clip_datacomp,
    load_vit_b_16_clip_dfn2b,
    load_vit_b_16_clip_metaclip_400m,
    load_vit_b_16_clip_metaclip_2p5b,
    load_vit_b_16_siglip_webli,
    load_vit_b_16_siglip2_webli,
    load_vit_b_16_eva02_clip_merged2b,
)


from .resnet import (
    load_resnet50_robust_l2_eps3,
    load_resnet50_toponet_tau30,
    load_resnet18_toponets_tau10,
    load_resnet50_ssl_yfcc100m,
    load_resnet50_clip_yfcc15m,
    load_resnet50_swsl_ig1b,
    load_resnet50_clip_openai,
    load_resnet50_clip_cc12m,
)

from .ssl import (
    load_vit_b_14_dino_v2,
    load_webssl_dino300m,
    load_webssl_mae300m,
    load_aimv2_large,
    load_beit_base_in21k,
    load_samvit_base_sa1b,
)

from .face_models import (
    load_resnet50_vggface2_official,
    load_facenet_vggface2,
    load_facenet_casia_webface,
    load_arcface_r50_webface600k,
    load_arcface_r50_ms1mv3,
    load_vit_l_face_cosface_webface42m,
    load_farl_vitb16_laionface,
)

from .conv import (
    load_convnext_base_sup_in21k,
    load_convnextv2_base_fcmae,
    load_convnext_base_clip_laion2b,
    load_convnext_base_clip_laiona,
    load_convnext_large_clip_laion2b,
    load_regnety_320_seer,
    load_regnety_160_swag,
    load_resnet50x4_clip_openai,
    load_inception_resnet_v2_sup_in1k,
    load_inception_resnet_v2_adv_in1k,
)

from .imagenet import (
    load_vit_b_16_random,
    load_rn50_random,
)


full_model_registry = {
    "vit_b16_sup_in1k": {
        "loader": load_vit_b_16_sup_in1k,
        "architecture": "vit_b16",
        "training_objective": "supervised",
        "training_dataset": "in1k",
        "training_objective_group": "supervised classification",
    },
    "vit_b16_sam_in1k": {
        "loader": load_vit_b_16_sam_in1k,
        "architecture": "vit_b16",
        "training_objective": "supervised-sam",
        "training_dataset": "in1k",
        "training_objective_group": "supervised classification",
    },
    "vit_b16_dino_v1": {
        "loader": load_vit_b_16_dino_v1,
        "architecture": "vit_b16",
        "training_objective": "dino",
        "training_dataset": "in1k",
        "training_objective_group": "self-supervised",
    },
    "vit_b16_mae": {
        "loader": load_vit_b_16_mae,
        "architecture": "vit_b16",
        "training_objective": "mae",
        "training_dataset": "in1k",
        "training_objective_group": "self-supervised",
    },
    "vit_b16_sup_in21k": {
        "loader": load_vit_b_16_sup_in21k,
        "architecture": "vit_b16",
        "training_objective": "supervised",
        "training_dataset": "in21k",
        "training_objective_group": "supervised classification",
    },
    "vit_b16_sup_in21k_ft_in1k": {
        "loader": load_vit_b_16_sup_in21k_ft_in1k,
        "architecture": "vit_b16",
        "training_objective": "supervised",
        "training_dataset": "in21k_ft_in1k",
        "training_objective_group": "supervised classification",
    },
    "vit_b16_clip_openai": {
        "loader": load_vit_b_16_clip_openai,
        "architecture": "vit_b16",
        "training_objective": "clip",
        "training_dataset": "wit400m",
        "training_objective_group": "language-supervised medium",
    },
    "vit_b16_clip_laion400m": {
        "loader": load_vit_b_16_clip_laion400m,
        "architecture": "vit_b16",
        "training_objective": "clip",
        "training_dataset": "laion400m",
        "training_objective_group": "language-supervised medium",
    },
    "vit_b16_clip_laion2b": {
        "loader": load_vit_b_16_clip_laion2b,
        "architecture": "vit_b16",
        "training_objective": "clip",
        "training_dataset": "laion2b",
        "training_objective_group": "language-supervised large",
    },
    "vit_b16_clip_datacomp": {
        "loader": load_vit_b_16_clip_datacomp,
        "architecture": "vit_b16",
        "training_objective": "clip",
        "training_dataset": "datacomp_xl",
        "training_objective_group": "language-supervised large",
    },
    "vit_b16_clip_dfn2b": {
        "loader": load_vit_b_16_clip_dfn2b,
        "architecture": "vit_b16",
        "training_objective": "clip",
        "training_dataset": "dfn2b",
        "training_objective_group": "language-supervised large",
    },
    "vit_b16_clip_metaclip_400m": {
        "loader": load_vit_b_16_clip_metaclip_400m,
        "architecture": "vit_b16",
        "training_objective": "clip",
        "training_dataset": "metaclip_400m",
        "training_objective_group": "language-supervised medium",
    },
    "vit_b16_clip_metaclip_2p5b": {
        "loader": load_vit_b_16_clip_metaclip_2p5b,
        "architecture": "vit_b16",
        "training_objective": "clip",
        "training_dataset": "metaclip_2p5b",
        "training_objective_group": "language-supervised large",
    },
    "vit_b16_siglip_webli": {
        "loader": load_vit_b_16_siglip_webli,
        "architecture": "vit_b16",
        "training_objective": "siglip",
        "training_dataset": "webli",
        "training_objective_group": "language-supervised large",
    },
    "vit_b16_siglip2_webli": {
        "loader": load_vit_b_16_siglip2_webli,
        "architecture": "vit_b16",
        "training_objective": "siglip2",
        "training_dataset": "webli",
        "training_objective_group": "language-supervised large",
    },
    "vit_b16_eva02_clip": {
        "loader": load_vit_b_16_eva02_clip_merged2b,
        "architecture": "eva02_b16",
        "training_objective": "mim+clip",
        "training_dataset": "merged2b",
        "training_objective_group": "language-supervised large",
    },
    "resnet50_sup_in1k": {
        "loader": load_resnet50_sup_in1k,
        "architecture": "resnet50",
        "training_objective": "supervised",
        "training_dataset": "in1k",
        "training_objective_group": "supervised classification",
    },
    "convnext_base_sup_in1k": {
        "loader": load_convnext_base_sup_in1k,
        "architecture": "convnext_base",
        "training_objective": "supervised",
        "training_dataset": "in1k",
        "training_objective_group": "supervised classification",
    },
    "swin_base_sup_in1k": {
        "loader": load_swin_base_sup_in1k,
        "architecture": "swin_base",
        "training_objective": "supervised",
        "training_dataset": "in1k",
        "training_objective_group": "supervised classification",
    },
    "regnety_032_sup_in1k": {
        "loader": load_regnety_032_sup_in1k,
        "architecture": "regnety_032",
        "training_objective": "supervised",
        "training_dataset": "in1k",
        "training_objective_group": "supervised classification",
    },
    "resnet50_ssl_yfcc100m": {
        "loader": load_resnet50_ssl_yfcc100m,
        "architecture": "resnet50",
        "training_objective": "semi-supervised",
        "training_dataset": "yfcc100m_ft_in1k",
        "training_objective_group": "weak / semi supervised",
    },
    "resnet50_clip_yfcc15m": {
        "loader": load_resnet50_clip_yfcc15m,
        "architecture": "resnet50",
        "training_objective": "clip",
        "training_dataset": "yfcc15m",
        "training_objective_group": "language-supervised small",
    },
    "resnet50_swsl_ig1b": {
        "loader": load_resnet50_swsl_ig1b,
        "architecture": "resnet50",
        "training_objective": "weakly-supervised",
        "training_dataset": "ig1b_ft_in1k",
        "training_objective_group": "weak / semi supervised",
    },
    "resnet50_clip_openai": {
        "loader": load_resnet50_clip_openai,
        "architecture": "resnet50",
        "training_objective": "clip",
        "training_dataset": "wit400m",
        "training_objective_group": "language-supervised medium",
    },
    "resnet50_clip_cc12m": {
        "loader": load_resnet50_clip_cc12m,
        "architecture": "resnet50",
        "training_objective": "clip",
        "training_dataset": "cc12m",
        "training_objective_group": "language-supervised small",
    },
    "resnet50_robust_l2_eps3": {
        "loader": load_resnet50_robust_l2_eps3,
        "architecture": "resnet50",
        "training_objective": "adversarial-l2-eps3",
        "training_dataset": "in1k",
        "training_objective_group": "robust",
    },
    "resnet50_toponet_tau30": {
        "loader": load_resnet50_toponet_tau30,
        "architecture": "resnet50",
        "training_objective": "supervised+topoloss",
        "training_dataset": "in1k",
        "training_objective_group": "topographic",
    },
    "resnet18_toponets_tau10": {
        "loader": load_resnet18_toponets_tau10,
        "architecture": "resnet18",
        "training_objective": "supervised+topoloss",
        "training_dataset": "in1k",
        "training_objective_group": "topographic",
    },
    "vit_b14_dino_v2": {
        "loader": load_vit_b_14_dino_v2,
        "architecture": "vit_b14",
        "training_objective": "dinov2",
        "training_dataset": "lvd142m",
        "training_objective_group": "self-supervised",
    },
    "webssl_dino300m": {
        "loader": load_webssl_dino300m,
        "architecture": "vit_l14",
        "training_objective": "dinov2",
        "training_dataset": "web2b_uncurated",
        "training_objective_group": "self-supervised",
    },
    "webssl_mae300m": {
        "loader": load_webssl_mae300m,
        "architecture": "vit_l16",
        "training_objective": "mae",
        "training_dataset": "web2b_uncurated",
        "training_objective_group": "self-supervised",
    },
    "aimv2_large": {
        "loader": load_aimv2_large,
        "architecture": "vit_l14",
        "training_objective": "multimodal-autoregressive",
        "training_dataset": "apple_data_mix",
        "training_objective_group": "language-supervised large",
    },
    "beit_base_in21k": {
        "loader": load_beit_base_in21k,
        "architecture": "beit_base",
        "training_objective": "beit-mim+supervised",
        "training_dataset": "in22k_ft_in22k",
        "training_objective_group": "supervised classification",
    },
    "vit_b16_random": {
        "loader": load_vit_b_16_random,
        "architecture": "vit_b16",
        "training_objective": "random",
        "training_dataset": "none",
        "training_objective_group": "untrained",
    },
    "rn50_random": {
        "loader": load_rn50_random,
        "architecture": "resnet50",
        "training_objective": "random",
        "training_dataset": "none",
        "training_objective_group": "untrained",
    },
    "arcface_r50_webface600k": {
        "loader": load_arcface_r50_webface600k,
        "architecture": "resnet50",
        "training_objective": "face-identification-arcface",
        "training_dataset": "webface600k",
        "training_objective_group": "face recognition",
    },
    "arcface_r50_ms1mv3": {
        "loader": load_arcface_r50_ms1mv3,
        "architecture": "resnet50",
        "training_objective": "face-identification-arcface",
        "training_dataset": "ms1mv3",
        "training_objective_group": "face recognition",
    },
    "resnet50_vggface2_official": {
        "loader": load_resnet50_vggface2_official,
        "architecture": "resnet50",
        "training_objective": "face-identification-softmax",
        "training_dataset": "vggface2",
        "training_objective_group": "face recognition",
    },
    "facenet_vggface2": {
        "loader": load_facenet_vggface2,
        "architecture": "inception_resnet_v1",
        "training_objective": "face-identification",
        "training_dataset": "vggface2",
        "training_objective_group": "face recognition",
    },
    "facenet_casia_webface": {
        "loader": load_facenet_casia_webface,
        "architecture": "inception_resnet_v1",
        "training_objective": "face-identification",
        "training_dataset": "casia_webface",
        "training_objective_group": "face recognition",
    },
    "vit_l_face_cosface_webface42m": {
        "loader": load_vit_l_face_cosface_webface42m,
        "architecture": "vit_large_face",
        "training_objective": "face-identification-cosface",
        "training_dataset": "webface42m",
        "training_objective_group": "face recognition",
    },
    "farl_vitb16_laionface": {
        "loader": load_farl_vitb16_laionface,
        "architecture": "vit_b16",
        "training_objective": "image-text-contrastive+masked-image-modeling",
        "training_dataset": "laion_face20m",
        "training_objective_group": "face recognition",
    },
    "samvit_base_sa1b": {
        "loader": load_samvit_base_sa1b,
        "architecture": "sam_vit_b16",
        "training_objective": "promptable-segmentation",
        "training_dataset": "sa1b",
        "training_objective_group": "segmentation",
    },
    "convnext_base_sup_in21k": {
        "loader": load_convnext_base_sup_in21k,
        "architecture": "convnext_base",
        "training_objective": "supervised",
        "training_dataset": "in21k",
        "training_objective_group": "supervised classification",
    },
    "convnextv2_base_fcmae": {
        "loader": load_convnextv2_base_fcmae,
        "architecture": "convnextv2_base",
        "training_objective": "fcmae",
        "training_dataset": "in1k",
        "training_objective_group": "self-supervised",
    },
    "convnext_base_clip_laion2b": {
        "loader": load_convnext_base_clip_laion2b,
        "architecture": "convnext_base",
        "training_objective": "clip",
        "training_dataset": "laion2b",
        "training_objective_group": "language-supervised large",
    },
    "convnext_base_clip_laiona": {
        "loader": load_convnext_base_clip_laiona,
        "architecture": "convnext_base",
        "training_objective": "clip",
        "training_dataset": "laion_aesthetic",
        "training_objective_group": "language-supervised medium",
    },
    "convnext_large_clip_laion2b": {
        "loader": load_convnext_large_clip_laion2b,
        "architecture": "convnext_large_mlp",
        "training_objective": "clip",
        "training_dataset": "laion2b",
        "training_objective_group": "language-supervised large",
    },
    "regnety_320_seer": {
        "loader": load_regnety_320_seer,
        "architecture": "regnety_320",
        "training_objective": "swav-ssl",
        "training_dataset": "ig1b",
        "training_objective_group": "self-supervised",
    },
    "regnety_160_swag": {
        "loader": load_regnety_160_swag,
        "architecture": "regnety_160",
        "training_objective": "weakly-supervised",
        "training_dataset": "ig3.6b",
        "training_objective_group": "weak / semi supervised",
    },
    "resnet50x4_clip_openai": {
        "loader": load_resnet50x4_clip_openai,
        "architecture": "resnet50x4",
        "training_objective": "clip",
        "training_dataset": "wit400m",
        "training_objective_group": "language-supervised medium",
    },
    "inception_resnet_v2_sup_in1k": {
        "loader": load_inception_resnet_v2_sup_in1k,
        "architecture": "inception_resnet_v2",
        "training_objective": "supervised",
        "training_dataset": "in1k",
        "training_objective_group": "supervised classification",
    },
    "inception_resnet_v2_adv_in1k": {
        "loader": load_inception_resnet_v2_adv_in1k,
        "architecture": "inception_resnet_v2",
        "training_objective": "adversarial-ens",
        "training_dataset": "in1k",
        "training_objective_group": "robust",
    },
}


def get_single_model(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    local_checkpoint_path: str = None,
):
    kwargs = {"local_checkpoint_path": local_checkpoint_path} if local_checkpoint_path else {}
    preprocessor, model = full_model_registry[model_name]["loader"](
        dtype=dtype, device=device, **kwargs
    )
    return {"preprocessor": preprocessor, "model": model}


class ModelRegistryManager:
    """Maintain manager to control for stuff like checkpoints and model loading."""
    
    def __init__(self, local_model_checkpoints: dict = None):
        if local_model_checkpoints is None:
            
            print("\033[38;5;208m"
                "No local model checkpoints provided. Will skip toponet models (resnet50_toponet_tau30, resnet18_toponets_tau10), "
                "arcface models (arcface_r50_webface600k, arcface_r50_ms1mv3), resnet50_vggface2_official, and resnet50_robust_l2_eps3." \
                "For instructions on how to download please check the README."
                "\033[0m")

        self.local_model_checkpoints = local_model_checkpoints or {}
        assert isinstance(
            self.local_model_checkpoints, dict
        ), f"local_model_checkpoints must be a dict, got {type(self.local_model_checkpoints).__name__}"

    def get_model(
        self,
        model_name: str,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda:0",
    ):
        return get_single_model(
            model_name,
            dtype=dtype,
            device=device,
            local_checkpoint_path=self.local_model_checkpoints.get(model_name),
        )

    def __repr__(self):
        return f"ModelRegistryManager(checkpoints={list(self.local_model_checkpoints)})"


def family(model_name: str) -> str:
    """Training family used to group and colour models in every figure (colours: config.grouping_colors)."""
    return full_model_registry[model_name]["training_objective_group"]
