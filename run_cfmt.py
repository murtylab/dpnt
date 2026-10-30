import argparse

import torch
from dpnt.run_cfmt import run_all_models_cfmt
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_csv",
    help="trial-level results csv (appended to unless --fresh)",
)
parser.add_argument(
    "--emb_dir",
    help="where per-model stimulus embeddings are saved",
)
parser.add_argument(
    "--models", 
    nargs="+", 
    default=None, 
    help="default: every model"
)
parser.add_argument(
    "--device", help="device to run on"
)
parser.add_argument(
    "--cfmt_csv_filepath", 
    default=None, 
    help="path to cfmt.csv"
)
parser.add_argument(
    "--cfmt_stimuli_filepath", 
    default=None, 
    help="path to cfmt stimuli"
)
parser.add_argument(
    "--facebank_dir", 
    default=None, 
    help="path to facebank"
)
parser.add_argument(
    "--crop",
    default="pad-square",
    help="how to crop the stimuli (default is pad-square)",
)
parser.add_argument(
    "--heldout",
    action="store_true",
    default=True,
    help="whether to use keep the test face perspective heldout from facebank in trial (default is True)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
)
parser.add_argument(
    "--dtype",
    choices=["float32", "float16", "bfloat16"],
    default="float32",
    help="model dtype",
)

args = parser.parse_args()

cortex_backbone_dir = "/research/cortex-paper/backbones"
toponets_checkpoint_dir = "/research/mayukh/toponets_resnet50_imagenet_checkpoints"
robust_rn50_l2_eps3_ckpt = os.path.join(cortex_backbone_dir, "robust_rn50_epsilon_3.0.pth")
toponet_rn50_tau30_ckpt = os.path.join(
    toponets_checkpoint_dir, "all_topo_scale_30_shrink_factor_3.0", "final_weights.pt"
)
toponets_rn50_tau1_shrink3_ckpt = os.path.join(
    toponets_checkpoint_dir, "all_topo_scale_1_shrink_factor_3.0", "final_weights.pt"
)
vggface2_resnet50_ckpt = os.path.join(cortex_backbone_dir, "resnet50_vggface2.pth")
arcface_r50_ms1mv3_onnx = os.path.join(cortex_backbone_dir, "ms1mv3_arcface_r50.onnx")
arcface_onnx_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
toponet_rn18_tau10_ckpt = os.path.join(cortex_backbone_dir, "resnet18_tau_10.pt")

local_model_checkpoints = {
    "resnet50_robust_l2_eps3": robust_rn50_l2_eps3_ckpt,
    "resnet50_toponet_tau30": toponet_rn50_tau30_ckpt,
    "resnet18_toponets_tau10": toponet_rn18_tau10_ckpt,
    "arcface_r50_webface600k": arcface_onnx_path,
    "arcface_r50_ms1mv3": arcface_r50_ms1mv3_onnx,
    "resnet50_vggface2_official": vggface2_resnet50_ckpt,
}

args = parser.parse_args()

run_all_models_cfmt(
    test_type="cfmt",
    results_csv=args.results_csv,
    csv_filepath=args.cfmt_csv_filepath,
    stimuli_dir=args.cfmt_stimuli_filepath,
    facebank_dir=args.facebank_dir,
    crop=args.crop,
    embeddings_dir=args.emb_dir,
    model_names=args.models,
    heldout=args.heldout,
    seed=args.seed,
    dtype=getattr(torch, args.dtype),
    device=args.device,
    local_model_checkpoints=local_model_checkpoints,
)
