import csv
import gc
import os
import traceback

import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
from torch.nn.functional import cosine_similarity

from .models import ModelRegistryManager, full_model_registry
from .utils.seed import seed_everything
from .utils.json import save_json
from .utils.datasets import CFMTDataset, FacebankDataset
from .utils.mapping import test_mappers

result_columns = [
    "model",
    "arch",
    "objective",
    "data",
    "accuracy",
    "trial_idx",
    "identity",
    "view",
    "key",
    "target_idx",
    "chosen_idx",
    "correct",
    "target_score",
    "best_distractor_score",
    "chosen_score",
    "score_face0",
    "score_face1",
    "score_face2",
    "num_ground_truth",
    "embedding_dim",
]



######################################################################
# 
# CFMT run funcs 
#
# 
######################################################################


def run_all_models_cfmt(
    test_type: str,
    results_csv: str,
    csv_filepath: str,
    stimuli_dir: str,
    facebank_dir: str,
    crop,
    answer_key: str = None,
    embeddings_dir: str = None,
    model_names: list = None,
    heldout: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda:0",
    local_model_checkpoints: dict = None,
):
    seed_everything(seed=seed)
    if model_names is None:
        model_names = list(full_model_registry)
    else:
        if not isinstance(model_names, list):
            raise TypeError(f"model_names must be a list, got {type(model_names).__name__}")
        unknown = [m for m in model_names if m not in full_model_registry]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")

    model_registry = ModelRegistryManager(local_model_checkpoints=local_model_checkpoints)
    subject_mapper, name_normalizer = test_mappers(test_type, answer_key)
    
    if embeddings_dir is not None:
        print(f"\033[92mEmbeddings will be saved to {embeddings_dir}\033[0m")
        os.makedirs(embeddings_dir, exist_ok=True)

    skipped = []
    with open(results_csv, "w", newline="") as results_file:

        

        writer = csv.DictWriter(results_file, fieldnames=result_columns)
        writer.writeheader()

        for model_name in tqdm(model_names, desc=f"{test_type} sweep", unit="model"):
            info = full_model_registry[model_name]
            loaded = facebank_dataset = cfmt_dataset = None

            try:
                # load
                model_stuff = model_registry.get_model(model_name, dtype=dtype, device=device)
                preprocessor = model_stuff["preprocessor"]
                model = model_stuff["model"]

                # datasets
                facebank_dataset = FacebankDataset(
                    cfmt_csv_filepath=csv_filepath,
                    cfmt_stimuli_dir=stimuli_dir,
                    facebank_dir=facebank_dir,
                    crop=crop,
                    device=device,
                    dtype=dtype,
                    transform=preprocessor,
                    name_normalizer=name_normalizer,
                )
                cfmt_dataset = CFMTDataset(
                    cfmt_csv_filepath=csv_filepath,
                    cfmt_stimuli_dir=stimuli_dir,
                    crop=crop,
                    transform=preprocessor,
                )

                # run
                rows, accuracy, trial_embeddings, bank_embeddings = run_single_model_cfmt(
                    cfmt_dataset=cfmt_dataset,
                    facebank_dataset=facebank_dataset,
                    model=model,
                    subject_mapper=subject_mapper,
                    device=device,
                    dtype=dtype,
                    heldout=heldout,
                )

                # save facebank + trial embeddings
                if embeddings_dir is not None:
                    save_embeddings_locally(
                        embeddings_dir, 
                        model_name, 
                        trial_embeddings, 
                        bank_embeddings, 
                        facebank_dataset
                    )

                base = {
                    "model": model_name,
                    "arch": info["architecture"],
                    "objective": info["training_objective"],
                    "data": info["training_dataset"],
                    "accuracy": accuracy,
                }
                writer.writerows({**base, **row} for row in rows)
                results_file.flush()  # keep the CSV readable mid-sweep

            except Exception:
                print(f"\033[38;5;208m\nSkipping {model_name}:\033[0m")
                traceback.print_exc()
                skipped.append(model_name)
                exit()

            finally:
                # drop references so the model's GPU memory can be freed
                model_stuff = None
                facebank_dataset = None
                cfmt_dataset = None
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nWrote {results_csv}")
    if skipped:
        print(f"Skipped {len(skipped)} model(s): {skipped}")


######################################################################
#
# One model: embed everything, score every trial
#
######################################################################


@torch.inference_mode()
def run_single_model_cfmt(
    cfmt_dataset,
    facebank_dataset,
    model,
    subject_mapper,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
    heldout: bool = True,
    batch_size: int = 64,
):
    """Score every CFMT trial against the target's study faces.

    Returns (rows, accuracy, test_embeddings, study_embeddings):
      rows              [per-trial dict]
      accuracy          fraction of trials answered correctly
      test_embeddings   [n_trials, 3, D] numpy
      study_embeddings  [n_study_faces, D] numpy
    """
    trials = organize_trials_for_single_model(cfmt_dataset, facebank_dataset, subject_mapper, heldout)

    # embed every test face and every study face
    test_images = torch.cat([t["images"] for t in trials]).to(device=device, dtype=dtype)
    study_images = facebank_dataset.get_all_images().to(device=device, dtype=dtype)
    test_embs_raw = chunk(model, test_images, batch_size)  # [n_trials*3, D]

    test_emb = rearrange(test_embs_raw, "(trial face) d -> trial face d", face=3) # [n_trials, 3, D]
    facebank_emb = chunk(model, study_images, batch_size)  # [n_study_faces, D]

    # score each trial: its 3 test faces vs. the target's study faces
    rows = []
    for trial_idx, (trial, faces) in enumerate(zip(trials, test_emb)):
        cfmt_trial_results = run_single_trial_cfmt(
            faces=faces,
            facebank=facebank_emb[trial["study_rows"]],
            key=trial["key"],

        )
        rows.append({
            "trial_idx": trial_idx,
            "identity": trial["identity"],
            "view": trial["view"],
            **cfmt_trial_results,
        })

    accuracy = sum(r["correct"] for r in rows) / len(rows)

    # to save
    test_emb = test_emb.float().cpu().numpy()
    facebank_emb = facebank_emb.float().cpu().numpy()

    return rows, accuracy, test_emb, facebank_emb


######################################################################
#
# One trial: score the 3 test faces
#
######################################################################


def run_single_trial_cfmt(faces, facebank, key, one_indexed_key=True):
    """
    Each test face's mean cosine similarity to the target identity's facebank images.

    faces    [3, D] the three test faces
    facebank [num_ground_truth, D] the target identity's study faces

    Take the cosine similarity between each of the test faces and the ground truth faces.
    Then average the similarities for each test face to get the final score. 

    (mean of cosine similarities)
    """

    if faces.shape[-1] != facebank.shape[-1]:
        raise ValueError(f"Embedding dims differ: {faces.shape[-1]} vs {facebank.shape[-1]}")
    if faces.shape[0] != 3:
        raise ValueError(f"Expected 3 test faces, got {faces.shape[0]}")

    # float32 so low-precision sweeps (fp16/bf16) don't cause near-ties
    similarity = cosine_similarity(
        faces.float()[:, None, :], 
        facebank.float()[None, :, :], dim=-1
    )  # [3, num_ground_truth]

    face_scores = similarity.mean(dim=-1).cpu().tolist()  # [3]

    # key is 1-indexed if one_indexed_key is True. usually is so!
    target_idx = key - 1 if one_indexed_key else key  
    chosen_idx = face_scores.index(max(face_scores))  # first face wins ties
    best_distractor_score = max(s for i, s in enumerate(face_scores) if i != target_idx)

    return {
        "key": key,
        "target_idx": target_idx,
        "chosen_idx": chosen_idx,
        "correct": int(chosen_idx == target_idx),
        "target_score": face_scores[target_idx],
        "best_distractor_score": best_distractor_score,
        "chosen_score": face_scores[chosen_idx],
        "score_face0": face_scores[0],
        "score_face1": face_scores[1],
        "score_face2": face_scores[2],
        "num_ground_truth": int(facebank.shape[0]),
        "embedding_dim": int(faces.shape[-1]),
    }


######################################################################
# 
# Utils 
# 
######################################################################


def chunk(model, images, batch_size):
    return torch.cat([model(images[i : i + batch_size]) for i in range(0, len(images), batch_size)])


def save_embeddings_locally(emb_dir, model_name, trial_embeddings, bank_embeddings, facebank_dataset):
    np.save(os.path.join(emb_dir, f"{model_name}_trial.npy"), trial_embeddings)
    np.save(os.path.join(emb_dir, f"{model_name}_facebank.npy"), bank_embeddings)

    labels_path = os.path.join(emb_dir, "facebank_labels.json")
    if not os.path.exists(labels_path):
        labels = [
            {"file": fn, "label": lb, "view": vw}
            for fn, lb, vw in zip(facebank_dataset.files, facebank_dataset.labels, facebank_dataset.views)
        ]
        save_json(labels, labels_path)


def organize_trials_for_single_model(cfmt_dataset, facebank_dataset, subject_mapper, heldout):
    """
    One dict per CFMT trial: its 3 test images, answer key, target identity,
    and the facebank rows of that identity's study faces.
    """
    trials = []
    for batch in cfmt_dataset.get_dataloader(shuffle=False):
        images = batch["images"][0]  # [3, C, H, W]
        if len(images) != 3:
            raise ValueError(f"Trial {len(trials)} has {len(images)} faces, expected 3")

        # handle CFMT trial name styles. will need to be changed depending on what dataset style you are using :)
        name, view = subject_mapper(label=batch["label"][0], candidates=facebank_dataset.labels)

        ### NOTE: IMPORTANT! determine which view to exclude based on whether this is a held-out trial
        excluded_view = view if heldout else "all"  # "all" keeps every view

        study_rows = facebank_dataset.facebank_indices(name, excluded_view)
        if not study_rows:
            raise ValueError(f"No study face for {name!r} (held-out view {excluded_view!r})")

        trials.append({
            "images": images,
            "key": batch["key"].item(),
            "identity": name,
            "view": view,
            "study_rows": study_rows,
        })
        
    return trials
