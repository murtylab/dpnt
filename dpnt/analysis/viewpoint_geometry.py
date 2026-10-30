from itertools import combinations, permutations
from math import comb

import numpy as np
import pandas as pd
from einops import rearrange
from scipy.stats import f as f_dist
from sklearn.manifold import MDS

from .viewpoint_lesioning import unit_order_viewpoint


######################################################################
#
# Identity separation
#
######################################################################


def cosine_matrix(feats):
    """
    Pairwise cosine similarity.

    feats    [N, d]
    returns  [N, N]
    """
    norm = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.where(norm > 0, norm, 1.0)
    return feats @ feats.T


def identity_separation_d(feats, ids):
    """
    Identity separation d: within- minus between-identity cosine similarity,
    in pooled sd units.

    feats    [N, d] embeddings
    ids      [N] identity label of each image
    returns  float (NaN if the pooled sd is zero)
    """
    sim = cosine_matrix(feats)
    i, j = np.triu_indices(len(feats), 1)
    same = ids[i] == ids[j]
    within, between = sim[i, j][same], sim[i, j][~same]

    pooled_sd = np.sqrt((within.var(ddof=1) + between.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return np.nan
    return float((within.mean() - between.mean()) / pooled_sd)


######################################################################
#
# Viewpoint / identity subspaces
#
######################################################################


def basis_from_means(means):
    """
    Orthonormal basis of the span of the centred group means.

    means    [groups, d]
    returns  [rank, d]

    g centred means span at most g - 1 directions, so the rank is capped there.
    The tolerance is float32-scaled: a float32 null direction has a singular value
    near 1e-7 of the largest rather than zero, and would otherwise be kept.
    """
    centred = (means - means.mean(0, keepdims=True)).astype(np.float64)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    if sv.max() <= 0:
        return np.zeros((0, means.shape[1]))

    keep = sv > 1e-6 * sv.max()
    keep[len(means) - 1 :] = False
    return vt[keep]


def view_basis(activations):
    """Viewpoint subspace: means over identities, so identity averages out. [rank, d]"""
    return basis_from_means(activations.mean(axis=0))


def identity_basis(activations):
    """Identity subspace: means over views, so viewpoint averages out. [rank, d]"""
    return basis_from_means(activations.mean(axis=1))


def project_out(feats, basis):
    """Orthogonal projection of feats [N, d] onto the complement of span(basis [rank, d])."""
    if len(basis) == 0:
        return feats.copy()
    return feats - (feats @ basis.T) @ basis


def ablate_units(feats, units):
    """Mean-impute `units` of feats [N, d]: their information goes, the scale stays."""
    ablated = feats.copy()
    if len(units):
        ablated[:, units] = feats[:, units].mean(0, keepdims=True)
    return ablated


def facebank_geometry(activations):
    """
    The facebank embeddings as is, with the viewpoint subspace removed, and with the
    identity subspace removed (in-sample).

    activations  [identity, view, d]
    returns      {name: [identity * view, d]}
    """
    feats = rearrange(activations, "id view d -> (id view) d")
    return {
        "original": feats,
        "view removed": project_out(feats, view_basis(activations)),
        "identity removed": project_out(feats, identity_basis(activations)),
    }


def embedding_mds(feats, seed=0):
    """2D MDS of the cosine distances between embeddings. [N, d] -> [N, 2]"""
    distance = np.clip(1 - cosine_matrix(feats), 0, None)
    np.fill_diagonal(distance, 0)
    return MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=4,
        normalized_stress="auto",
    ).fit_transform(distance)


######################################################################
#
# Fit / test identity splits
#
######################################################################


def identity_splits(n_id, n_fit, max_splits=60, seed=0):
    """
    Fit identity sets: every n_fit-subset of the identities, or a fixed-seed random
    subset of max_splits of them when there are more than that.
    """
    splits = list(combinations(range(n_id), n_fit))
    if comb(n_id, n_fit) <= max_splits:
        return splits

    keep = np.random.default_rng(seed).choice(len(splits), max_splits, replace=False)
    return [splits[i] for i in sorted(keep)]


def split_activations(activations, fit):
    """
    Fit identities of an [identity, view, d] array, and the held-out ones as images.

    returns  fit activations [fit id, view, d], test features [test id * view, d],
             test identity labels [test id * view]
    """
    n_id, n_view, _ = activations.shape
    test = [i for i in range(n_id) if i not in fit]
    assert len(test) >= 2, "identity separation needs at least 2 held-out identities"

    test_feats = rearrange(activations[test], "id view d -> (id view) d")
    test_ids = np.repeat(np.arange(len(test)), n_view)
    return activations[list(fit)], test_feats, test_ids


######################################################################
#
# Viewpoint plane removal and unit ablation, on held-out identities
#
######################################################################


def viewplane_removal(activations, seed=0):
    """
    Identity separation d of held-out identities before and after projecting out
    the viewpoint subspace of the fit identities, one row per identity split.

    activations  [identity, view, d]
    returns      DataFrame {split, before, after}
    """
    n_id = activations.shape[0]

    rows = []
    for split_i, fit in enumerate(identity_splits(n_id, n_id // 2, seed=seed)):
        fit_acts, test_feats, test_ids = split_activations(activations, fit)
        after = project_out(test_feats, view_basis(fit_acts))
        rows.append(
            {
                "split": split_i,
                "before": identity_separation_d(test_feats, test_ids),
                "after": identity_separation_d(after, test_ids),
            }
        )

    return pd.DataFrame(rows)


def ablation_specificity(
    activations,
    ks: list = [1, 2, 5, 10, 20, 50, 100, 200],  # unit counts ablated in the k sweep
    vi: str = "pairwise",
    n_random: int = 40,
    seed: int = 0,
):

    k_grid = [1, 2, 5, 10, 20, 50, 100, 200]
    """
    Change in identity separation d of held-out identities from ablating the top-k
    units of each ranking (fit on the other identities), minus the mean change from
    ablating k random units.

    activations  [identity, view, d]
    ks           unit counts to ablate (those >= d are skipped)
    vi           view-invariant ranking, see unit_order_viewpoint
    returns      DataFrame {split, index, k, baseline, targeted_change, random_change,
                 specificity}, with index as named by unit_order_viewpoint
    """
    n_id, _, dim = activations.shape

    rows = []
    for split_i, fit in enumerate(identity_splits(n_id, n_id // 2, seed=seed)):
        fit_acts, test_feats, test_ids = split_activations(activations, fit)
        baseline = identity_separation_d(test_feats, test_ids)
        orders = unit_order_viewpoint(fit_acts, vi)

        for k in [k for k in ks if k < dim]:
            # one random control per (split, k), shared by the rankings
            rng = np.random.default_rng([seed, split_i, k])
            random_change = (
                np.mean(
                    [
                        identity_separation_d(
                            ablate_units(test_feats, rng.choice(dim, k, replace=False)),
                            test_ids,
                        )
                        for _ in range(n_random)
                    ]
                )
                - baseline
            )

            for index, order in orders.items():
                targeted_change = (
                    identity_separation_d(ablate_units(test_feats, order[:k]), test_ids)
                    - baseline
                )
                rows.append(
                    {
                        "split": split_i,
                        "index": index,
                        "k": k,
                        "baseline": baseline,
                        "targeted_change": targeted_change,
                        "random_change": random_change,
                        "specificity": targeted_change - random_change,
                    }
                )

    return pd.DataFrame(rows)


######################################################################
#
# Model-level indices and the CFMT regression
#
######################################################################


def unit_normalize(activations):
    """Scale every embedding of an [identity, view, d] array to unit length."""
    return activations / np.linalg.norm(activations, axis=2, keepdims=True)


def cosine_distance_matrix(activations):
    """
    Cosine distance between every pair of images.

    activations  [identity, view, d]
    returns      [identity * view, identity * view], image = identity * n_view + view
    """
    feats = rearrange(activations, "id view d -> (id view) d")
    return 1 - cosine_matrix(feats)


def index_distances(distance, subsample, n_view):
    """
    The three kinds of image pair of one identity subsample, as cosine distances.

    distance   [images, images] from cosine_distance_matrix
    subsample  identities of the subsample
    returns    same identity across views, different identities in the same view,
               different identities in different views
    """
    image = lambda identity, view: identity * n_view + view

    same_identity = [
        distance[image(i, a), image(i, b)]
        for i in subsample
        for a, b in combinations(range(n_view), 2)
    ]
    same_view = [
        distance[image(i, v), image(j, v)]
        for i, j in combinations(subsample, 2)
        for v in range(n_view)
    ]
    cross = [
        distance[image(i, a), image(j, b)]
        for i, j in combinations(subsample, 2)
        for a, b in permutations(range(n_view), 2)
    ]
    return np.array(same_identity), np.array(same_view), np.array(cross)


def glm_indices(activations, n=3):
    """
    Model-level viewpoint invariance and identity selectivity.

    activations  [identity, view, d]
    n            identities per subsample
    returns      (viewpoint invariance, identity selectivity)

    Cosine distances of every n-identity subsample are z-scored with one mean and sd
    pooled over all three kinds of pair of all subsamples. Invariance is the negated
    mean z of same-identity pairs across views, selectivity the mean z of
    different-identity pairs in the same view.
    """
    n_id, n_view, _ = activations.shape
    distance = cosine_distance_matrix(activations)

    pairs = [
        index_distances(distance, subsample, n_view)
        for subsample in combinations(range(n_id), n)
    ]
    pooled = np.concatenate([np.concatenate(p) for p in pairs])
    mean, sd = pooled.mean(), pooled.std()

    invariance = -np.mean([((same - mean) / sd).mean() for same, _, _ in pairs])
    selectivity = np.mean([((view - mean) / sd).mean() for _, view, _ in pairs])
    return invariance, selectivity


def variance_indices(activations, n=3):
    """
    Model-level identity-tuned and viewpoint-tuned variance, on unit-length embeddings:
    per-unit between-identity and between-view variance of each n-identity subsample,
    summed over units, then averaged over subsamples.

    activations  [identity, view, d]
    returns      (identity-tuned, viewpoint-tuned)
    """
    activations = unit_normalize(activations)

    variances = [
        (
            activations[list(subsample)].mean(axis=1).var(axis=0, ddof=1).sum(),
            activations[list(subsample)].mean(axis=0).var(axis=0, ddof=1).sum(),
        )
        for subsample in combinations(range(activations.shape[0]), n)
    ]
    return np.mean(variances, axis=0)


def regress_scores(indices, scores):
    """
    Least-squares prediction of scores from model-level indices.

    indices  [models, predictors]
    scores   [models] CFMT score of each model
    returns  predicted scores [models], R^2, and the p-value of the F test against
             the intercept-only model
    """
    n_models, n_predictors = indices.shape
    design = np.column_stack([np.ones(n_models), indices])  # [models, 1 + predictors]
    predicted = design @ np.linalg.lstsq(design, scores, rcond=None)[0]

    r2 = 1 - ((scores - predicted) ** 2).sum() / ((scores - scores.mean()) ** 2).sum()
    dof = n_models - n_predictors - 1
    p = f_dist.sf((r2 / n_predictors) / ((1 - r2) / dof), n_predictors, dof)
    return predicted, r2, p
