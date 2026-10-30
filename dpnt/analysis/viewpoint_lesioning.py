from itertools import combinations

import numpy as np
import torch
from einops import rearrange


######################################################################
#
# Per-unit viewpoint / identity indices
#
######################################################################


def view_invariance_pairwise(activations, n=3):
    """
    Per-unit viewpoint invariance (the GLM / paper definition):
    -(mean z-scored distance between the same identity in different views).

    activations     [identity, view, d] activations
    n               identities per subsample
    returns         [d] invariance per unit (higher = more invariant)

    For every n-identity subsample, take each image pair's distance per unit.
    Pool one mean and sd per unit over all pairs of all subsamples, z-score the
    same-identity pairs, average them, and negate.

    Distance is |a - b|: cosine distance between two single numbers would only
    compare their signs.
    """
    n_id, n_view, dim = activations.shape

    # every image of an n-identity subsample, as (identity, view) positions
    images = [(identity, view) for identity in range(n) for view in range(n_view)]
    image_pairs = [
        (a, b) for a in range(len(images)) for b in range(a + 1, len(images))
    ]

    # same identity => different views
    same_identity = np.array([images[a][0] == images[b][0] for a, b in image_pairs])
    image_pairs = np.array(image_pairs)  # [pairs, 2]

    # per-unit distance of every image pair, in every subsample
    distances = []
    for subsample in combinations(range(n_id), n):
        subsample_acts = rearrange(
            activations[list(subsample)], "id view d -> (id view) d"
        )  # [n * n_view, d]
        distances.append(
            np.abs(
                subsample_acts[image_pairs[:, 0]] - subsample_acts[image_pairs[:, 1]]
            )
        )
    distances = np.stack(distances)  # [subsample, pair, d]

    # z-score with one mean / sd per unit, pooled over all pairs of all subsamples
    pooled = distances.reshape(-1, dim)  # [subsample * pair, d]
    pooled_sd = pooled.std(0)
    z = (distances - pooled.mean(0)) / np.where(pooled_sd > 0, pooled_sd, np.nan)

    # mean over same-identity pairs, then over subsamples; negate so higher = more invariant
    return -z[:, same_identity].mean(1).mean(0)


def unit_order_viewpoint(fit, vi="pairwise"):
    """
    Three unit rankings from an [identity, view, d] array, as unit orders, highest first.

    view-dependent   between-view variance (his "viewpoint-tuned")
    identity-tuned   between-identity variance
    view-invariant   view_invariance_pairwise (his "viewpoint-tolerant"),
                     or with vi="ratio" his between-identity / between-view variance
    """
    view_var = fit.mean(axis=0).var(axis=0, ddof=1)  # [d] variance across view means
    id_var = fit.mean(axis=1).var(axis=0, ddof=1)  # [d] variance across identity means

    if vi == "ratio":
        view_inv = id_var / (view_var + 1e-12)
    else:
        view_inv = view_invariance_pairwise(fit)

    return {
        "view-dependent": np.argsort(-view_var),
        "identity-tuned": np.argsort(-id_var),
        "view-invariant": np.argsort(
            -np.nan_to_num(view_inv, nan=-np.inf)
        ),  # NaN units last
    }


def unit_corr(a, b):
    """Per-unit Pearson r over the first axis."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a**2).sum(0) * (b**2).sum(0))


def cross_view_r(cell):
    """
    Mean correlation of each unit's identity profile between each pair of views.

    cell     [ids, 3 views, d]
    returns  [d]
    """
    left, front, right = cell[:, 0], cell[:, 1], cell[:, 2]
    return np.nanmean(
        [
            unit_corr(left, front),
            unit_corr(right, front),
            unit_corr(left, right),
        ],
        0,
    )


######################################################################
#
# Per-unit view properties, pooled and per test
#
######################################################################


def view_properties(view_acts, tags=("aus", "reg"), views=("left", "front", "right")):
    """
    Per-unit viewpoint and identity properties, from z-scored activations.

    view_acts  list over tests of (acts [N, d], identity labels, view labels),
               in the order of `tags`

    view_tol   log10(between-identity / between-view variance of the identity x view
               cell means)  (view tolerant / invariant: high)
    eta2_view  share of the unit's variance explained by view  (view dependent: high)
    eta2_id    share explained by identity  (identity selective: high)
    xview_r    correlation of the unit's identity profile across views  (view invariant: high)

    Pooled over the tests, and per test as {name}_{tag}, so a property measured on one
    test can be related to importance on another (disjoint identities and images).
    """
    id_vars, view_vars, eta2_views, eta2_ids, profiles = [], [], [], [], []

    for acts, ids, test_views in view_acts:
        acts = acts.numpy() if torch.is_tensor(acts) else np.asarray(acts)

        # z-score each unit over this test's images
        sd = acts.std(0)
        z = (acts - acts.mean(0)) / np.where(sd > 0, sd, np.nan)  # [N, d]

        # identity x view cell means
        cell_means = np.stack(
            [
                np.stack(
                    [
                        z[(ids == identity) & (test_views == view)].mean(0)
                        for view in views
                    ]
                )
                for identity in sorted(set(ids))
            ]
        )  # [ids, views, d]
        id_vars.append(
            cell_means.mean(1).var(0, ddof=1)
        )  # [d] variance across identities
        view_vars.append(cell_means.mean(0).var(0, ddof=1))  # [d] variance across views

        # eta^2: between-group sum of squares / total sum of squares
        total_ss = (z**2).sum(0)  # [d]
        view_mean_per_image = np.stack(
            [z[test_views == view].mean(0) for view in test_views]
        )  # [N, d]
        identity_mean_per_image = np.stack(
            [z[ids == identity].mean(0) for identity in ids]
        )  # [N, d]
        eta2_views.append((view_mean_per_image**2).sum(0) / total_ss)
        eta2_ids.append((identity_mean_per_image**2).sum(0) / total_ss)

        # identity profiles with the test / view offset removed
        profiles.append(cell_means - cell_means.mean(0, keepdims=True))

    # pooled over tests
    properties = {
        "view_tol": np.log10(
            (np.mean(id_vars, 0) + 1e-12) / (np.mean(view_vars, 0) + 1e-12)
        ),
        "eta2_view": np.mean(eta2_views, 0),
        "eta2_id": np.mean(eta2_ids, 0),
        "xview_r": cross_view_r(np.concatenate(profiles)),
    }

    # per test
    for test_idx, tag in enumerate(tags):
        properties[f"view_tol_{tag}"] = np.log10(
            (id_vars[test_idx] + 1e-12) / (view_vars[test_idx] + 1e-12)
        )
        properties[f"eta2_view_{tag}"] = eta2_views[test_idx]
        properties[f"eta2_id_{tag}"] = eta2_ids[test_idx]
        properties[f"xview_r_{tag}"] = cross_view_r(profiles[test_idx])

    return properties


######################################################################
#
# Lesion sets ranked by those properties
#
######################################################################


def property_lesion_masks(props, n_units, rankings, fractions, n_random):
    """
    Embedding-level lesion sets: the top `fraction` of units under each ranking.

    props      DataFrame of per-unit properties, one row per unit
    rankings   {name: property column(s) of `props`}, highest first, later columns break ties
    fractions  fractions of units to lesion
    n_random   number of random unit orders added as controls

    returns    bool [lesions, units] masks, and one
               {"ranking", "seed", "frac", "k"} label per row (seed = -1 for non-random)
    """
    # unit orders, highest first
    unit_orders = {
        name: props.sort_values(
            cols, ascending=False, na_position="last", kind="stable"
        ).index.to_numpy()
        for name, cols in rankings.items()
    }
    for seed in range(n_random):
        unit_orders[("random", seed)] = np.random.default_rng(seed).permutation(n_units)

    masks, labels = [], []
    for order_key, order in unit_orders.items():
        ranking, seed = order_key if isinstance(order_key, tuple) else (order_key, -1)

        for frac in fractions:
            n_lesioned = int(round(frac * n_units))
            mask = np.zeros(n_units, dtype=bool)
            mask[order[:n_lesioned]] = True

            masks.append(mask)
            labels.append(
                {"ranking": ranking, "seed": seed, "frac": frac, "k": n_lesioned}
            )

    return torch.from_numpy(np.stack(masks)), labels
