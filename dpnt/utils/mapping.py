import os
import re
from functools import partial
from typing import Optional


subjects = ("Bob", "Dan", "Jim", "Matt", "Nate", "Zack")


# Matched against "_"-split tokens, not as substrings: "Bottomlit" must not
# read as "bottom", but the bare "_L" / "_ri" suffixes must still resolve.
view_tokens = {
    "left": ("left", "l"),
    "right": ("right", "r", "ri"),
    "front": ("center", "front", "frontal", "bottom"),
}


longform_code_views = {
    "OE": "front",
    "HU": "front",
    "S": "front",
}


longform_profile_side = {
    ("Bob", "IF"): "right",
    ("Bob", "P"): "left",
    ("Dan", "IF"): "right",
    ("Dan", "P"): "left",
    ("Jim", "IF"): "right",
    ("Jim", "P"): "right",
    ("Matt", "IF"): "right",
    ("Matt", "P"): "left",
    ("Nate", "IF"): "right",
    ("Nate", "P"): "left",
    ("Zack", "IF"): "right",
    ("Zack", "P"): "right",
}


def resolve_view(label: str) -> str:
    # Leftmost token wins: "Dan_Right_Frontal" is Dan's right view, and the
    # trailing "Frontal" describes the lighting.
    view = next(
        (
            canonical
            for token in label.strip().split("_")
            for canonical, tokens in view_tokens.items()
            if token.lower() in tokens
        ),
        None,
    )
    if view is not None:
        return view

    code = label.strip().split("_")[-1]
    if code in longform_code_views:
        return longform_code_views[code]

    return "other"


def resolve_subject(label: str, candidates=subjects) -> str:
    initial = label.strip()[:1].upper()
    matches = sorted({c for c in candidates if c[:1].upper() == initial})
    assert len(matches) == 1, (
        f"Could not uniquely determine subject from label: {label} "
        f"(initial {initial!r} matched {matches})"
    )
    return matches[0]


aus_pose_views = {
    "L": "left",
    "R": "right",
    "F": "front",  # learn block: the three studied views
    "30L": "left",
    "60L": "left",  # left profiles
    "30R": "right",
    "60R": "right",
    "30RHS": "right",  # right profiles
    "HLF": "front",
    "FBL": "front",  # half-lit / bottom-lit frontal
    "FHS": "front",
    "FLS": "front",  # frontal with high / low side-shadow
}


aus_stim_re = re.compile(r"^(Learn|Novel|Noise)(.*)A(\d{3})(?:_T\d+)?$")


def aus_subject_mapping(label: str, candidates=None) -> tuple[str, str]:
    # CFMT-Australia identities are 3-digit codes (004, 011, ...). The facebank
    # holds the same three studied views per identity as the other CFMT tests,
    # so resolve the viewpoint here for the held-out-view lookup.
    match = aus_stim_re.match(label.strip())
    assert match is not None, f"Could not parse AUS stim name: {label!r}"
    pose, identity = match.group(2), match.group(3)
    if candidates is not None:
        assert (
            identity in candidates
        ), f"AUS identity {identity!r} (from {label!r}) not in facebank {sorted(candidates)}"
    view = aus_pose_views.get(pose)
    assert view is not None, f"Unknown AUS pose {pose!r} in {label!r}"
    return identity, view


def get_subject_mapping(label: str, candidates=subjects) -> tuple[str, str]:
    subject = resolve_subject(label, candidates)
    view = resolve_view(label)

    if view == "other":
        code = label.strip().split("_")[-1]
        override = longform_profile_side.get((subject, code))
        if override is not None:
            view = override

    return subject, view


def normalize_stimuli_name(stimuli: str) -> str:
    # Source stim names are inconsistent (e.g. "Dan_Right_Frontal",
    # "Matt_left_cropped"). Normalize to "{Name}_{Pose}.jpg".
    person = stimuli.strip().split("_")[0].capitalize()
    return f"{person}_{resolve_view(stimuli)}.jpg"


def normalize_aus_stimuli_name(stimuli: str) -> Optional[str]:
    # CFMT-Australia learn images are "Learn{L,F,R}A{ddd}" -- the three studied
    # views per identity, mirrored into the facebank as "{ddd}_{view}.jpg" so
    # the held-out-view lookup works the same as the other CFMT tests.
    match = re.match(r"Learn([LFR])A(\d{3})$", stimuli.strip())
    if match is None:
        return None
    view = {"L": "left", "F": "front", "R": "right"}[match.group(1)]
    return f"{match.group(2)}_{view}.jpg"


def test_mappers(test_type: str, answer_key: str = None):
    valid_tests = ["cfmt", "cfmt_aus", "cfmt_long"]
    assert (
        test_type in valid_tests
    ), f"unknown test_type {test_type!r}, expected one of {valid_tests}"
    mappers = {
        "cfmt": (get_subject_mapping, normalize_stimuli_name),
        "cfmt_long": (get_subject_mapping, normalize_stimuli_name),
        "cfmt_aus": (aus_subject_mapping, normalize_aus_stimuli_name),
    }
    subject_mapper, name_normalizer = mappers[test_type]
    if test_type in ("ccmt", "cbmt"):
        assert answer_key is not None, f"{test_type} needs its answer_key csv"
        subject_mapper = partial(subject_mapper, answer_key=answer_key)
    return subject_mapper, name_normalizer
