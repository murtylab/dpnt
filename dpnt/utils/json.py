import os

import ujson


def load_json(filename: str):
    assert os.path.exists(filename), f"File {filename} does not exist"
    with open(filename, "r") as f:
        return ujson.load(f)


def save_json(list_or_dict, filename: str):
    with open(filename, "w") as f:
        ujson.dump(list_or_dict, f, indent=4)
