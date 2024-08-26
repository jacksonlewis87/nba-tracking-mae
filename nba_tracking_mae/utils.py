import json
import os
from torch import load as torch_load


def load_json(path: str):
    with open(path, "r") as f:
        json_file = json.load(fp=f)

    return json_file


def write_json(path: str, json_object):
    with open(path, "w") as f:
        json.dump(json_object, fp=f)


def list_files_in_directory(path: str, suffix: str = None):
    return [
        f[: -len(suffix)] if suffix is not None else f
        for f in os.listdir(path)
        if suffix is None or f[-len(suffix) :] == suffix
    ]


def load_tensor(path: str, tensor_name: str):
    return torch_load(f"{path}/{tensor_name}.pt")
