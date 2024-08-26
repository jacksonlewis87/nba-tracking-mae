import os
from random import shuffle

from constants import EVAL_GAME_IDS
from data.mae.data_config import DataConfig
from utils import load_json, write_json


def get_data_split(config: DataConfig, game_ids: list[str], stage: str):
    if stage == "eval":
        return {
            "train": [],
            "val": get_eval_game_ids(game_ids=game_ids),
        }
    elif os.path.isfile(config.data_split_path):
        return load_json(path=config.data_split_path)
    else:
        return create_data_split(config=config, game_ids=game_ids)


def create_data_split(config: DataConfig, game_ids: list[str]):
    game_ids = get_non_eval_game_ids(game_ids=game_ids)

    if len(list(set(game_ids))) != len(game_ids):
        print("Error: duplicate game_ids")
        raise Exception

    shuffle(game_ids)
    data_split = {
        "train": game_ids[: round(len(game_ids) * config.train_size)],
        "val": game_ids[round(len(game_ids) * config.train_size) :],
    }
    write_json(config.data_split_path, data_split)

    return data_split


def get_non_eval_game_ids(game_ids: list[str]):
    game_ids = [game_id for game_id in game_ids if game_id not in EVAL_GAME_IDS]
    return game_ids


def get_eval_game_ids(game_ids: list[str]):
    game_ids = [game_id for game_id in game_ids if game_id in EVAL_GAME_IDS]
    return game_ids
