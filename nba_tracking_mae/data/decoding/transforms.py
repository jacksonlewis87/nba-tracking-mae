import torch
from enum import Enum
from random import randint

from constants import X_MAX
from data.decoding.data_config import DataConfig
from data.transforms import random_crop, shuffle_players, normalize_coordinates, flip_x_axis


class Task(Enum):
    BALL_HEIGHT_CLASSIFICATION = "ball_height_classification"
    MADE_BASKET_CLASSIFICATION = "made_basket_classification"


def get_tracking_transforms_function(task_str: str):
    if task_str == Task.BALL_HEIGHT_CLASSIFICATION.value:
        return tracking_to_ball_height_classification
    elif task_str == Task.MADE_BASKET_CLASSIFICATION.value:
        return tracking_to_made_basket_classification
    else:
        return None


def tracking_to_ball_height_classification(x: torch.tensor, config: DataConfig, stage: str) -> torch.tensor:
    x = random_crop(x=x, length=config.num_frames, dim=1)

    y = (x[0, :, 2] >= config.min_z).long()

    if not config.include_z:
        x = x[:, :, :2]

    if stage != "eval":
        x = shuffle_players(x=x, shuffle_players=config.shuffle_players)
        x = flip_x_axis(x=x)

    x = normalize_coordinates(x=x)

    return x, y


def tracking_to_made_basket_classification(x: torch.tensor, config: DataConfig, stage: str) -> torch.tensor:
    max_crop_index = x.size(dim=1) - config.min_frames
    start_index = randint(0, max_crop_index)
    x = x[:, start_index : config.num_frames + config.y_frames + start_index]

    if stage != "eval":
        x = shuffle_players(x=x, shuffle_players=config.shuffle_players)
        x = flip_x_axis(x=x)

    y = get_made_shot_label(x=x, label_start_index=x.size(dim=1) - config.y_frames)
    x = x[:, : config.y_frames]

    # ensure x is divisible by patch length
    x = x[:, x.size(dim=1) % config.patch_length :]

    if not config.include_z:
        x = x[:, :, :2]

    x = normalize_coordinates(x=x)

    if x.size(dim=1) < config.num_frames:
        x = torch.cat((x, -10 * torch.ones(x.size(dim=0), config.num_frames - x.size(dim=1), x.size(dim=2))), dim=1)

    return x, y


def get_made_shot_label(x: torch.tensor, label_start_index: int) -> torch.tensor:
    made_x_min = 4.7
    made_x_max = 6
    made_y_min = 24.3
    made_y_max = 25.7
    made_z_min = 9
    made_z_max = 10.1

    label = torch.zeros(2)

    ball = x[0, label_start_index:]

    label[0] = torch.any(
        (ball[:, 0] >= made_x_min)
        * (ball[:, 0] <= made_x_max)
        * (ball[:, 1] >= made_y_min)
        * (ball[:, 1] <= made_y_max)
        * (ball[:, 2] >= made_z_min)
        * (ball[:, 2] <= made_z_max)
    ).long()
    label[1] = torch.any(
        (ball[:, 0] >= X_MAX - made_x_max)
        * (ball[:, 0] <= X_MAX - made_x_min)
        * (ball[:, 1] >= made_y_min)
        * (ball[:, 1] <= made_y_max)
        * (ball[:, 2] >= made_z_min)
        * (ball[:, 2] <= made_z_max)
    ).long()

    return label
