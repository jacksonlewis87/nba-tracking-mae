import torch
from random import randint

from constants import X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


def random_crop(x: torch.tensor, length: int, dim: int) -> torch.tensor:
    current_length = x.size(dim=dim)

    if current_length < dim:
        print("Error: current length smaller than crop length")
        raise Exception

    if current_length == dim:
        return x

    rand = randint(0, current_length - length)
    indexes = [i + rand for i in range(length)]

    return torch.index_select(x, dim=dim, index=torch.tensor(indexes, dtype=torch.long))


def shuffle_players(x: torch.tensor, shuffle_players: bool) -> torch.tensor:
    if not shuffle_players:
        return x

    # dont shuffle ball (0), and keep teams together (1-5, 6-10)
    x[1:6] = shuffle_tensor_dim(x=x[1:6], dim=0)
    x[6:] = shuffle_tensor_dim(x=x[6:], dim=0)

    return x


def shuffle_tensor_dim(x: torch.tensor, dim: int) -> torch.tensor:
    idx = torch.randperm(x.shape[dim])
    return torch.index_select(x, dim=dim, index=torch.tensor(idx, dtype=torch.long))


def normalize_coordinates(x: torch.tensor) -> torch.tensor:
    # (P, T, C)
    x[:, :, 0] = normalize_tensor(x=x[:, :, 0], _min=X_MIN, _max=X_MAX)
    x[:, :, 1] = normalize_tensor(x=x[:, :, 1], _min=Y_MIN, _max=Y_MAX)
    if x.size(dim=2) > 2:
        x[:, :, 2] = normalize_tensor(x=x[:, :, 2], _min=Z_MIN, _max=Z_MAX)

    return x


def normalize_tensor(x: torch.tensor, _min, _max) -> torch.tensor:
    return (2 * (x - _min) / (_max - _min)) - 1


def flip_x_axis(x: torch.tensor) -> torch.tensor:
    if torch.rand(1) >= 0.5:
        x[:, 0] = X_MAX - x[:, 0]

    return x
