from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from data.mae.data_config import DataConfig
from data.transforms import random_crop, shuffle_players, normalize_coordinates, flip_x_axis
from data.utils import get_data_split
from utils import list_files_in_directory, load_tensor


class MAEDataset(Dataset):
    def __init__(self, config: DataConfig, game_ids: list[str], stage: str = "train") -> None:
        self.config = config
        self.game_ids = game_ids
        self.stage = stage

    def __getitem__(self, index: int):
        x = load_tensor(path=self.config.tensor_path, tensor_name=self.game_ids[index])
        x = random_crop(x=x, length=self.config.num_frames, dim=1)
        if not self.config.include_z:
            x = x[:, :, :2]

        if self.stage != "eval":
            x = shuffle_players(x=x, shuffle_players=self.config.shuffle_players)
            x = flip_x_axis(x=x)

        x = normalize_coordinates(x=x)

        return x

    def __len__(self) -> int:
        return len(self.game_ids)


class MAEDataModule(LightningDataModule):
    def __init__(self, config: DataConfig, stage: str = None) -> None:
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.setup(stage=stage)

    def setup(self, stage: Optional[str] = None):
        game_ids = list_files_in_directory(path=self.config.tensor_path, suffix=".pt")

        data_split = get_data_split(
            config=self.config,
            game_ids=game_ids,
            stage=stage,
        )

        self.train_dataset = MAEDataset(
            config=self.config,
            game_ids=data_split["train"],
        )
        self.val_dataset = MAEDataset(
            config=self.config,
            game_ids=data_split["val"],
            stage="eval",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)


def setup_data_module(config: DataConfig, stage: str = None):
    return MAEDataModule(config=config, stage=stage)
