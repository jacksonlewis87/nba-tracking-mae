from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from data.decoding.data_config import DataConfig
from data.decoding.transforms import get_tracking_transforms_function
from data.utils import get_data_split
from utils import list_files_in_directory, load_tensor


class DecodingDataset(Dataset):
    def __init__(self, config: DataConfig, game_ids: list[str], stage: str = "train") -> None:
        self.config = config
        self.game_ids = game_ids
        self.stage = stage
        self.tracking_transforms_function = get_tracking_transforms_function(task_str=self.config.task)

    def __getitem__(self, index: int):
        x = load_tensor(path=self.config.tensor_path, tensor_name=self.game_ids[index])
        x, y = self.tracking_transforms_function(x=x, config=self.config, stage=self.stage)
        return x, y

    def __len__(self) -> int:
        return len(self.game_ids)


class DecodingDataModule(LightningDataModule):
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

        self.train_dataset = DecodingDataset(
            config=self.config,
            game_ids=data_split["train"],
        )
        self.val_dataset = DecodingDataset(
            config=self.config,
            game_ids=data_split["val"],
            stage="eval",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)


def setup_data_module(config: DataConfig, stage: str = None):
    return DecodingDataModule(config=config, stage=stage)
