from dataclasses import dataclass


@dataclass
class DataConfig:
    tensor_path: str
    data_split_path: str
    batch_size: int
    train_size: float
    shuffle_players: bool
    num_frames: int
    include_z: bool
    task: str
    min_z: float = None
    num_event_classification_tasks: int = None
    y_frames: int = None
    min_frames: int = None
    patch_length: int = None
