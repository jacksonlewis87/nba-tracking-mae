from dataclasses import dataclass

from data.decoding.data_config import DataConfig


@dataclass
class ModelConfig:
    experiment_path: str
    learning_rate: float
    epochs: int
    checkpoint_path: str
    encoder_checkpoint_path: str
    encoder_checkpoint_config_path: str
    freeze_encoder: bool
    loss: str
    decoder_embedding_dimension: int
    decoder_depth: int
    num_decoder_heads: int


@dataclass
class FullConfig:
    data_config: DataConfig
    model_config: ModelConfig
