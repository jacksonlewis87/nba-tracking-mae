from dataclasses import dataclass

from data.mae.data_config import DataConfig


@dataclass
class ModelConfig:
    experiment_path: str
    learning_rate: float
    epochs: int
    checkpoint_path: str
    num_players: int
    num_sequence_patches: int
    embedding_dimension: int
    encoder_depth: int
    num_encoder_heads: int
    decoder_embedding_dimension: int
    decoder_depth: int
    num_decoder_heads: int
    masking_strategy: str
    masking_ratio: float = None
    masking_indexes: list[int] = None
    random_indexes: int = None


@dataclass
class FullConfig:
    data_config: DataConfig
    model_config: ModelConfig
