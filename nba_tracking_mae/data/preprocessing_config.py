from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    raw_tracking_path: str
    tensor_path: str
    event_duration: int
    target_frame_rate: int
    training_frame_rate: int
    split_long_events: bool = True
