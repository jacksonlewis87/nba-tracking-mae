import os

from data.mae.data_module import setup_data_module
from model.mae.model import TrackingMaskedAutoEncoder
from model.mae.model_config import FullConfig
from pytorch_model_commons.model.model_driver import ModelDriver


def run_training(config: FullConfig):
    os.makedirs(config.model_config.experiment_path, exist_ok=True)

    data_module = setup_data_module(config=config.data_config)
    model = TrackingMaskedAutoEncoder(config=config)

    ModelDriver(
        full_config=config,
        model_config=config.model_config,
        model=model,
        data_module=data_module,
    ).run_training()
