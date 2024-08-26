import torch
import yaml
from pytorch_lightning import LightningModule
from torch import nn
from typing import Tuple

from data.decoding.transforms import Task
from loss.losses import get_loss
from model.decoding.blocks import Decoder
from model.decoding.model_config import FullConfig
from model.mae.model import TrackingMaskedAutoEncoder


class TrackingDecoder(LightningModule):
    def __init__(
        self,
        config: FullConfig,
    ):
        super().__init__()
        self.config = config

        self.save_hyperparameters("config")

        self.loss = get_loss(loss=self.config.model_config.loss)

        self.encoder, self.encoder_config = self.load_encoder()
        self.decoder = Decoder(
            num_patches_x=self.encoder_config.model_config.num_players,
            num_patches_y=self.encoder_config.model_config.num_sequence_patches,
            embedding_dimension=self.encoder_config.model_config.embedding_dimension,
            decoder_embedding_dimension=config.model_config.decoder_embedding_dimension,
            depth=self.config.model_config.decoder_depth,
            num_heads=self.config.model_config.num_decoder_heads,
        )
        self.proj_head = self.get_proj_head()

    def load_encoder(self):
        # load encoder architecture
        with open(self.config.model_config.encoder_checkpoint_config_path) as stream:
            mae_config = yaml.load(stream=stream, Loader=yaml.Loader)["config"]
        encoder = TrackingMaskedAutoEncoder(config=mae_config).encoder

        # load encoder weights
        encoder_prefix = "encoder."
        state_dict = torch.load(self.config.model_config.encoder_checkpoint_path)["state_dict"]
        encoder_state_dict = {k.replace(encoder_prefix, ""): v for k, v in state_dict.items() if encoder_prefix in k}
        encoder.load_state_dict(state_dict=encoder_state_dict)

        if self.config.model_config.freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder, mae_config

    def get_proj_head(self):
        if self.config.data_config.task == Task.BALL_HEIGHT_CLASSIFICATION.value:
            return self.get_temporal_classification_proj_head()
        elif self.config.data_config.task == Task.MADE_BASKET_CLASSIFICATION.value:
            return self.get_event_classification_proj_head()
        else:
            return None

    def get_temporal_classification_proj_head(self):
        head = nn.ModuleList()
        head.add_module("flatten", nn.Flatten())
        head.add_module(
            "linear_0",
            nn.Linear(
                self.config.model_config.decoder_embedding_dimension
                * (
                    (
                        self.encoder_config.model_config.num_sequence_patches
                        * self.encoder_config.model_config.num_players
                    )
                    + 1
                ),
                self.config.data_config.num_frames,
                bias=True,
            ),
        )
        head.add_module("sigmoid", nn.Sigmoid())
        return head

    def get_event_classification_proj_head(self):
        head = nn.ModuleList()
        head.add_module(
            "linear_0",
            nn.Linear(
                self.config.model_config.decoder_embedding_dimension,
                self.config.data_config.num_event_classification_tasks,
                bias=True,
            ),
        )
        head.add_module("sigmoid", nn.Sigmoid())
        return head

    def post_process_decoder(self, x: torch.tensor):
        if self.config.data_config.task == Task.MADE_BASKET_CLASSIFICATION.value:
            x = x[:, 0]  # only use class token
        return x

    def forward_loss(self, output, labels):
        return self.loss(output, labels)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        pred = self.encoder(x)
        pred = self.decoder(pred)
        pred = self.post_process_decoder(pred)
        for module in self.proj_head:
            pred = module(pred)
        return pred

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> float:
        x, y = batch
        pred = self.forward(x)
        loss = self.forward_loss(output=pred, labels=y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> None:
        x, y = batch
        pred = self.forward(x)
        loss = self.forward_loss(output=pred, labels=y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.model_config.learning_rate)
        # scheduler = {
        #     "scheduler": optim.lr_scheduler.OneCycleLR(
        #         optimizer,
        #         max_lr=self.config.learning_rate,
        #         total_steps=self.config.epochs,
        #     ),
        # }
        return optimizer  # , [scheduler]
