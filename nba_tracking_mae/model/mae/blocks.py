import torch
import torch.nn as nn
from typing import Optional

from model.blocks import TransformerBlock, get_2d_pos_encoding


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_length: int,
        channels: int,
        embedding_dimension: int,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            channels, embedding_dimension, kernel_size=(1, patch_length), stride=(1, patch_length), bias=True
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MaskedEncoder2D(nn.Module):
    def __init__(
        self,
        patch_length: int,
        channels: int,
        num_patches_x: int,
        num_patches_y: int,
        embedding_dimension: int,
        depth: int,
        num_heads: int,
        masking_strategy: nn.Module,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.channels = channels
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.embedding_dimension = embedding_dimension
        self.depth = depth
        self.num_heads = num_heads
        self.masking_strategy = masking_strategy
        self.patch_embed = self.get_patch_embed()
        self.pos_encoding = self.get_pos_encoding()
        self.class_token = self.get_class_token()
        self.transformer_blocks = self.get_transformer_blocks()
        self.norm = self.get_norm()

    def get_patch_embed(self):
        return PatchEmbed(
            patch_length=self.patch_length,
            channels=self.channels,
            embedding_dimension=self.embedding_dimension,
        )

    def get_pos_encoding(self):
        parameter = nn.Parameter(
            torch.zeros(
                1, (self.num_patches_x * self.num_patches_y) + 1, self.embedding_dimension
            ),  # add 1 for class token
            requires_grad=False,
        )
        encoding = get_2d_pos_encoding(
            embedding_dimension=self.embedding_dimension,
            num_patches_x=self.num_patches_x,
            num_patches_y=self.num_patches_y,
            cls_token=True,
        )
        return parameter.data.copy_(torch.from_numpy(encoding).float().unsqueeze(0))

    def get_class_token(self):
        return nn.Parameter(torch.zeros(1, 1, self.embedding_dimension))

    def get_transformer_blocks(self):
        transformer_blocks = nn.ModuleList()
        for i in range(self.depth):
            transformer_blocks.add_module(
                name=f"transformer_block_{i}",
                module=TransformerBlock(
                    dim=self.embedding_dimension,
                    num_heads=self.num_heads,
                    qkv_bias=True,
                ),
            )
        return transformer_blocks

    def get_norm(self):
        return nn.LayerNorm(self.embedding_dimension)

    def masked_forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_encoding[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.masking_strategy(x)

        # append cls token
        class_token = self.class_token + self.pos_encoding[:, :1, :]
        class_tokens = class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        # apply Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> (torch.Tensor, Optional[list]):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_encoding[:, 1:, :]

        # append cls token
        class_token = self.class_token + self.pos_encoding[:, :1, :]
        class_tokens = class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        attention_outputs = []

        # apply Transformer blocks
        for transformer_block in self.transformer_blocks:
            if return_attention:
                x, attention_output = transformer_block.forward(x=x, return_attention=return_attention)
                attention_outputs += [attention_output]
            else:
                x = transformer_block(x)
        x = self.norm(x)

        if return_attention:
            return x, attention_outputs
        else:
            return x


class MaskedDecoder2D(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_patches_x: int,
        num_patches_y: int,
        embedding_dimension: int,
        decoder_embedding_dimension: int,
        depth: int,
        num_heads: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.embedding_dimension = embedding_dimension
        self.decoder_embedding_dimension = decoder_embedding_dimension
        self.depth = depth
        self.num_heads = num_heads

        self.decoder_embed = self.get_decoder_embed()
        self.mask_token = self.get_mask_token()
        self.pos_encoding = self.get_pos_encoding()
        self.transformer_blocks = self.get_transformer_blocks()
        self.norm = self.get_norm()
        self.decoder_pred = self.get_decoder_pred()

    def get_decoder_embed(self):
        return nn.Linear(self.embedding_dimension, self.decoder_embedding_dimension, bias=True)

    def get_mask_token(self):
        return nn.Parameter(torch.zeros(1, 1, self.decoder_embedding_dimension))

    def get_pos_encoding(self):
        parameter = nn.Parameter(
            torch.zeros(
                1, (self.num_patches_x * self.num_patches_y) + 1, self.decoder_embedding_dimension
            ),  # add 1 for class token
            requires_grad=False,
        )
        encoding = get_2d_pos_encoding(
            embedding_dimension=self.decoder_embedding_dimension,
            num_patches_x=self.num_patches_x,
            num_patches_y=self.num_patches_y,
            cls_token=True,
        )
        return parameter.data.copy_(torch.from_numpy(encoding).float().unsqueeze(0))

    def get_transformer_blocks(self):
        transformer_blocks = nn.ModuleList()
        for i in range(self.depth):
            transformer_blocks.add_module(
                name=f"transformer_block_{i}",
                module=TransformerBlock(
                    dim=self.decoder_embedding_dimension,
                    num_heads=self.num_heads,
                    qkv_bias=True,
                ),
            )
        return transformer_blocks

    def get_norm(self):
        return nn.LayerNorm(self.decoder_embedding_dimension)

    def get_decoder_pred(self):
        return nn.Linear(self.decoder_embedding_dimension, self.patch_size, bias=True)

    def masked_forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.pos_encoding

        # apply Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> (torch.Tensor, Optional[list]):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_encoding

        attention_outputs = []

        # apply Transformer blocks
        for transformer_block in self.transformer_blocks:
            if return_attention:
                x, attention_output = transformer_block.forward(x=x, return_attention=return_attention)
                attention_outputs += [attention_output]
            else:
                x = transformer_block(x)
        x = self.norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        if return_attention:
            return x, attention_outputs
        else:
            return x
