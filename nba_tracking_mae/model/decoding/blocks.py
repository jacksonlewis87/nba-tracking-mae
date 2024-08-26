import torch
import torch.nn as nn

from model.blocks import TransformerBlock, get_2d_pos_encoding


class Decoder(nn.Module):
    def __init__(
        self,
        num_patches_x: int,
        num_patches_y: int,
        embedding_dimension: int,
        decoder_embedding_dimension: int,
        depth: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        self.embedding_dimension = embedding_dimension
        self.decoder_embedding_dimension = decoder_embedding_dimension
        self.depth = depth
        self.num_heads = num_heads

        self.decoder_embed = self.get_decoder_embed()
        self.pos_encoding = self.get_pos_encoding()
        self.transformer_blocks = self.get_transformer_blocks()
        self.norm = self.get_norm()

    def get_decoder_embed(self):
        return nn.Linear(self.embedding_dimension, self.decoder_embedding_dimension, bias=True)

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

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_encoding

        # apply Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.norm(x)

        return x
