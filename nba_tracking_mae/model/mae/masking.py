import torch
from dataclasses import dataclass
from enum import Enum


@dataclass
class MaskingConfig:
    mask_ratio: float = None
    indexes: list[int] = None
    patches_per_index: int = None
    random_indexes: int = None


class MaskingStrategy(Enum):
    INDEX = "index"
    RANDOM = "random"


def get_masking_strategy(masking_strategy: str, config: MaskingConfig):
    if masking_strategy == MaskingStrategy.INDEX.value:
        return IndexMasking(config=config)
    elif masking_strategy == MaskingStrategy.RANDOM.value:
        return RandomMasking(config=config)
    else:
        return None


class IndexMasking(torch.nn.Module):
    def __init__(self, config: MaskingConfig):
        super(IndexMasking, self).__init__()
        self.config = config

        if self.config.random_indexes is not None and self.config.indexes is not None:
            print("Error: random indexes and preset indexes not allowed")
            raise Exception

        self.num_random_indexes = len(self.config.indexes) if self.config.indexes else self.config.random_indexes

    def forward(self, x):
        B, L, D = x.shape  # batch, length, dim
        len_keep = L - (self.num_random_indexes * self.config.patches_per_index)

        noise = torch.rand(B, L)  # noise in [0, 1]

        if self.config.random_indexes is not None:
            indexes = torch.stack([torch.randperm(11)[: self.num_random_indexes] for _ in range(B)], dim=0)
        else:
            indexes = self.config.indexes

        # cherry-pick certain indexes
        if self.config.random_indexes > 0:
            for bi in range(B):
                for index in indexes[bi]:
                    for i in range(self.config.patches_per_index):
                        noise[bi, (index * self.config.patches_per_index) + i] = 2

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small=keep, large=remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0=keep, 1=remove
        mask = torch.ones([B, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


class RandomMasking(torch.nn.Module):
    def __init__(self, config: MaskingConfig):
        super(RandomMasking, self).__init__()
        self.config = config

    def forward(self, x):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.config.mask_ratio))

        noise = torch.rand(N, L)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
