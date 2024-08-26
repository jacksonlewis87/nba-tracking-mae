import torch


def patchify(
    x: torch.tensor, channels: int, num_players: int, num_sequence_patches: int, patch_length: int, total_patches: int
):
    # (B, C, P, F) -> (B, T, S)
    x = x.reshape(
        shape=(
            x.shape[0],
            channels,
            num_players,
            1,
            num_sequence_patches,
            patch_length,
        )
    )
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape((x.shape[0], total_patches, patch_length * channels))
    return x


def unpatchify(
    x: torch.tensor, channels: int, num_players: int, num_sequence_patches: int, patch_length: int, num_frames: int
):
    # (B, T, S) -> (B, C, P, F)
    x = x.reshape(
        shape=(
            x.shape[0],
            num_players,
            num_sequence_patches,
            1,
            patch_length,
            channels,
        )
    )
    x = torch.einsum("nhwpqc->nchpwq", x)
    x = x.reshape((x.shape[0], channels, num_players, num_frames))
    return x
