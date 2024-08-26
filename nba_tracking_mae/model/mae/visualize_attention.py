import matplotlib.pyplot as plt
import torch

from model.mae.model import TrackingMaskedAutoEncoder
from model.mae.model_config import FullConfig


class AttentionVisualizer:
    def __init__(self, config: FullConfig):
        self.config = config
        self.model = self.load_model()

    def load_model(self):
        model = TrackingMaskedAutoEncoder.load_from_checkpoint(
            checkpoint_path=self.config.model_config.checkpoint_path, config=self.config
        )
        model.eval()
        return model

    def model_forward(self, x: torch.tensor):
        latent, pred, encoder_attention, decoder_attention = self.model.forward_return_attention(
            torch.stack([x], dim=0)
        )
        return latent, pred, torch.stack(encoder_attention, dim=0), decoder_attention

    def visualize_attention(self, x: torch.tensor):
        latent, pred, encoder_attention, decoder_attention = self.model_forward(x=x)
        print(encoder_attention.size())

        # Visualize attention weights
        attention_weights = encoder_attention.squeeze(1).detach().cpu().numpy()
        # attention_weights = encoder_attention.detach().cpu().numpy()
        print(attention_weights.shape)

        fig, axs = plt.subplots(
            nrows=self.config.model_config.encoder_depth,
            ncols=self.config.model_config.num_encoder_heads,
            figsize=(self.config.model_config.encoder_depth * 3, self.config.model_config.num_encoder_heads * 3),
        )
        for layer in range(attention_weights.shape[0]):
            for head in range(attention_weights.shape[1]):
                ax = axs[layer, head]
                ax.imshow(attention_weights[layer, head], cmap="hot", interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Layer {layer + 1}, Head {head + 1}")

        plt.tight_layout()
        # plt.show()
        plt.savefig("C://Users/Jackson/Downloads/attention.png")
        print(asfasd)
