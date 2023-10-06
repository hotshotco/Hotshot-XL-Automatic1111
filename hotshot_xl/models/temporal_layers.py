import torch.nn as nn
from .transformer_temporal import TransformerTemporal

class Block(nn.Module):
    def __init__(self, temporal_attentions: list):
        super().__init__()
        self.temporal_attentions = nn.ModuleList(temporal_attentions)

class HotshotXLTemporalLayers(nn.Module):
    def __init__(self):
        super().__init__()

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for channel in [320, 640, 1280]:
            blks = []
            for _ in range(2):
                blks.append(TransformerTemporal(
                num_attention_heads=8,
                attention_head_dim=channel // 8,
                in_channels=channel,
                cross_attention_dim=None,
            ))
            self.down_blocks.append(Block(blks))

        for channel in [1280, 640, 320]:
            blks = []
            for _ in range(3):
                blks.append(TransformerTemporal(
                num_attention_heads=8,
                attention_head_dim=channel // 8,
                in_channels=channel,
                cross_attention_dim=None,
            ))
            self.up_blocks.append(Block(blks))

if __name__ == "__main__":
    layers = HotshotXLTemporalLayers()
    from safetensors import safe_open

    torch_model = {}
    with safe_open("/path/to/hsxl_temporal_layers.safetensors", framework="pt",
                   device="cuda") as f:
        for key in f.keys():
            torch_model[key] = f.get_tensor(key)

    layers.load_state_dict(torch_model)

