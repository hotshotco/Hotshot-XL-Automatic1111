from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from einops import rearrange
from hotshot_xl.models.temporal_layers import HotshotXLTemporalLayers
from safetensors import safe_open
from sgm.modules.attention import SpatialTransformer
from sgm.modules.diffusionmodules.openaimodel import ResBlock, TimestepEmbedSequential
from sgm.modules.diffusionmodules.util import GroupNorm32

from hotshot_xl.utils import hash_str

@dataclass
class TemporalModel:
    model: HotshotXLTemporalLayers
    model_hash: str

class TimeCentricTensorReshaper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.video_length = 8

    def forward(self, hidden_states, encoder_hidden_states=None):

        if self.video_length < 1:
            # skip this module
            return hidden_states

        hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=self.video_length)
        hidden_states = self.module(hidden_states, encoder_hidden_states)
        return rearrange(hidden_states, "b c f h w -> (b f) c h w")


class HotshotXLModelController:

    def __init__(self):
        self.current_loaded_temporal_layers: Optional[TemporalModel] = None
        self.time_centric_tensor_reshaper_cache = []

    def set_video_length(self, video_length: int = 1):
        if video_length < 1:
            print("Warning - video length is < 1 - temporal layers will be deactivated.")

        for module in self.time_centric_tensor_reshaper_cache:
            module.video_length = video_length

    def load_and_inject(self, sd_model, temporal_layers_model_path: str):

        if (not self.current_loaded_temporal_layers or
                self.current_loaded_temporal_layers.model_hash != hash_str(temporal_layers_model_path)):
            temporal_layers = HotshotXLTemporalLayers()
            torch_model = {}
            with safe_open(temporal_layers_model_path, framework="pt", device="cuda") as f:
                for key in f.keys():
                    torch_model[key] = f.get_tensor(key)
            temporal_layers.load_state_dict(torch_model)
            self.current_loaded_temporal_layers = TemporalModel(
                model=temporal_layers.to(device=sd_model.device, dtype=sd_model.dtype),
                model_hash=temporal_layers_model_path
            )

        self._hijack_sdxl_model(sd_model, self.current_loaded_temporal_layers.model)

    def _inject(self, spatial_module, temporal_module, type_to_insert_after: type):
        for i, module in enumerate(spatial_module):
            if type(module) == type_to_insert_after:
                self.time_centric_tensor_reshaper_cache.append(
                    TimeCentricTensorReshaper(temporal_module)
                )
                spatial_module.insert(i + 1, self.time_centric_tensor_reshaper_cache[-1])
                break

    def _hijack_sdxl_model(self, sd_model, temporal_layers: HotshotXLTemporalLayers):
        unet = sd_model.model.diffusion_model

        unet_input_block_index_to_temporal_layer_map = {
            1: {"block_index": 0, "attention_index": 0, "insert_after": ResBlock},
            2: {"block_index": 0, "attention_index": 1, "insert_after": ResBlock},

            4: {"block_index": 1, "attention_index": 0, "insert_after": SpatialTransformer},
            5: {"block_index": 1, "attention_index": 1, "insert_after": SpatialTransformer},

            7: {"block_index": 2, "attention_index": 0, "insert_after": SpatialTransformer},
            8: {"block_index": 2, "attention_index": 1, "insert_after": SpatialTransformer}
        }

        unet_output_block_index_to_temporal_layer_map = {
            0: {"block_index": 0, "attention_index": 0, "insert_after": SpatialTransformer},
            1: {"block_index": 0, "attention_index": 1, "insert_after": SpatialTransformer},
            2: {"block_index": 0, "attention_index": 2, "insert_after": SpatialTransformer},

            3: {"block_index": 1, "attention_index": 0, "insert_after": SpatialTransformer},
            4: {"block_index": 1, "attention_index": 1, "insert_after": SpatialTransformer},
            5: {"block_index": 1, "attention_index": 2, "insert_after": SpatialTransformer},

            6: {"block_index": 2, "attention_index": 0, "insert_after": ResBlock},
            7: {"block_index": 2, "attention_index": 1, "insert_after": ResBlock},
            8: {"block_index": 2, "attention_index": 2, "insert_after": ResBlock},
        }

        for unet_block_index, temporal_info in unet_input_block_index_to_temporal_layer_map.items():
            temporal_module = temporal_layers.get_temporal_layer(
                -1,
                temporal_info['block_index'],
                temporal_info['attention_index']
            )

            self._inject(unet.input_blocks[unet_block_index], temporal_module, temporal_info['insert_after'])

        for unet_block_index, temporal_info in unet_output_block_index_to_temporal_layer_map.items():
            temporal_module = temporal_layers.get_temporal_layer(
                1,
                temporal_info['block_index'],
                temporal_info['attention_index']
            )

            self._inject(unet.output_blocks[unet_block_index], temporal_module, temporal_info['insert_after'])

        # hotshot xl group norms with a different tensor arrangement - same as animate diff

        self.gn32_original_forward = GroupNorm32.forward
        gn32_original_forward = self.gn32_original_forward

        def groupnorm32_mm_forward(self, x):
            x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
            x = gn32_original_forward(self, x)
            x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
            return x

        GroupNorm32.forward = groupnorm32_mm_forward


    def restore(self, sd_model):
        unet = sd_model.model.diffusion_model
        GroupNorm32.forward = self.gn32_original_forward

        self.time_centric_tensor_reshaper_cache = []

        for block in unet.input_blocks:
            if type(block) is TimestepEmbedSequential:
                for i, module in enumerate(block):
                    if type(module) is TimeCentricTensorReshaper:
                        block.pop(i)
                        break

        for block in unet.output_blocks:
            if type(block) is TimestepEmbedSequential:
                for i, module in enumerate(block):
                    if type(module) is TimeCentricTensorReshaper:
                        block.pop(i)
                        break


model_controller = HotshotXLModelController()