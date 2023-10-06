import torch
from hotshot_xl.models.temporal_layers import HotshotXLTemporalLayers
from sgm.modules.attention import SpatialTransformer
from sgm.modules.diffusionmodules.openaimodel import ResBlock, TimestepEmbedSequential
import torch.nn as nn
from einops import rearrange
from sgm.modules.diffusionmodules.util import GroupNorm32

class InflateTime(nn.Module):
    def __init__(self, module, video_length):
        super().__init__()
        self.module = module
        self.video_length = video_length

    def forward(self, hidden_states, encoder_hidden_states=None):
        hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=self.video_length)
        hidden_states = self.module(hidden_states, encoder_hidden_states)
        return rearrange(hidden_states, "b c f h w -> (b f) c h w")

class HotshotXLModelController:

    def inject(self, spatial_module, temporal_module, type_to_insert_after: type, video_length=8):
        for i, module in enumerate(spatial_module):
            if type(module) == type_to_insert_after:
                spatial_module.insert(i + 1, InflateTime(temporal_module, video_length=video_length))
                break

    def hijack_sdxl_model(self, sd_model, temporal_layers: HotshotXLTemporalLayers):
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

            self.inject(unet.input_blocks[unet_block_index], temporal_module, temporal_info['insert_after'])

        for unet_block_index, temporal_info in unet_output_block_index_to_temporal_layer_map.items():
            temporal_module = temporal_layers.get_temporal_layer(
                1,
                temporal_info['block_index'],
                temporal_info['attention_index']
            )

            self.inject(unet.output_blocks[unet_block_index], temporal_module, temporal_info['insert_after'])

        # hotshot xl group norms with a different tensor arrangement - same as animate diff

        self.gn32_original_forward = GroupNorm32.forward
        gn32_original_forward = self.gn32_original_forward

        def groupnorm32_mm_forward(self, x):
            x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
            x = gn32_original_forward(self, x)
            x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
            return x

        GroupNorm32.forward = groupnorm32_mm_forward

        # modules
        # - TimestepEmbedSequential
        # -- Res block
        # -- Spatial Transformer
        # -- (Optional Upscaler)

        # todo - need to insert hooks for each resnet conv2d call

    def restore(self, sd_model):
        unet = sd_model.model.diffusion_model
        GroupNorm32.forward = self.gn32_original_forward

        for block in unet.input_blocks:
            if type(block) is TimestepEmbedSequential:
                for i, module in enumerate(block):
                    if type(module) is InflateTime:
                        block.pop(i)
                        break

        for block in unet.output_blocks:
            if type(block) is TimestepEmbedSequential:
                for i, module in enumerate(block):
                    if type(module) is InflateTime:
                        block.pop(i)
                        break


model_controller = HotshotXLModelController()