import torch
from hotshot_xl.models.temporal_layers import HotshotXLTemporalLayers


class HotshotXLModelController:

    def hijack_sdxl_model(self, sd_model, temporal_layers: HotshotXLTemporalLayers):
        unet = sd_model.model.diffusion_model

        # todo - need to do something to input and output conv2d layers... ? ??
        # todo - need to insert hooks for the down and up blocks.
        # todo - need to insert hooks for each resnet conv2d call


    def restore(self):
        ...

model_controller = HotshotXLModelController()