import torch
from hotshot_xl.models.temporal_layers import HotshotXLTemporalLayers


class HotshotXLModelController:

    def hijack_sdxl_model(self, model, temporal_layers: HotshotXLTemporalLayers):
        ...

    def restore(self):
        ...

model_controller = HotshotXLModelController()