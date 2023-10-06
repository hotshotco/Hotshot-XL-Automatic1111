import install

import os
import gradio as gr
from modules import script_callbacks, scripts, shared
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img)
from typing import Any, Union, Dict
from scripts.hotshot_xl_ui import HotshotXLUiGroup, HotshotXLParams
script_ref = None

script_dir = scripts.basedir()

class HotshotXLScript(scripts.Script):

    def __init__(self):
        print("HotshotXLScript init")
        self.lora_hacker = None
        self.cfg_hacker = None
        self.cn_hacker = None
        global script_ref
        script_ref = self

    def title(self):
        return "Hotshot-XL"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        model_dir = shared.opts.data.get("hotshot_xl_model_path", os.path.join(script_dir, "model"))
        return (HotshotXLUiGroup().render(is_img2img, model_dir),)

    def before_process(
            self, p: StableDiffusionProcessing, params: Union[Dict, HotshotXLParams]
    ):
        import modules.shared as shared
        if shared.sd_model and not shared.sd_model.is_sdxl:
            print("disabling because sdxl is not loaded...")
            return

        if isinstance(params, dict): params = HotshotXLParams(**params)
        if params.enable:
            # todo - setup the temporal layers IF ENABLED
            ...

    def before_process_batch(
            self, p: StableDiffusionProcessing, params: Union[Dict, HotshotXLParams], **kwargs
    ):
        if isinstance(params, dict): params = HotshotXLParams(**params)
        ...
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img):
            ...
            # todo - randomize latents?

    def postprocess(
            self, p: StableDiffusionProcessing, res: Processed, params: Union[Dict, HotshotXLParams]
    ):
        if isinstance(params, dict): params = HotshotXLParams(**params)
        ...

        if params.enable:
            ...
            # todo - restore the pipeline


def on_ui_settings():
    section = ("hotshotxl", "Hotshot-XL")
    pass

def on_model_load(model):
    print("model loaded. ")

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(HotshotXLUiGroup.on_after_component)
script_callbacks.on_model_loaded(on_model_load)

