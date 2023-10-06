import os
import cv2
import gradio as gr

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)


    def get_block_name(self):
        return "button"


class HotshotXLParams:

    def __init__(
        self,
        model="temporal_layers.safetensors",
        enable=False,
        video_length=8,
        fps=8,
        loop_number=0,
        format="GIF",
        video_source=None
    ):
        self.model = model
        self.enable = enable
        self.video_length = video_length
        self.fps = fps
        self.loop_number = loop_number
        self.format = format
        self.video_source = video_source


    def get_list(self, is_img2img: bool):
        list_var = list(vars(self).values())

        return list_var


    def _check(self):
        assert (
            self.video_length >= 0 and self.fps > 0
        ), "Video length and FPS should be positive."
        assert not set(["GIF", "MP4", "PNG"]).isdisjoint(
            self.format
        ), "At least one saving format should be selected."


    def set_p(self, p):
        self._check()
        p.batch_size = p.batch_size * self.video_length


class HotshotXLUiGroup:
    txt2img_submit_button = None
    img2img_submit_button = None

    def __init__(self):
        self.params = HotshotXLParams()

    def render(self, is_img2img: bool, model_dir: str):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        elemid_prefix = "img2img-hsxl-" if is_img2img else "txt2img-hsxl-"
        model_list = [f for f in os.listdir(model_dir) if f != ".gitkeep"]
        with gr.Accordion("Hotshot-XL", open=False):
            with gr.Row():

                def refresh_models(*inputs):
                    new_model_list = [
                        f for f in os.listdir(model_dir) if f != ".gitkeep"
                    ]
                    dd = inputs[0]
                    if dd in new_model_list:
                        selected = dd
                    elif len(new_model_list) > 0:
                        selected = new_model_list[0]
                    else:
                        selected = None
                    return gr.Dropdown.update(choices=new_model_list, value=selected)

                self.params.model = gr.Dropdown(
                    choices=model_list,
                    value=(self.params.model if self.params.model in model_list else None),
                    label="Temporal Layers",
                    type="value",
                    tooltip="Choose which temporal layers will be injected into the generation process.",
                    elem_id=f"{elemid_prefix}temporal-layers",
                )
                refresh_model = ToolButton(value="\U0001f504")
                refresh_model.click(
                    refresh_models, self.params.model, self.params.model
                )
            with gr.Row():
                self.params.enable = gr.Checkbox(
                    value=self.params.enable, label="Enable Hotshot-XL",
                    elem_id=f"{elemid_prefix}enable"
                )
                self.params.video_length = gr.Number(
                    minimum=0,
                    value=self.params.video_length,
                    label="Number of frames",
                    precision=0,
                    tooltip="Total length of video in frames.",
                    elem_id=f"{elemid_prefix}video-length",
                )
                self.params.fps = gr.Number(
                    value=self.params.fps, label="FPS", precision=0,
                    tooltip="How many frames per second the gif will run.",
                    elem_id=f"{elemid_prefix}fps"
                )
                self.params.loop_number = gr.Number(
                    minimum=0,
                    value=self.params.loop_number,
                    label="Display loop number",
                    precision=0,
                    tooltip="How many times the animation will loop, a value of 0 will loop forever.",
                    elem_id=f"{elemid_prefix}loop-number",
                )

            with gr.Row():
                self.params.format = gr.CheckboxGroup(
                    choices=["GIF", "MP4", "PNG", "TXT"],
                    label="Save",
                    type="value",
                    tooltip="Which formats the animation should be saved in",
                    elem_id=f"{elemid_prefix}save-format",
                    value=self.params.format,
                )

            self.params.video_source = gr.Video(
                value=self.params.video_source,
                label="Video source",
            )

            def update_fps(video_source):
                if video_source is not None and video_source != '':
                    cap = cv2.VideoCapture(video_source)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    cap.release()
                    return fps
                else:
                    return int(self.params.fps.value)

            self.params.video_source.change(update_fps, inputs=self.params.video_source, outputs=self.params.fps)


            with gr.Row():
                unload = gr.Button(
                    value="Move Temporal Layers to CPU (default if lowvram)"
                )
                remove = gr.Button(value="Remove Temporal Layers from any memory")
                #unload.click(fn=motion_module.unload)
                #remove.click(fn=motion_module.remove)
        return self.register_unit(is_img2img)


    def register_unit(self, is_img2img: bool):

        unit = gr.State(value=HotshotXLParams)
        button = HotshotXLUiGroup.img2img_submit_button if is_img2img else HotshotXLUiGroup.txt2img_submit_button
        if button:
            button.click(
            fn=HotshotXLParams,
            inputs=self.params.get_list(is_img2img),
            outputs=unit,
            queue=False,
        )

        return unit


    @staticmethod
    def on_after_component(component, **_kwargs):
        elem_id = getattr(component, "elem_id", None)

        if elem_id == "txt2img_generate":
            HotshotXLUiGroup.txt2img_submit_button = component
            return

        if elem_id == "img2img_generate":
            HotshotXLUiGroup.img2img_submit_button = component
            return