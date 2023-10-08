import os
import cv2
import gradio as gr
try:
    from scripts.hotshot_xl_model_controller import model_controller
except:
    ...
import packages

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)


    def get_block_name(self):
        return "button"

class HotshotXLParams:

    def __init__(
            self,
            model="hsxl_temporal_layers.f16.safetensors",
            enable=False,
            video_length=8,
            fps=8,
            loop_number=0,
            batch_size=8,
            stride=1,
            overlap=-1,
            format=["GIF"],
            #interp='Off',
            #interp_x=10,
            reverse=[],
            original_size_width=1920,
            original_size_height=1080,
            target_size_width=512,
            target_size_height=512,

            negative_original_size_width=512,
            negative_original_size_height=512,
            negative_target_size_width=64,
            negative_target_size_height=64,
    ):
        self.model = model
        self.enable = enable
        self.video_length = video_length
        self.fps = fps
        self.loop_number = loop_number
        self.batch_size = batch_size
        self.stride = stride
        self.overlap = overlap
        self.format = format
        #self.interp = interp
        #self.interp_x = interp_x
        self.reverse = reverse

        self.original_size_width = original_size_width
        self.original_size_height = original_size_height
        self.target_size_width = target_size_width
        self.target_size_height = target_size_height

        self.negative_original_size_width = negative_original_size_width
        self.negative_original_size_height = negative_original_size_height
        self.negative_target_size_width = negative_target_size_width
        self.negative_target_size_height = negative_target_size_height

    def get_list(self, is_img2img: bool):
        list_var = list(vars(self).values())

        return list_var


    def _check(self):
        assert (
            self.video_length >= 0 and self.fps > 0
        ), "Video length and FPS should be positive."
        assert not set(["GIF", "PNG"]).isdisjoint(
            self.format
        ), "At least one saving format should be selected."

    def set_p(self, p):
        self._check()
        if self.video_length < self.batch_size:
            p.batch_size = self.batch_size
        else:
            p.batch_size = self.video_length
        if self.video_length == 0:
            self.video_length = p.batch_size
            self.video_default = True
        else:
            self.video_default = False
        if self.overlap == -1:
            self.overlap = self.batch_size // 4
        if "PNG" not in self.format:
            p.do_not_save_samples = True


class HotshotXLUiGroup:
    txt2img_submit_button = None
    img2img_submit_button = None

    def __init__(self):
        self.params = HotshotXLParams()

    def render(self, is_img2img: bool, model_dir: str):

        if is_img2img:
            return gr.State()

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        elemid_prefix = "img2img-hsxl-" if is_img2img else "txt2img-hsxl-"
        model_list = [f for f in os.listdir(model_dir) if f != ".gitkeep"]

        def padding():
            with gr.Row():
                gr.Markdown('<div style="width:100%; margin-bottom:1em;"></div>')

        with gr.Accordion("Hotshot-XL", open=False):

            further_setup_needed = False

            if not packages.is_diffusers_installed():
                further_setup_needed = True
                with gr.Row():
                    gr.Markdown(f"""This package relies on diffusers=={packages.min_diffusers_version}. Please make sure it is installed!
                    pip install diffusers=={packages.min_diffusers_version}""")

            if len(model_list) == 0:
                further_setup_needed = True
                with gr.Row():
                    gr.Markdown("No models found!")
                with gr.Row():
                    gr.Markdown(f"Please Install models to '{model_dir}'")
                with gr.Row():
                    gr.Markdown(f"""Download the weights from <a href='https://huggingface.co/hotshotco/Hotshot-XL/blob/main/hsxl_temporal_layers.f16.safetensors'>here</a>""")

            if further_setup_needed:
                with gr.Row():
                    gr.Markdown(f"After completing the above, please reload your UI")
                return gr.State()

            with gr.Row():
                gr.Markdown("""This model was trained at aspects close to 512x512 so we recommend using the <a href='https://huggingface.co/hotshotco/SDXL-512/blob/main/hsxl_base_1.0.f16.safetensors'>hotshotco/SDXL-512</a> model which was fine tuned off SDXL.
                For more info about this project you can read more at the <a href='https://github.com/hotshotco/Hotshot-XL'>Github repository.</a>""")

            padding()

            with gr.Row():

                self.params.enable = gr.Checkbox(
                    value=self.params.enable, label="Enabled",
                    elem_id=f"{elemid_prefix}enable"
                )

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
                    label="Model",
                    type="value",
                    tooltip="Choose which temporal layers will be injected into the generation process.",
                    elem_id=f"{elemid_prefix}temporal-layers",
                )

                refresh_model = ToolButton(value="\U0001f504")
                refresh_model.click(
                    refresh_models, self.params.model, self.params.model
                )

            padding()

            with gr.Row():

                self.params.video_length = gr.Number(
                    minimum=0,
                    value=self.params.video_length,
                    label="Total Frames",
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
                    label="Loop Count",
                    precision=0,
                    tooltip="How many times the animation will loop, a value of 0 will loop forever.",
                    elem_id=f"{elemid_prefix}loop-number",
                )

            padding()

            with gr.Row():
                # self.params.closed_loop = gr.Checkbox(
                #     value=self.params.closed_loop,
                #     label="Closed loop",
                #     tooltip="If enabled, will try to make the last frame the same as the first frame.",
                #     elem_id=f"{elemid_prefix}closed-loop",
                # )

                self.params.batch_size = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=self.params.batch_size,
                    label="Context batch size",
                    step=1,
                    precision=0,
                    elem_id=f"{elemid_prefix}batch-size",
                )
                self.params.stride = gr.Number(
                    minimum=1,
                    value=self.params.stride,
                    label="Stride",
                    precision=0,
                    tooltip="",
                    elem_id=f"{elemid_prefix}stride",
                )
                self.params.overlap = gr.Number(
                    minimum=-1,
                    value=self.params.overlap,
                    label="Overlap",
                    precision=0,
                    tooltip="Number of frames to overlap in context.",
                    elem_id=f"{elemid_prefix}overlap",
                )

            padding()

            with gr.Row():
                self.params.format = gr.CheckboxGroup(
                    choices=["GIF", "PNG"],
                    label="Output Format",
                    type="value",
                    tooltip="Which formats the animation should be saved in",
                    elem_id=f"{elemid_prefix}save-format",
                    value=self.params.format,
                )
                self.params.reverse = gr.CheckboxGroup(
                    choices=["Enabled", "Remove head", "Remove tail"],
                    label="Ping Pong Effect",
                    type="index",
                    tooltip="Reverse the resulting animation, remove the first and/or last frame from duplication.",
                    elem_id=f"{elemid_prefix}reverse",
                    value=self.params.reverse
                )

            padding()

            with gr.Row():
                gr.Markdown("SDXL Conditioning")

            with gr.Row():
                with gr.Accordion("Positive"):
                    with gr.Row():
                        self.params.original_size_width = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=self.params.original_size_width,
                            label="Original Width",
                            step=8,
                            precision=0,
                            elem_id=f"{elemid_prefix}og-width",
                        )
                        self.params.target_size_width = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=self.params.target_size_width,
                            label="Target Width",
                            step=8,
                            precision=0,
                            elem_id=f"{elemid_prefix}tgt-width",
                        )

                    padding()

                    with gr.Row():
                        self.params.original_size_height = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=self.params.original_size_height,
                            label="Original Height",
                            step=8,
                            precision=0,
                            elem_id=f"{elemid_prefix}og-height",
                        )
                        self.params.target_size_height = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=self.params.target_size_height,
                            label="Target Height",
                            step=8,
                            precision=0,
                            elem_id=f"{elemid_prefix}tgt-height",
                        )

            with gr.Row():
                with gr.Accordion("Negative"):
                    with gr.Row():
                        self.params.negative_original_size_width = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=self.params.negative_original_size_width,
                            label="Original Width",
                            step=8,
                            precision=0,
                            elem_id=f"{elemid_prefix}ng-og-width",
                        )
                        self.params.negative_target_size_width = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=self.params.negative_target_size_width,
                            label="Target Width",
                            step=8,
                            precision=0,
                            elem_id=f"{elemid_prefix}ng-tgt-width",
                        )

                    padding()

                    with gr.Row():
                        self.params.negative_original_size_height = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=self.params.negative_original_size_height,
                            label="Original Height",
                            step=8,
                            precision=0,
                            elem_id=f"{elemid_prefix}ng-og-height",
                        )
                        self.params.negative_target_size_height = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=self.params.negative_target_size_height,
                            label="Target Height",
                            step=8,
                            precision=0,
                            elem_id=f"{elemid_prefix}ng-tgt-height",
                        )

            # with gr.Row():
            #     self.params.interp = gr.Radio(
            #         choices=["Off", "FILM"],
            #         label="Frame Interpolation Mode",
            #         tooltip="Interpolate between frames with Deforum's FILM implementation. Requires Deforum extension.",
            #         elem_id=f"{elemid_prefix}interp-choice",
            #         value=self.params.interp
            #     )
            #     self.params.interp_x = gr.Number(
            #         value=self.params.interp_x, label="Interp X", precision=0,
            #         tooltip="Replace each input frame with X interpolated output frames.",
            #         elem_id=f"{elemid_prefix}interp-x"
            #     )
            # self.params.video_source = gr.Video(
            #     value=self.params.video_source,
            #     label="Video source",
            # )

            def update_fps(video_source):
                if video_source is not None and video_source != '':
                    cap = cv2.VideoCapture(video_source)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    cap.release()
                    return fps
                else:
                    return int(self.params.fps.value)

            #self.params.video_source.change(update_fps, inputs=self.params.video_source, outputs=self.params.fps)
            padding()
            with gr.Row():
                unload = gr.Button(
                    value="Unload Model"
                )
                #remove = gr.Button(value="Clean Memory")
                unload.click(fn=model_controller.unload)
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