import math
import torch
from modules import script_callbacks, images
from modules.shared import opts
import gradio as gr

from PIL import Image
from scripts.aura_sr import AuraSR

def upscale(imageSource, *args):
    torch.set_grad_enabled(False)

    if imageSource != None:
        aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")
        upscaledImage = aura_sr.upscale_4x_overlapped(imageSource)
        
        del aura_sr
    else:
        upscaledImage = None

    #   re-enable the go button, return result
    return gr.Button.update(value='Upscale', variant='primary', interactive=True), upscaledImage

def on_ui_tabs():
    def toggleGo ():
        #   disable the go button while processing
        return gr.Button.update(value='...', variant='secondary', interactive=False)

    def saveImage (image, suffix):
        #   use the built-in webui save function
        if image is not None:
            images.save_image(
                image,
                opts.outdir_samples or opts.outdir_extras_samples,
                '',
                extension='png',
                suffix=suffix,
            )
        return

    with gr.Blocks() as auraSR_block:
        with gr.Row():
            with gr.Column():
                #show image dimensions?
                imageSource = gr.Image(label='image source', sources=['upload'], height=640, type='pil', interactive=True, show_download_button=False, )
                go_button = gr.Button(value="Upscale", variant='primary', visible=True)

            with gr.Column():
                outputImage = gr.Image(label='Output', height=640, type='pil', interactive=False, show_label=False,)
                
                with gr.Row():
                    filename = gr.Textbox(value='', placeholder='filename suffix for saving ... (regular pattern first)', lines=1, max_lines=1, scale=3, show_label=False)
                    save_button = gr.Button(value='Save', variant='secondary')

        go_button.click(toggleGo, inputs=[], outputs=[go_button])
        go_button.click(upscale, inputs=imageSource, outputs=[go_button, outputImage])

        save_button.click(fn=saveImage, inputs=[outputImage, filename], outputs=[])

    ####    UI block name, tab display name, internal name
    return [(auraSR_block, "auraSR", "aura_sr")]

script_callbacks.on_ui_tabs(on_ui_tabs)

