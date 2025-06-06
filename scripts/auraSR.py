import torch
from PIL import Image
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

from scripts.aura_sr import AuraSR
from modules import devices
from modules.upscaler import Upscaler, UpscalerData

class UpscalerAuraSR(Upscaler):
    model = None
    scalers = []

    def do_upscale(self, img, selected_model=None):
        if self.model is None:
            self.model = AuraSR.from_pretrained("fal/AuraSR-v2",)
            self.model.upsampler.to(devices.dtype)
        else:
            self.model.upsampler.to(devices.get_optimal_device())

        with torch.no_grad():
            upscaledImage = self.model.upscale_4x_overlapped(img.convert('RGB'))
            self.model.upsampler.to('cpu')

        return upscaledImage.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=LANCZOS)

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "AuraSR_4x"
        self.scalers = [UpscalerData("AuraSR_4x", None, self)]
