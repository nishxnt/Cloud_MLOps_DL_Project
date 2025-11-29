import torch
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io

class SqueezeNetHandler(BaseHandler):
    """
    Custom TorchServe handler for SqueezeNet Intel dataset.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        model_pt_path = f"{model_dir}/model_scripted.pt"

        self.model = torch.jit.load(model_pt_path)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor()
        ])

        self.initialized = True

    def preprocess(self, data):
        img_bytes = data[0].get("data") or data[0].get("body")

        if isinstance(img_bytes, bytes):
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            raise ValueError("Unsupported input format")

        tensor = self.transform(image).unsqueeze(0)
        return tensor

    def inference(self, input_batch):
        with torch.no_grad():
            outputs = self.model(input_batch)
        return outputs

    def postprocess(self, inference_output):
        _, predicted = torch.max(inference_output, 1)
        return [int(predicted.item())]