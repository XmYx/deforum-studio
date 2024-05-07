import os
import tempfile
import time
from queue import Queue
import cv2
import numpy as np
import requests
import torch
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image

from deforum.utils.constants import config


class RIFEInterpolator:
    def __init__(self, model_path='rife/flownet-v46.pkl', fp16=False):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16

        model_path = os.path.join(config.root_path, model_path)
        self.ensure_model(model_path)
        self.load_model(model_path, fp16)
    def ensure_model(self, model_path):
        model_url = 'https://github.com/vladmandic/rife/raw/main/model/flownet-v46.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, downloading from {model_url}")
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print("Model downloaded successfully.")
    def load_model(self, model_path, fp16):
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if fp16:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        from deforum.models.RIFE.RIFE_HDv3 import Model
        self.model = Model()
        self.model.load_model(model_path, -1)
        self.model.eval()
        self.model.device()

    def pad(self, img, scale=1.0):
        _, h, w = img.shape[1:]
        tmp = max(128, int(128 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        return F.pad(img, padding).half() if self.fp16 else F.pad(img, padding)

    def interpolate(self, img1, img2, interp_amount=4):
        I0 = self.prep_image(img1)
        I1 = self.prep_image(img2)
        output_images = []

        if interp_amount > 1:
            results = self.execute(I0, I1, interp_amount - 1)
            for mid in results:
                mid_img = (mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
                output_images.append(Image.fromarray(mid_img))
        output_images.append(Image.fromarray(img2))
        return output_images

    def prep_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if isinstance(img, np.ndarray) else np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
        return self.pad(tensor)

    def execute(self, I0, I1, n):
        if self.model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(self.model.inference(I0, I1, (i + 1) * 1. / (n + 1), 1.0))
            return res
        else:
            middle = self.model.inference(I0, I1, 1.0)
            if n == 1:
                return [middle]
            first_half = self.execute(I0, middle, n=n//2)
            second_half = self.execute(middle, I1, n=n//2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]
