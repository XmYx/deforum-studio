import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from deforum.utils.device import empty_cache, get_device


class RAFT:
    def __init__(self):
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.device = get_device()
        self.model = raft_large(weights=weights, progress=False).to(self.device).eval()

    def predict(self, image1, image2, num_flow_updates: int = 12):
        img1 = F.to_tensor(image1)
        img2 = F.to_tensor(image2)
        img1_batch, img2_batch = img1.unsqueeze(0), img2.unsqueeze(0)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)

        dev = torch.device(self.device)
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            flow = self.model(image1=img1_batch.to(dev), image2=img2_batch.to(dev),
                              num_flow_updates=num_flow_updates)[-1].float().cpu().numpy()[0]

        # align the flow array to have the shape (w, h, 2) so it's compatible with the rest of CV2's flow methods
        flow = np.transpose(flow, (1, 2, 0))

        return flow

    def delete_model(self):
        del self.model