import torch
from PIL import Image

from . import ZoeDepthNK
from .zoedepth.models.builder import build_model
from .zoedepth.utils.config import get_config
from .zoedepth.models import zoedepth_nk
from deforum.utils.device import empty_cache, get_device

z_conf = {
    "model": {
        "name": "ZoeDepthNK",
        "version_name": "v1",
        "bin_conf": [
            {
                "name": "nyu",
                "n_bins": 64,
                "min_depth": 1e-3,
                "max_depth": 10.0
            },
            {
                "name": "kitti",
                "n_bins": 64,
                "min_depth": 1e-3,
                "max_depth": 80.0
            }
        ],
        "bin_embedding_dim": 128,
        "bin_centers_type": "softplus",
        "n_attractors": [16, 8, 4, 1],
        "attractor_alpha": 1000,
        "attractor_gamma": 2,
        "attractor_kind": "mean",
        "attractor_type": "inv",
        "min_temp": 0.0212,
        "max_temp": 50.0,
        "memory_efficient": True,
        "midas_model_type": "DPT_BEiT_L_384",
        "img_size": [384, 512]
    },

    "train": {
        "train_midas": True,
        "use_pretrained_midas": True,
        "trainer": "zoedepth_nk",
        "epochs": 5,
        "bs": 16,
        "optim_kwargs": {"lr": 0.0002512, "wd": 0.01},
        "sched_kwargs": {"div_factor": 1, "final_div_factor": 10000, "pct_start": 0.7, "three_phase": False,
                         "cycle_momentum": True},
        "same_lr": False,
        "w_si": 1,
        "w_domain": 100,
        "avoid_boundary": False,
        "random_crop": False,
        "input_width": 640,
        "input_height": 480,
        "w_grad": 0,
        "w_reg": 0,
        "midas_lr_factor": 10,
        "encoder_lr_factor": 10,
        "pos_enc_lr_factor": 10
    },

    "infer": {
        "train_midas": False,
        "pretrained_resource": "url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt",
        "use_pretrained_midas": False,
        "force_keep_ar": True
    },

    "eval": {
        "train_midas": False,
        "pretrained_resource": "url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt",
        "use_pretrained_midas": False
    }
}


class ZoeDepth:
    def __init__(self, width=512, height=512):
        #conf = get_config("zoedepth_nk", "infer")
        conf = z_conf
        conf["img_size"] = [width, height]
        #self.model_zoe = build_model(conf)
        model = zoedepth_nk
        get_version = getattr(model, "get_version")
        # print("CONF", conf)
        self.model_zoe = ZoeDepthNK.build_from_config(
            {**conf["model"], "pretrained_resource": conf["infer"]["pretrained_resource"]})


        self.DEVICE = get_device()
        self.zoe = self.model_zoe.to(self.DEVICE)
        self.width = width
        self.height = height
        
    def predict(self, image):
        # ZoeD_NK is trained at 384x512: run near that size (aspect kept, multiple of 32) instead of the
        # full frame, and under fp16 autocast on GPUs. ~10x faster than full-res fp32, same depth layout.
        w, h = image.size
        scale = 384 / min(w, h)
        self.zoe.core.prep.resizer._Resize__width = max(32, round(w * scale / 32) * 32)
        self.zoe.core.prep.resizer._Resize__height = max(32, round(h * scale / 32) * 32)
        dev = torch.device(self.DEVICE)
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            depth_tensor = self.zoe.infer_pil(image, output_type="tensor")
        return depth_tensor.float()
        
    def to(self, device):
        self.DEVICE = device
        self.zoe = self.model_zoe.to(device)
        
    def save_raw_depth(self, depth, filepath):
        depth.save(filepath, format='PNG', mode='I;16')
    
    def delete(self):
        del self.model_zoe
        del self.zoe