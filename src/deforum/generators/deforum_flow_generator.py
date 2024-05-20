import os
import random

import cv2
import numpy as np
import torch

from .deforum_flow_consistency import make_consistency
from deforum.utils.image_utils import (get_resized_image_from_filename,
                                 custom_gaussian_blur,
                                 center_crop_image)
from deforum.utils.logging_config import logger


def get_flow_for_hybrid_motion(frame_idx,
                               dimensions,
                               inputfiles,
                               hybrid_frame_path,
                               prev_flow,
                               method,
                               raft_model,
                               consistency_check=True,
                               consistency_blur=0,
                               do_flow_visualization=False):
    # logger.info(f"Calculating {method} optical flow {'w/consistency mask' if consistency_check else ''} for frames {frame_idx} to {frame_idx + 1}")
    i1 = get_resized_image_from_filename(str(inputfiles[frame_idx]), dimensions)
    i2 = get_resized_image_from_filename(str(inputfiles[frame_idx + 1]), dimensions)
    if consistency_check:
        flow, reliable_flow = get_reliable_flow_from_images(i1, i2, method, raft_model, prev_flow,
                                                            consistency_blur)  # forward flow w/backward consistency check
        if do_flow_visualization: save_flow_mask_visualization(frame_idx, reliable_flow, hybrid_frame_path)
    else:
        flow = get_flow_from_images(i1, i2, method, raft_model, prev_flow)  # old single flow forward
    if do_flow_visualization: save_flow_visualization(frame_idx, dimensions, flow, inputfiles, hybrid_frame_path)
    return flow


def get_flow_for_hybrid_motion_prev(frame_idx,
                                    dimensions,
                                    inputfiles,
                                    hybrid_frame_path,
                                    prev_flow,
                                    prev_img,
                                    method,
                                    raft_model,
                                    consistency_check=True,
                                    consistency_blur=0,
                                    do_flow_visualization=False):
    # logger.info(f"Calculating {method} optical flow {'w/consistency mask' if consistency_check else ''} for frames {frame_idx} to {frame_idx + 1}")
    reliable_flow = None
    # first handle invalid images by returning default flow
    height, width = prev_img.shape[:2]
    if height == 0 or width == 0:
        flow = get_hybrid_motion_default_flow(dimensions)
    else:
        i1 = prev_img.astype(np.uint8)
        i2 = get_resized_image_from_filename(str(inputfiles[frame_idx + 1]), dimensions)
        if consistency_check:
            flow, reliable_flow = get_reliable_flow_from_images(i1, i2, method, raft_model, prev_flow,
                                                                consistency_blur)  # forward flow w/backward consistency check
            if do_flow_visualization: save_flow_mask_visualization(frame_idx, reliable_flow, hybrid_frame_path)
        else:
            flow = get_flow_from_images(i1, i2, method, raft_model, prev_flow)
    if do_flow_visualization: save_flow_visualization(frame_idx, dimensions, flow, inputfiles, hybrid_frame_path)
    return flow

def get_flow_for_hybrid_motion_prev_imgs(current_img,
                                        prev_flow,
                                        prev_img,
                                        method,
                                        raft_model,
                                        consistency_check=True,
                                        consistency_blur=0):
    # print(
        # f"Calculating {method} optical flow {'w/consistency mask' if consistency_check else ''} for frames {frame_idx} to {frame_idx + 1}")
    reliable_flow = None
    # first handle invalid images by returning default flow
    height, width = prev_img.shape[:2]
    # if height == 0 or width == 0:
    #     flow = get_hybrid_motion_default_flow(dimensions)
    # else:
    i1 = prev_img.astype(np.uint8)
    i2 = current_img.astype(np.uint8)
    if consistency_check:


        flow, reliable_flow = get_reliable_flow_from_images(i1, i2, method, raft_model, prev_flow,
                                                            consistency_blur)  # forward flow w/backward consistency check
        #if do_flow_visualization: save_flow_mask_visualization(frame_idx, reliable_flow, hybrid_frame_path)
    else:
        flow = get_flow_from_images(i1, i2, method, raft_model, prev_flow)
    #if do_flow_visualization: save_flow_visualization(frame_idx, dimensions, flow, inputfiles, hybrid_frame_path)
    return flow


def get_reliable_flow_from_images(i1,
                                  i2,
                                  method,
                                  raft_model,
                                  prev_flow,
                                  consistency_blur,
                                  reliability=0):
    flow_forward = get_flow_from_images(i1, i2, method, raft_model, prev_flow)
    flow_backward = get_flow_from_images(i2, i1, method, raft_model, None)
    reliable_flow = make_consistency(flow_forward, flow_backward, edges_unreliable=False)
    if consistency_blur > 0:
        reliable_flow = custom_gaussian_blur(reliable_flow.astype(np.float32), 1, consistency_blur)
    return filter_flow(flow_forward, reliable_flow, consistency_blur, reliability), reliable_flow


def filter_flow(flow,
                reliable_flow,
                reliability=0.5,
                consistency_blur=0):
    # reliability from reliabile flow: -0.75 is bad, 0 is meh/outside, 1 is great
    # Create a mask from the first channel of the reliable_flow array
    mask = reliable_flow[..., 0]
    # to set everything to 1 or 0 based on reliability
    mask = np.where(mask >= reliability, 1, 0)
    # Expand the mask to match the shape of the forward_flow array
    mask = np.repeat(mask[..., np.newaxis], flow.shape[2], axis=2)
    # Apply the mask to the flow
    return flow * mask


def get_custom_optical_flow(i1, i2):
    """
    Custom optical flow implementation that aims to be faster than existing methods.

    Args:
        i1: First input image (numpy array).
        i2: Second input image (numpy array).

    Returns:
        flow: Computed optical flow (numpy array).
    """
    # Convert images to grayscale
    i1_gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2_gray = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    # Convert images to PyTorch tensors
    i1_tensor = torch.from_numpy(i1_gray).half().to('cuda').unsqueeze(0).unsqueeze(0)
    i2_tensor = torch.from_numpy(i2_gray).half().to('cuda').unsqueeze(0).unsqueeze(0)

    # Normalize images
    i1_tensor = i1_tensor / 255.0
    i2_tensor = i2_tensor / 255.0

    # Define Sobel kernels for gradient computation
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to('cuda')
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to('cuda')

    # Compute gradients
    grad_x1 = torch.nn.functional.conv2d(torch.nn.functional.pad(i1_tensor, (1, 1, 1, 1)), sobel_x)
    grad_y1 = torch.nn.functional.conv2d(torch.nn.functional.pad(i1_tensor, (1, 1, 1, 1)), sobel_y)

    grad_x2 = torch.nn.functional.conv2d(torch.nn.functional.pad(i2_tensor, (1, 1, 1, 1)), sobel_x)
    grad_y2 = torch.nn.functional.conv2d(torch.nn.functional.pad(i2_tensor, (1, 1, 1, 1)), sobel_y)

    # Compute temporal gradients
    It = i2_tensor - i1_tensor

    # Compute optical flow using the gradients
    flow_x = (grad_x1 + grad_x2) / 2
    flow_y = (grad_y1 + grad_y2) / 2

    # Combine gradients into a single tensor for flow computation
    flow = torch.cat((flow_x, flow_y), dim=1).squeeze(0).permute(1, 2, 0)

    # Normalize the flow to match the scale of other methods
    flow = (flow / flow.max()) * 255.0

    # Convert flow back to numpy array
    flow_np = flow.cpu().numpy()

    return flow_np
def get_flow_from_images(i1, i2, method, raft_model, prev_flow=None):
    if method == "RAFT":
        if raft_model is None:
            raise Exception("RAFT Model not provided to get_flow_from_images function, cannot continue.")
        return get_flow_from_images_RAFT(i1, i2, raft_model)
    elif method == "DIS Medium":
        return get_flow_from_images_DIS(i1, i2, 'medium', prev_flow)
    elif method == "DIS Fine":
        return get_flow_from_images_DIS(i1, i2, 'fine', prev_flow)
    elif method == "DenseRLOF":  # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_Dense_RLOF(i1, i2, prev_flow)
    elif method == "SF":  # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_SF(i1, i2, prev_flow)
    elif method == "DualTVL1":  # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_DualTVL1(i1, i2, prev_flow)
    elif method == "DeepFlow":  # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_DeepFlow(i1, i2, prev_flow)
    elif method == "PCAFlow":  # Unused - requires running opencv-contrib-python (full opencv) INSTEAD of opencv-python
        return get_flow_from_images_PCAFlow(i1, i2, prev_flow)
    elif method == "Farneback":  # Farneback Normal:
        return get_flow_from_images_Farneback(i1, i2, "normal", prev_flow)
    elif method == "DIS Medium Torch":  # Farneback Normal:
        return get_flow_from_images_DIS_torch(i1, i2, "medium", prev_flow)
    elif method == "Farneback Torch":  # Farneback Normal:
        return get_flow_from_images_Farneback_torch(i1, i2, "normal", prev_flow)
    elif method == "Custom":
        return get_custom_optical_flow(i1, i2)
    # if we reached this point, something went wrong. raise an error:
    raise RuntimeError(f"Invald flow method name: '{method}'")


def get_flow_from_images_RAFT(i1, i2, raft_model):
    flow = raft_model.predict(i1, i2)
    return flow


def get_flow_from_images_DIS(i1, i2, preset, prev_flow):
    preset_code = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    # DIS PRESETS CHART KEY: finest scale, grad desc its, patch size
    # DIS_MEDIUM: 1, 25, 8 | DIS_FAST: 2, 16, 8 | DIS_ULTRAFAST: 2, 12, 8
    if preset == 'medium':
        preset_code = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    elif preset == 'fast':
        preset_code = cv2.DISOPTICAL_FLOW_PRESET_FAST
    elif preset == 'ultrafast':
        preset_code = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    elif preset in ['slow', 'fine']:
        preset_code = None
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(preset_code)
    # custom presets
    if preset == 'slow':
        dis.setGradientDescentIterations(192)
        dis.setFinestScale(1)
        dis.setPatchSize(8)
        dis.setPatchStride(4)
    if preset == 'fine':
        dis.setGradientDescentIterations(192)
        dis.setFinestScale(0)
        dis.setPatchSize(8)
        dis.setPatchStride(4)
    return dis.calc(i1, i2, prev_flow)

def get_flow_from_images_DIS_torch(i1, i2, preset, prev_flow):
    import torch.nn.functional as F
    # Convert images to grayscale
    i1_gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2_gray = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    # Convert images to PyTorch tensors and move to GPU
    i1_tensor = torch.from_numpy(i1_gray).float().unsqueeze(0).unsqueeze(0).to('cuda')
    i2_tensor = torch.from_numpy(i2_gray).float().unsqueeze(0).unsqueeze(0).to('cuda')

    # Define kernel sizes based on preset
    if preset == 'medium':
        grad_desc_iterations = 25
        finest_scale = 1
        patch_size = 8
    elif preset == 'fine':
        grad_desc_iterations = 192
        finest_scale = 0
        patch_size = 8
    else:  # default to medium
        grad_desc_iterations = 25
        finest_scale = 1
        patch_size = 8

    # Define Sobel kernels for gradient computation
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to('cuda')
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to('cuda')

    # Compute gradients
    grad_x1 = F.conv2d(F.pad(i1_tensor, (1, 1, 1, 1)), sobel_x)
    grad_y1 = F.conv2d(F.pad(i1_tensor, (1, 1, 1, 1)), sobel_y)

    grad_x2 = F.conv2d(F.pad(i2_tensor, (1, 1, 1, 1)), sobel_x)
    grad_y2 = F.conv2d(F.pad(i2_tensor, (1, 1, 1, 1)), sobel_y)

    # Initialize flow tensors
    if prev_flow is None:
        u = torch.zeros_like(i1_tensor).to('cuda')
        v = torch.zeros_like(i1_tensor).to('cuda')
    else:
        u = torch.from_numpy(prev_flow[..., 0]).float().unsqueeze(0).unsqueeze(0).to('cuda')
        v = torch.from_numpy(prev_flow[..., 1]).float().unsqueeze(0).unsqueeze(0).to('cuda')


    # Iterative refinement
    for _ in range(grad_desc_iterations):
        # Create the grid for warping
        N, C, H, W = i1_tensor.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid = torch.stack((grid_x, grid_y), dim=2).float().to('cuda')  # Shape: [H, W, 2]
        grid = grid.unsqueeze(0).permute(0, 3, 1, 2)  # Shape: [1, 2, H, W]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        # Adjust the grid by the current flow estimates
        flow_grid = torch.cat((u, v), dim=1) + grid  # Shape: [1, 2, H, W]
        flow_grid = flow_grid.permute(0, 2, 3, 1)  # Shape: [1, H, W, 2]

        # Warp the second image to the first image based on current flow estimates
        i2_warped = F.grid_sample(i2_tensor, flow_grid, align_corners=True)

        # Compute differences
        Ix = (grad_x1 + grad_x2) / 2
        Iy = (grad_y1 + grad_y2) / 2
        It = i2_warped - i1_tensor

        # Update flow estimates
        u = u - Ix * (Ix * u + Iy * v + It) / (Ix * Ix + Iy * Iy + 1e-8)
        v = v - Iy * (Ix * u + Iy * v + It) / (Ix * Ix + Iy * Iy + 1e-8)

    # Combine flow estimates into a single tensor
    flow = torch.cat((u, v), dim=1).squeeze(0).permute(1, 2, 0)

    # Convert flow back to numpy array
    flow_np = flow.cpu().numpy()

    return flow_np
def get_flow_from_images_Dense_RLOF(i1, i2, last_flow=None):
    return cv2.optflow.calcOpticalFlowDenseRLOF(i1, i2, flow=last_flow)


def get_flow_from_images_SF(i1, i2, last_flow=None, layers=3, averaging_block_size=2, max_flow=4):
    return cv2.optflow.calcOpticalFlowSF(i1, i2, layers, averaging_block_size, max_flow)


def get_flow_from_images_DualTVL1(i1, i2, prev_flow):
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    f = cv2.optflow.DualTVL1OpticalFlow_create()
    return f.calc(i1, i2, prev_flow)


def get_flow_from_images_DeepFlow(i1, i2, prev_flow):
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    f = cv2.optflow.createOptFlow_DeepFlow()
    return f.calc(i1, i2, prev_flow)


def get_flow_from_images_PCAFlow(i1, i2, prev_flow):
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    f = cv2.optflow.createOptFlow_PCAFlow()
    return f.calc(i1, i2, prev_flow)

import torch.nn.functional as F
import numpy as np
import cv2

# Compute gradients
try:
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to('cuda')
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to('cuda')
except Exception:
    logger.warning("Could not create sobel kernels for optical flow computation. This most likely means you do not have CUDA device available.")
    sobel_x = None
    sobel_y = None

@torch.inference_mode()
def get_flow_from_images_Farneback_torch(i1, i2, preset="normal", last_flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                   iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
    # Convert images to grayscale
    i1_gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2_gray = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

    # Convert images to PyTorch tensors and move to GPU
    i1_tensor = torch.from_numpy(i1_gray).float().unsqueeze(0).unsqueeze(0).to('cuda')
    i2_tensor = torch.from_numpy(i2_gray).float().unsqueeze(0).unsqueeze(0).to('cuda')

    # Define parameters based on preset
    if preset == "fine":
        pyr_scale = 0.5
        levels = 13
        winsize = 77
        iterations = 13
        poly_n = 15
        poly_sigma = 0.8
    else:  # normal
        pyr_scale = 0.5
        levels = 5
        winsize = 21
        iterations = 5
        poly_n = 7
        poly_sigma = 1.2

    # # Initialize flow tensors
    # u = torch.zeros_like(i1_tensor).to('cuda')
    # v = torch.zeros_like(i1_tensor).to('cuda')
    if last_flow is None:
        u = torch.zeros_like(i1_tensor).to('cuda')
        v = torch.zeros_like(i1_tensor).to('cuda')
    else:
        u = torch.from_numpy(last_flow[..., 0]).float().unsqueeze(0).unsqueeze(0).to('cuda')
        v = torch.from_numpy(last_flow[..., 1]).float().unsqueeze(0).unsqueeze(0).to('cuda')


    for level in range(levels):
        scale = pyr_scale ** (levels - level - 1)
        new_size = (int(i1_tensor.size(3) * scale), int(i1_tensor.size(2) * scale))

        # Resize images
        i1_resized = F.interpolate(i1_tensor, size=new_size, mode='bilinear', align_corners=False)
        i2_resized = F.interpolate(i2_tensor, size=new_size, mode='bilinear', align_corners=False)


        grad_x1 = F.conv2d(F.pad(i1_resized, (1, 1, 1, 1)), sobel_x)
        grad_y1 = F.conv2d(F.pad(i1_resized, (1, 1, 1, 1)), sobel_y)
        grad_x2 = F.conv2d(F.pad(i2_resized, (1, 1, 1, 1)), sobel_x)
        grad_y2 = F.conv2d(F.pad(i2_resized, (1, 1, 1, 1)), sobel_y)

        # Resize flow to the current pyramid level
        u_resized = F.interpolate(u, size=new_size, mode='bilinear', align_corners=False)
        v_resized = F.interpolate(v, size=new_size, mode='bilinear', align_corners=False)

        for _ in range(iterations):
            # Create the grid for warping
            N, C, H, W = i1_resized.size()
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device='cuda'), torch.arange(W, device='cuda'))
            grid = torch.stack((grid_x, grid_y), dim=2).float().to('cuda')  # Shape: [H, W, 2]
            grid = grid.unsqueeze(0).permute(0, 3, 1, 2)  # Shape: [1, 2, H, W]
            grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
            grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

            # Adjust the grid by the current flow estimates
            flow_grid = torch.cat((u_resized, v_resized), dim=1) + grid  # Shape: [1, 2, H, W]
            flow_grid = flow_grid.permute(0, 2, 3, 1)  # Shape: [1, H, W, 2]

            # Warp the second image to the first image based on current flow estimates
            i2_warped = F.grid_sample(i2_resized, flow_grid, align_corners=True)

            # Compute differences
            Ix = (grad_x1 + grad_x2) / 2
            Iy = (grad_y1 + grad_y2) / 2
            It = i2_warped - i1_resized

            # Update flow estimates
            u_resized = u_resized - Ix * (Ix * u_resized + Iy * v_resized + It) / (Ix * Ix + Iy * Iy + 1e-8)
            v_resized = v_resized - Iy * (Ix * u_resized + Iy * v_resized + It) / (Ix * Ix + Iy * Iy + 1e-8)

        # Upsample the flow to the next pyramid level
        if level != levels - 1:
            u = F.interpolate(u_resized, size=(i1_tensor.size(2), i1_tensor.size(3)), mode='bilinear', align_corners=False) * (1.0 / pyr_scale)
            v = F.interpolate(v_resized, size=(i1_tensor.size(2), i1_tensor.size(3)), mode='bilinear', align_corners=False) * (1.0 / pyr_scale)
        else:
            u = u_resized
            v = v_resized

    # Combine flow estimates into a single tensor
    flow = torch.cat((u, v), dim=1).squeeze(0).permute(1, 2, 0)

    # Convert flow back to numpy array
    flow_np = flow.cpu().numpy()

    return flow_np
def get_flow_from_images_Farneback(i1, i2, preset="normal", last_flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                   iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN  # Specify the operation flags
    pyr_scale = 0.5  # The image scale (<1) to build pyramids for each image
    if preset == "fine":
        levels = 13  # The number of pyramid layers, including the initial image
        winsize = 77  # The averaging window size
        iterations = 13  # The number of iterations at each pyramid level
        poly_n = 15  # The size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma = 0.8  # The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion
    else:  # "normal"
        levels = 5  # The number of pyramid layers, including the initial image
        winsize = 21  # The averaging window size
        iterations = 5  # The number of iterations at each pyramid level
        poly_n = 7  # The size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma = 1.2  # The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    flags = 0  # flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    flow = cv2.calcOpticalFlowFarneback(i1, i2, last_flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma,
                                        flags)
    return flow


def save_flow_visualization(frame_idx, dimensions, flow, inputfiles, hybrid_frame_path):
    flow_img_file = os.path.join(hybrid_frame_path, f"flow{frame_idx:09}.jpg")
    flow_img = cv2.imread(str(inputfiles[frame_idx]))
    flow_img = cv2.resize(flow_img, (dimensions[0], dimensions[1]), cv2.INTER_AREA)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2GRAY)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_GRAY2BGR)
    flow_img = draw_flow_lines_in_grid_in_color(flow_img, flow)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(flow_img_file, flow_img)
    logger.info(f"Saved optical flow visualization: {flow_img_file}")


def save_flow_mask_visualization(frame_idx, reliable_flow, hybrid_frame_path, color=True):
    flow_mask_img_file = os.path.join(hybrid_frame_path, f"flow_mask{frame_idx:09}.jpg")
    if color:
        # Normalize the reliable_flow array to the range [0, 255]
        normalized_reliable_flow = (reliable_flow - reliable_flow.min()) / (
                reliable_flow.max() - reliable_flow.min()) * 255
        # Change the data type to np.uint8
        mask_image = normalized_reliable_flow.astype(np.uint8)
    else:
        # Extract the first channel of the reliable_flow array
        first_channel = reliable_flow[..., 0]
        # Normalize the first channel to the range [0, 255]
        normalized_first_channel = (first_channel - first_channel.min()) / (
                first_channel.max() - first_channel.min()) * 255
        # Change the data type to np.uint8
        grayscale_image = normalized_first_channel.astype(np.uint8)
        # Replicate the grayscale channel three times to form a BGR image
        mask_image = np.stack((grayscale_image, grayscale_image, grayscale_image), axis=2)
    cv2.imwrite(flow_mask_img_file, mask_image)
    logger.info(f"Saved mask flow visualization: {flow_mask_img_file}")


def reliable_flow_to_image(reliable_flow):
    # Extract the first channel of the reliable_flow array
    first_channel = reliable_flow[..., 0]
    # Normalize the first channel to the range [0, 255]
    normalized_first_channel = (first_channel - first_channel.min()) / (first_channel.max() - first_channel.min()) * 255
    # Change the data type to np.uint8
    grayscale_image = normalized_first_channel.astype(np.uint8)
    # Replicate the grayscale channel three times to form a BGR image
    bgr_image = np.stack((grayscale_image, grayscale_image, grayscale_image), axis=2)
    return bgr_image


def draw_flow_lines_in_grid_in_color(img, flow, step=8, magnitude_multiplier=1, min_magnitude=0, max_magnitude=10000):
    flow = flow * magnitude_multiplier
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    vis = cv2.add(vis, bgr)

    # Iterate through the lines
    for (x1, y1), (x2, y2) in lines:
        # Calculate the magnitude of the line
        magnitude = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Only draw the line if it falls within the magnitude range
        if min_magnitude <= magnitude <= max_magnitude:
            b = int(bgr[y1, x1, 0])
            g = int(bgr[y1, x1, 1])
            r = int(bgr[y1, x1, 2])
            color = (b, g, r)
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, thickness=1, tipLength=0.1)
    return vis


def draw_flow_lines_in_color(img, flow, threshold=3, magnitude_multiplier=1, min_magnitude=0, max_magnitude=10000):
    # h, w = img.shape[:2]
    vis = img.copy()  # Create a copy of the input image

    # Find the locations in the flow field where the magnitude of the flow is greater than the threshold
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    idx = np.where(mag > threshold)

    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV image to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Add color from bgr
    vis = cv2.add(vis, bgr)

    # Draw an arrow at each of these locations to indicate the direction of the flow
    for i, (y, x) in enumerate(zip(idx[0], idx[1])):
        # Calculate the magnitude of the line
        x2 = x + magnitude_multiplier * int(flow[y, x, 0])
        y2 = y + magnitude_multiplier * int(flow[y, x, 1])
        magnitude = np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)

        # Only draw the line if it falls within the magnitude range
        if min_magnitude <= magnitude <= max_magnitude:
            if i % random.randint(100, 200) == 0:
                b = int(bgr[y, x, 0])
                g = int(bgr[y, x, 1])
                r = int(bgr[y, x, 2])
                color = (b, g, r)
                cv2.arrowedLine(vis, (x, y), (x2, y2), color, thickness=1, tipLength=0.25)

    return vis


def extend_flow(flow, w, h):
    # Get the shape of the original flow image
    flow_h, flow_w = flow.shape[:2]
    # Calculate the position of the image in the new image
    x_offset = int((w - flow_w) / 2)
    y_offset = int((h - flow_h) / 2)
    # Generate the X and Y grids
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    # Create the new flow image and set it to the X and Y grids
    new_flow = np.dstack((x_grid, y_grid)).astype(np.float32)
    # Shift the values of the original flow by the size of the border
    flow[:, :, 0] += x_offset
    flow[:, :, 1] += y_offset
    # Overwrite the middle of the grid with the original flow
    new_flow[y_offset:y_offset + flow_h, x_offset:x_offset + flow_w, :] = flow
    # Return the extended image
    return new_flow


def abs_flow_to_rel_flow(flow, width, height):
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    max_flow_x = np.max(np.abs(fx))
    max_flow_y = np.max(np.abs(fy))
    max_flow = max(max_flow_x, max_flow_y)

    # Avoid division by zero by ensuring max_flow, width, and height are non-zero
    max_flow = max(max_flow, 1e-8)
    width = max(width, 1e-8)
    height = max(height, 1e-8)

    rel_fx = fx / (max_flow * width)
    rel_fy = fy / (max_flow * height)
    return np.dstack((rel_fx, rel_fy))

def rel_flow_to_abs_flow(rel_flow, width, height):
    rel_fx, rel_fy = rel_flow[:, :, 0], rel_flow[:, :, 1]

    max_flow_x = np.max(np.abs(rel_fx * width))
    max_flow_y = np.max(np.abs(rel_fy * height))
    max_flow = max(max_flow_x, max_flow_y)

    # Avoid division by zero by ensuring max_flow, width, and height are non-zero
    max_flow = max(max_flow, 1e-8)
    width = max(width, 1e-8)
    height = max(height, 1e-8)

    fx = rel_fx * (max_flow * width)
    fy = rel_fy * (max_flow * height)
    return np.dstack((fx, fy))


def remap(img,
          flow):
    border_mode = cv2.BORDER_REFLECT_101
    h, w = img.shape[:2]
    displacement = int(h * 0.25), int(w * 0.25)
    larger_img = cv2.copyMakeBorder(img, displacement[0], displacement[0], displacement[1], displacement[1],
                                    border_mode)
    lh, lw = larger_img.shape[:2]
    larger_flow = extend_flow(flow, lw, lh)
    remapped_img = cv2.remap(larger_img, larger_flow, None, cv2.INTER_LINEAR, border_mode)
    output_img = center_crop_image(remapped_img, w, h)
    return output_img


def get_hybrid_motion_default_flow(dimensions):
    cols, rows = dimensions
    flow = np.zeros((rows, cols, 2), np.float32)
    return flow
