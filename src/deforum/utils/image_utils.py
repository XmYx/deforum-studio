import gc
import os
import re
import socket
import time
from threading import Thread

import PIL.Image
import cv2
import numpy as np
import requests
import torch
import torchvision.transforms.functional as TF
from PIL import (Image, ImageChops, ImageOps)

from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms

from .deforum_word_masking_util import get_word_mask
from .video_frame_utils import get_frame_name
# from modules.shared import opts
from ..utils.gradio_utils import clean_gradio_path_strings

from deforum.utils.logging_config import logger

# from ainodes_frontend import singleton as gs
DEBUG_MODE = True


# IMAGE FUNCTIONS

def maintain_colors(prev_img, color_match_sample, mode):
    is_skimage_v20_or_higher = True

    match_histograms_kwargs = {'channel_axis': -1} if is_skimage_v20_or_higher else {'multichannel': True}

    if mode == 'RGB':
        return cv2.cvtColor(match_histograms(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB), color_match_sample, **match_histograms_kwargs), cv2.COLOR_RGB2BGR)
    elif mode == 'HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_BGR2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, **match_histograms_kwargs)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2BGR)
    else:  # LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_BGR2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, **match_histograms_kwargs)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


def load_image(image_path: str):
    if isinstance(image_path, str):
        image_path = clean_gradio_path_strings(image_path)
        if image_path.startswith('http://') or image_path.startswith('https://'):
            try:
                host = socket.gethostbyname("www.google.com")
                s = socket.create_connection((host, 80), 2)
                s.close()
            except:
                raise ConnectionError(
                    "There is no active internet connection available - please use local masks and init files only.")

            try:
                response = requests.get(image_path, stream=True)
            except requests.exceptions.RequestException as e:
                raise ConnectionError("Failed to download image due to no internet connection. Error: {}".format(e))
            if response.status_code == 404 or response.status_code != 200:
                raise ConnectionError("Init image url or mask image url is not valid")
            image = Image.open(response.raw).convert('RGB')
        else:
            if not os.path.exists(image_path):
                raise RuntimeError("Init image path or mask image path is not valid")
            image = Image.open(image_path).convert('RGB')
        return image
    elif isinstance(image_path, PIL.Image.Image):
        return image_path


def blank_if_none(mask, w, h, mode):
    return Image.new(mode, (w, h), 0) if mask is None else mask


def none_if_blank(mask):
    return None if mask.getextrema() == (0, 0) else mask


def get_resized_image_from_filename(im, dimensions):
    img = cv2.imread(im)
    return cv2.resize(img, (dimensions[0], dimensions[1]), cv2.INTER_AREA)


def center_crop_image(img, w, h):
    y, x, _ = img.shape
    width_indent = int((x - w) / 2)
    height_indent = int((y - h) / 2)
    cropped_img = img[height_indent:y - height_indent, width_indent:x - width_indent]
    return cropped_img


def autocontrast_grayscale(image, low_cutoff=0, high_cutoff=100):
    # Perform autocontrast on a grayscale np array image.
    # Find the minimum and maximum values in the image
    min_val = np.percentile(image, low_cutoff)
    max_val = np.percentile(image, high_cutoff)

    # Scale the image so that the minimum value is 0 and the maximum value is 255
    image = 255 * (image - min_val) / (max_val - min_val)

    # Clip values that fall outside the range [0, 255]
    image = np.clip(image, 0, 255)

    return image


def image_transform_ransac(image_cv2, m, hybrid_motion, depth=None):
    if hybrid_motion == "Perspective":
        return image_transform_perspective(image_cv2, m, depth)
    else:  # Affine
        return image_transform_affine(image_cv2, m, depth)


def image_transform_optical_flow(img, flow, flow_factor):
    # if flow factor not normal, calculate flow factor
    if flow_factor != 1:
        flow = flow * flow_factor
    # flow is reversed, so you need to reverse it:
    flow = -flow
    h, w = img.shape[:2]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    return remap(img, flow)


def image_transform_affine(image_cv2, m, depth=None):
    if depth is None:
        return cv2.warpAffine(
            image_cv2,
            m,
            (image_cv2.shape[1], image_cv2.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101
        )
    else:  # NEED TO IMPLEMENT THE FOLLOWING FUNCTION
        return image_cv2
        # return depth_based_affine_warp(
        #     image_cv2,
        #     depth,
        #     M
        # )


def image_transform_perspective(image_cv2, m, depth=None):
    if depth is None:
        return cv2.warpPerspective(
            image_cv2,
            m,
            (image_cv2.shape[1], image_cv2.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101
        )
    else:
        return image_cv2  # NEED TO IMPLEMENT THE FOLLOWING FUNCTION
        # return render_3d_perspective(
        #     image_cv2,
        #     depth,
        #     M
        # )


def custom_gaussian_blur(input_array, blur_size, sigma):
    return gaussian_filter(input_array, sigma=(sigma, sigma, 0), order=0, mode='constant', cval=0.0, truncate=blur_size)


# MASK FUNCTIONS

def load_image_with_mask(path: str, shape=None, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    image = load_image(path)
    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    if shape is not None:
        image = image.resize(shape, resample=Image.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

        # check using init image alpha as mask if mask is not blank
        extrema = mask_image.getextrema()
        if (extrema == (0, 0)) or extrema == (255, 255):
            logger.info(
                "use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha "
                "channel is blank.")
            logger.info("ignoring alpha as mask.")
            mask_image = None

    return image, mask_image


def prepare_mask(mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0):
    """
    prepares mask for use in webui
    """
    if isinstance(mask_input, Image.Image):
        mask = mask_input
    else:
        mask = load_image(mask_input)
    mask = mask.resize(mask_shape, resample=Image.LANCZOS)
    # TODO I've added the tensor and back conversion, need to check if it works as intended (mix)
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(img=torch.from_numpy(np.array(mask)), brightness_factor=mask_brightness_adjust)
        mask = Image.fromarray(np.array(mask))
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(torch.from_numpy(np.array(mask)), mask_contrast_adjust)
        mask = Image.fromarray(np.array(mask))
    mask = mask.convert('L')
    return mask


# "check_mask_for_errors" may have prevented errors in composable masks,
# but it CAUSES errors on any frame where it's all black.
# Bypassing the check below until we can fix it even better.
# This may break composable masks, but it makes ACTUAL masks usable.
def check_mask_for_errors(mask_input, invert_mask=False):
    extrema = mask_input.getextrema()
    if invert_mask:
        if extrema == (255, 255):
            logger.info("after inverting mask will be blank. ignoring mask")
            return None
    elif extrema == (0, 0):
        logger.info("mask is blank. ignoring mask")
        return None
    else:
        return mask_input


def get_mask(args):
    # return check_mask_for_errors(
    #     prepare_mask(args.mask_file, (args.width, args.height), args.mask_contrast_adjust, args.mask_brightness_adjust)
    # )
    return prepare_mask(args.mask_file, (args.width, args.height), args.mask_contrast_adjust, args.mask_brightness_adjust)


def get_mask_from_file(mask_file, args):
    # return check_mask_for_errors(
    #     prepare_mask(mask_file, (args.width, args.height), args.mask_contrast_adjust, args.mask_brightness_adjust)
    # )
    return prepare_mask(mask_file, (args.width, args.height), args.mask_contrast_adjust, args.mask_brightness_adjust)


# def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0, mask=None):
#     if amount == 0:
#         return img
#     # Return a sharpened version of the image, using an unsharp mask.
#     # If mask is not None, only areas under mask are handled
#     blurred = cv2.GaussianBlur(img, kernel_size, sigma)
#     sharpened = float(amount + 1) * img - float(amount) * blurred
#     sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
#     sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
#     sharpened = sharpened.round().astype(np.uint8)
#     if threshold > 0:
#         low_contrast_mask = np.absolute(img - blurred) < threshold
#         np.copyto(sharpened, img, where=low_contrast_mask)
#     if mask is not None:
#         mask = np.array(mask)
#         masked_sharpened = cv2.bitwise_and(sharpened, sharpened, mask=mask)
#         masked_img = cv2.bitwise_and(img, img, mask=255 - mask)
#         sharpened = cv2.add(masked_img, masked_sharpened)
#     return sharpened

def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0, mask=None):
    if amount == 0:
        return img

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_tensor = torch.from_numpy(img).float().to(device)
    blurred_tensor = cv2.GaussianBlur(img, kernel_size, sigma)
    blurred_tensor = torch.from_numpy(blurred_tensor).float().to(device)

    sharpened_tensor = (amount + 1.0) * img_tensor - amount * blurred_tensor
    sharpened_tensor = torch.clamp(sharpened_tensor, 0, 255).round().to(torch.uint8)

    if threshold > 0:
        low_contrast_mask = torch.abs(img_tensor - blurred_tensor) < threshold
        sharpened_tensor = torch.where(low_contrast_mask, img_tensor, sharpened_tensor)

    if mask is not None:
        mask_tensor = torch.from_numpy(np.array(mask)).to(device)
        masked_sharpened = sharpened_tensor * mask_tensor
        masked_img = img_tensor * (1 - mask_tensor)
        sharpened_tensor = masked_sharpened + masked_img

    return sharpened_tensor.cpu().numpy()
def do_overlay_mask(args, anim_args, img, frame_idx, is_bgr_array=False):
    current_mask = None
    current_frame = None
    if is_bgr_array:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    if anim_args.use_mask_video:
        current_mask = Image.open(
            os.path.join(args.outdir, 'maskframes', get_frame_name(anim_args.video_mask_path) + f"{frame_idx:09}.jpg"))
        current_frame = Image.open(
            os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame_idx:09}.jpg"))
    elif args.use_mask:
        current_mask = args.mask_image if args.mask_image is not None else load_image(args.mask_file)
        if args.init_image is None:
            current_frame = img
        else:
            current_frame = load_image(args.init_image)
    if current_mask is not None and current_frame is not None:
        current_mask = current_mask.resize((args.width, args.height), Image.LANCZOS)
        current_frame = current_frame.resize((args.width, args.height), Image.LANCZOS)
        current_mask = ImageOps.grayscale(current_mask)

        if args.invert_mask:
            current_mask = ImageOps.invert(current_mask)

        img = Image.composite(img, current_frame, current_mask)

        if is_bgr_array:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # del (current_mask, current_frame)
        # gc.collect()

    return img


def compose_mask(root, args, mask_seq, val_masks, frame_image, inner_idx: int = 0):
    # Compose_mask recursively: go to inner brackets, then b-op it and go upstack

    # Step 1:
    # recursive parenthesis pass
    # regex is not powerful here

    seq = ""
    inner_seq = ""
    parentheses_counter = 0

    for c in mask_seq:
        if c == ')':
            parentheses_counter = parentheses_counter - 1
        if parentheses_counter > 0:
            inner_seq += c
        if c == '(':
            parentheses_counter = parentheses_counter + 1
        if parentheses_counter == 0:
            if len(inner_seq) > 0:
                inner_idx += 1
                seq += compose_mask(root, args, inner_seq, val_masks, frame_image, inner_idx)
                inner_seq = ""
            else:
                seq += c

    if parentheses_counter != 0:
        raise Exception('Mismatched parentheses in {mask_seq}!')

    mask_seq = seq

    # Step 2:
    # Load the word masks and file masks as vars

    # File masks
    pattern = r'\[(?P<inner>[\S\s]*?)\]'

    def parse(match_object):
        nonlocal inner_idx
        inner_idx += 1
        content = match_object.groupdict()['inner']
        val_masks[str(inner_idx)] = get_mask_from_file(content, args).convert('1')  # TODO: add caching
        return f"{{{inner_idx}}}"

    mask_seq = re.sub(pattern, parse, mask_seq)

    # Word masks
    pattern = r'<(?P<inner>[\S\s]*?)>'

    def parse(match_object):
        nonlocal inner_idx
        inner_idx += 1
        content = match_object.groupdict()['inner']
        val_masks[str(inner_idx)] = get_word_mask(root, frame_image, content).convert('1')
        return f"{{{inner_idx}}}"

    mask_seq = re.sub(pattern, parse, mask_seq)

    # Now that all inner parenthesis are eliminated we're left with a linear string

    # Step 3:
    # Boolean operations with masks
    # Operators: invert !, and &, or |, xor ^, difference \

    # Invert vars with '!'
    pattern = r'![\S\s]*{(?P<inner>[\S\s]*?)}'

    def parse(match_object):
        nonlocal inner_idx
        inner_idx += 1
        content = match_object.groupdict()['inner']
        savename = content
        if content in root.mask_preset_names:
            inner_idx += 1
            savename = str(inner_idx)
        val_masks[savename] = ImageChops.invert(val_masks[content])
        return f"{{{savename}}}"

    mask_seq = re.sub(pattern, parse, mask_seq)

    # Multiply neighbouring vars with '&'
    # Wait for replacements stall (like in Markov chains)
    while True:
        pattern = r'{(?P<inner1>[\S\s]*?)}[\s]*&[\s]*{(?P<inner2>[\S\s]*?)}'

        def parse(match_object):
            nonlocal inner_idx
            inner_idx += 1
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            savename = content
            if content in root.mask_preset_names:
                inner_idx += 1
                savename = str(inner_idx)
            val_masks[savename] = ImageChops.logical_and(val_masks[content], val_masks[content_second])
            return f"{{{savename}}}"

        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
        if mask_seq is prev_mask_seq:
            break

    # Add neighbouring vars with '|'
    while True:
        pattern = r'{(?P<inner1>[\S\s]*?)}[\s]*?\|[\s]*?{(?P<inner2>[\S\s]*?)}'

        def parse(match_object):
            nonlocal inner_idx
            inner_idx += 1
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            savename = content
            if content in root.mask_preset_names:
                inner_idx += 1
                savename = str(inner_idx)
            val_masks[savename] = ImageChops.logical_or(val_masks[content], val_masks[content_second])
            return f"{{{savename}}}"

        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
        if mask_seq is prev_mask_seq:
            break

    # Mutually exclude neighbouring vars with '^'
    while True:
        pattern = r'{(?P<inner1>[\S\s]*?)}[\s]*\^[\s]*{(?P<inner2>[\S\s]*?)}'

        def parse(match_object):
            nonlocal inner_idx
            inner_idx += 1
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            savename = content
            if content in root.mask_preset_names:
                inner_idx += 1
                savename = str(inner_idx)
            val_masks[savename] = ImageChops.logical_xor(val_masks[content], val_masks[content_second])
            return f"{{{savename}}}"

        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
        if mask_seq is prev_mask_seq:
            break

    # Set-difference the regions with '\'
    while True:
        pattern = r'{(?P<inner1>[\S\s]*?)}[\s]*\\[\s]*{(?P<inner2>[\S\s]*?)}'

        def parse(match_object):
            content = match_object.groupdict()['inner1']
            content_second = match_object.groupdict()['inner2']
            savename = content
            if content in root.mask_preset_names:
                nonlocal inner_idx
                inner_idx += 1
                savename = str(inner_idx)
            val_masks[savename] = ImageChops.logical_and(val_masks[content],
                                                         ImageChops.invert(val_masks[content_second]))
            return f"{{{savename}}}"

        prev_mask_seq = mask_seq
        mask_seq = re.sub(pattern, parse, mask_seq)
        if mask_seq is prev_mask_seq:
            break

    # Step 4:
    # Output
    # Now we should have a single var left to return. If not, raise an error message
    pattern = r'{(?P<inner>[\S\s]*?)}'
    matches = re.findall(pattern, mask_seq)

    if len(matches) != 1:
        raise Exception(f'Wrong composable mask expression format! Broken mask sequence: {mask_seq}')

    return f"{{{matches[0]}}}"


def compose_mask_with_check(root, args, mask_seq, val_masks, frame_image):
    for k, v in val_masks.items():
        val_masks[k] = blank_if_none(v, args.width, args.height, '1').convert('1')
    return check_mask_for_errors(
        val_masks[compose_mask(root, args, mask_seq, val_masks, frame_image, 0)[1:-1]].convert('L'))


def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path, time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path


import os
from multiprocessing import Process


def save_image_thread(image, path, cls):
    # Save the image directly


    # if cls.gen.color_match_sample is not None:
    #
    #     logger.info("Applying subtle color correction.")
    #     sample = cls.gen.color_match_sample
    #     original_image = image
    #     original_lab = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2LAB)
    #     correction = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)
    #
    #     corrected_lab = match_histograms(original_lab, correction, channel_axis=2)
    #     corrected_image = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB).astype("uint8")
    #
    #     corrected_pil_image = Image.fromarray(corrected_image)
    #     blended_image = blendLayers(corrected_pil_image, original_image, BlendType.LUMINOSITY,
    #                                 opacity=cls.gen.colorCorrectionFactor)
    #
    #     image = blended_image.convert('RGB')

    image.save(path, "PNG")

def save_image(image, image_type, filename, args, video_args, root, cls=None):
    if video_args.store_frames_in_ram:
        # Storing in RAM as an alternative
        root.frames_cache.append({
            'path': os.path.join(args.outdir, filename),
            'image': image,
            'image_type': image_type
        })
    else:
        # Construct the full path where the image will be saved
        full_path = os.path.join(args.outdir, filename)

        # Create a new thread for saving the image
        thread = Thread(target=save_image_thread, args=(image, full_path, cls))

        # Start the thread
        thread.start()
# def save_image_subprocess(image, path):
#     # Save the image at the specified path
#     image.save(path)
#
# def save_image(image, image_type, filename, args, video_args, root):
#     if video_args.store_frames_in_ram:
#         return
#         # You can uncomment the following lines if you need to cache the frames instead of saving them directly
#         # root.frames_cache.append(
#         #     {'path': os.path.join(args.outdir, filename), 'image': image, 'image_type': image_type})
#     else:
#         # Construct the full path where the image will be saved
#         full_path = os.path.join(args.outdir, filename)
#
#         # Create a new process for saving the image
#         process = Process(target=save_image_subprocess, args=(image, full_path))
#
#         # Start the process
#         process.start()


def reset_frames_cache(root):
    root.frames_cache = []
    gc.collect()


def dump_frames_cache(root):
    for image_cache in root.frames_cache:
        if image_cache['image_type'] == 'cv2':
            cv2.imwrite(image_cache['path'], image_cache['image'])
        elif image_cache['image_type'] == 'PIL':
            image_cache['image'].save(image_cache['path'])
    # do not reset the cache since we're going to add frame erasing later function #TODO


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
