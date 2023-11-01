import os
import pathlib

import PIL
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps, ImageEnhance

from ..utils.deforum_human_masking import video2humanmasks
from ..utils.image_utils import (get_resized_image_from_filename,
                                 autocontrast_grayscale,
                                 load_image_with_mask)
from ..utils.video_frame_utils import (vid2frames,
                                       get_quick_vid_info,
                                       get_frame_name)


def delete_all_imgs_in_folder(folder_path):
    files = list(pathlib.Path(folder_path).glob('*.jpg'))
    files.extend(list(pathlib.Path(folder_path).glob('*.png')))
    for f in files: os.remove(f)


def hybrid_generation(args, anim_args, root):
    video_in_frame_path = os.path.join(args.outdir, 'inputframes')
    hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')
    human_masks_path = os.path.join(args.outdir, 'human_masks')

    # create hybridframes folder whether using init_image or inputframes
    os.makedirs(hybrid_frame_path, exist_ok=True)

    if anim_args.hybrid_generate_inputframes:
        # create folders for the video input frames and optional hybrid frames to live in
        os.makedirs(video_in_frame_path, exist_ok=True)

        # delete frames if overwrite = true
        if anim_args.overwrite_extracted_frames:
            delete_all_imgs_in_folder(hybrid_frame_path)

        # save the video frames from input video
        print(f"Video to extract: {anim_args.video_init_path}")
        print(f"Extracting video (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}...")
        video_fps = vid2frames(video_path=anim_args.video_init_path, video_in_frame_path=video_in_frame_path,
                               n=anim_args.extract_nth_frame, overwrite=anim_args.overwrite_extracted_frames,
                               extract_from_frame=anim_args.extract_from_frame,
                               extract_to_frame=anim_args.extract_to_frame)

    # extract alpha masks of humans from the extracted input video imgs
    if anim_args.hybrid_generate_human_masks != "None":
        # create a folder for the human masks imgs to live in
        print(f"Checking /creating a folder for the human masks")
        os.makedirs(human_masks_path, exist_ok=True)

        # delete frames if overwrite = true
        if anim_args.overwrite_extracted_frames:
            delete_all_imgs_in_folder(human_masks_path)

        # in case that generate_input_frames isn't selected, we won't get the video fps rate as vid2frames isn't called, So we'll check the video fps in here instead
        if not anim_args.hybrid_generate_inputframes:
            _, video_fps, _ = get_quick_vid_info(anim_args.video_init_path)

        # calculate the correct fps of the masked video according to the original video fps and 'extract_nth_frame'
        output_fps = video_fps / anim_args.extract_nth_frame

        # generate the actual alpha masks from the input imgs
        print(f"Extracting alpha humans masks from the input frames")
        video2humanmasks(video_in_frame_path, human_masks_path, anim_args.hybrid_generate_human_masks, output_fps)

    # get sorted list of inputfiles
    inputfiles = sorted(pathlib.Path(video_in_frame_path).glob('*.jpg'))

    if not anim_args.hybrid_use_init_image:
        # determine max frames from length of input frames
        if args.hybrid_use_full_video:
            anim_args.max_frames = len(inputfiles)
        if anim_args.max_frames > len(inputfiles):
            anim_args.max_frames = len(inputfiles)
        if anim_args.max_frames < 1:
            raise Exception(
                f"Error: No input frames found in {video_in_frame_path}! Please check your input video path and whether you've opted to extract input frames.")
        print(f"Using {anim_args.max_frames} input frames from {video_in_frame_path}...")

    # use first frame as init
    if anim_args.hybrid_use_first_frame_as_init_image:
        for f in inputfiles:
            args.init_image = str(f)
            args.init_image_box = None  # init_image_box not used in this case
            args.use_init = True
            print(f"Using init_image from video: {args.init_image}")
            break

    return args, anim_args, inputfiles


def hybrid_composite(args, anim_args, frame_idx, prev_img, depth_model, hybrid_comp_schedules, root):
    video_frame = os.path.join(args.outdir, 'inputframes',
                               get_frame_name(anim_args.video_init_path) + f"{frame_idx:09}.jpg")
    video_depth_frame = os.path.join(args.outdir, 'hybridframes',
                                     get_frame_name(anim_args.video_init_path) + f"_vid_depth{frame_idx:09}.jpg")
    depth_frame = os.path.join(args.outdir, f"{root.timestring}_depth_{frame_idx - 1:09}.png")
    mask_frame = os.path.join(args.outdir, 'hybridframes',
                              get_frame_name(anim_args.video_init_path) + f"_mask{frame_idx:09}.jpg")
    comp_frame = os.path.join(args.outdir, 'hybridframes',
                              get_frame_name(anim_args.video_init_path) + f"_comp{frame_idx:09}.jpg")
    prev_frame = os.path.join(args.outdir, 'hybridframes',
                              get_frame_name(anim_args.video_init_path) + f"_prev{frame_idx:09}.jpg")
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    prev_img_hybrid = Image.fromarray(prev_img)
    if anim_args.hybrid_use_init_image:
        video_image = load_image_with_mask(args.init_image, args.init_image_box)
    else:
        video_image = Image.open(video_frame)
    video_image = video_image.resize((args.W, args.H), PIL.Image.LANCZOS)
    hybrid_mask = None

    # composite mask types
    if anim_args.hybrid_comp_mask_type == 'Depth':  # get depth from last generation
        hybrid_mask = Image.open(depth_frame)
    elif anim_args.hybrid_comp_mask_type == 'Video Depth':  # get video depth
        video_depth = depth_model.predict(np.array(video_image), anim_args.midas_weight, root.half_precision)
        depth_model.save(video_depth_frame, video_depth)
        hybrid_mask = Image.open(video_depth_frame)
    elif anim_args.hybrid_comp_mask_type == 'Blend':  # create blend mask image
        hybrid_mask = Image.blend(ImageOps.grayscale(prev_img_hybrid), ImageOps.grayscale(video_image),
                                  hybrid_comp_schedules['mask_blend_alpha'])
    elif anim_args.hybrid_comp_mask_type == 'Difference':  # create difference mask image
        hybrid_mask = ImageChops.difference(ImageOps.grayscale(prev_img_hybrid), ImageOps.grayscale(video_image))

    # optionally invert mask, if mask type is defined
    if anim_args.hybrid_comp_mask_inverse and anim_args.hybrid_comp_mask_type != "None":
        hybrid_mask = ImageOps.invert(hybrid_mask)

    # if a mask type is selected, make composition
    if hybrid_mask is None:
        hybrid_comp = video_image
    else:
        # ensure grayscale
        hybrid_mask = ImageOps.grayscale(hybrid_mask)
        # equalization before
        if anim_args.hybrid_comp_mask_equalize in ['Before', 'Both']:
            hybrid_mask = ImageOps.equalize(hybrid_mask)
            # contrast
        hybrid_mask = ImageEnhance.Contrast(hybrid_mask).enhance(hybrid_comp_schedules['mask_contrast'])
        # auto contrast with cutoffs lo/hi
        if anim_args.hybrid_comp_mask_auto_contrast:
            hybrid_mask = autocontrast_grayscale(np.array(hybrid_mask),
                                                 hybrid_comp_schedules['mask_auto_contrast_cutoff_low'],
                                                 hybrid_comp_schedules['mask_auto_contrast_cutoff_high'])
            hybrid_mask = Image.fromarray(hybrid_mask)
            hybrid_mask = ImageOps.grayscale(hybrid_mask)
        if anim_args.hybrid_comp_save_extra_frames:
            hybrid_mask.save(mask_frame)
            # equalization after
        if anim_args.hybrid_comp_mask_equalize in ['After', 'Both']:
            hybrid_mask = ImageOps.equalize(hybrid_mask)
            # do compositing and save
        hybrid_comp = Image.composite(prev_img_hybrid, video_image, hybrid_mask)
        if anim_args.hybrid_comp_save_extra_frames:
            hybrid_comp.save(comp_frame)

    # final blend of composite with prev_img, or just a blend if no composite is selected
    hybrid_blend = Image.blend(prev_img_hybrid, hybrid_comp, hybrid_comp_schedules['alpha'])
    if anim_args.hybrid_comp_save_extra_frames:
        hybrid_blend.save(prev_frame)

    prev_img = cv2.cvtColor(np.array(hybrid_blend), cv2.COLOR_RGB2BGR)

    # restore to np array and return
    return args, prev_img


def get_matrix_for_hybrid_motion(frame_idx, dimensions, inputfiles, hybrid_motion):
    print(f"Calculating {hybrid_motion} RANSAC matrix for frames {frame_idx} to {frame_idx + 1}")
    img1 = cv2.cvtColor(get_resized_image_from_filename(str(inputfiles[frame_idx]), dimensions), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(get_resized_image_from_filename(str(inputfiles[frame_idx + 1]), dimensions), cv2.COLOR_BGR2GRAY)
    M = get_transformation_matrix_from_images(img1, img2, hybrid_motion)
    return M


def get_matrix_for_hybrid_motion_prev(frame_idx, dimensions, inputfiles, prev_img, hybrid_motion):
    print(f"Calculating {hybrid_motion} RANSAC matrix for frames {frame_idx} to {frame_idx + 1}")
    # first handle invalid images by returning default matrix
    height, width = prev_img.shape[:2]
    if height == 0 or width == 0 or prev_img != np.uint8:
        return get_hybrid_motion_default_matrix(hybrid_motion)
    else:
        prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(get_resized_image_from_filename(str(inputfiles[frame_idx + 1]), dimensions),
                           cv2.COLOR_BGR2GRAY)
        M = get_transformation_matrix_from_images(prev_img_gray, img, hybrid_motion)
        return M


def get_matrix_for_hybrid_motion_from_images(input_image, prev_img, hybrid_motion):
    # print(f"Calculating {hybrid_motion} RANSAC matrix for frames {frame_idx} to {frame_idx + 1}")
    # first handle invalid images by returning default matrix
    height, width = prev_img.shape[:2]
    if height == 0 or width == 0 or prev_img != np.uint8:
        return get_hybrid_motion_default_matrix(hybrid_motion)
    else:
        prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        M = get_transformation_matrix_from_images(prev_img_gray, img, hybrid_motion)
        return M


def get_hybrid_motion_default_matrix(hybrid_motion):
    if hybrid_motion == "Perspective":
        arr = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    else:
        arr = np.array([[1., 0., 0.], [0., 1., 0.]])
    return arr


def get_transformation_matrix_from_images(img1, img2, hybrid_motion, confidence=0.75):
    # Create SIFT detector and feature extractor
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Create BFMatcher object and match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < confidence * n.distance:
            good_matches.append(m)

    if len(good_matches) <= 8:
        get_hybrid_motion_default_matrix(hybrid_motion)

    # Convert keypoints to numpy arrays
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if len(src_pts) <= 8 or len(dst_pts) <= 8:
        return get_hybrid_motion_default_matrix(hybrid_motion)
    elif hybrid_motion == "Perspective":  # Perspective transformation (3x3)
        transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return transformation_matrix
    else:  # Affine - rigid transformation (no skew 3x2)
        transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        return transformation_rigid_matrix
