import os
import cv2

from deforum.utils.constants import config


def get_resume_vars(resume_path, timestring, cadence):
    DEBUG_MODE = False
    frame_count = 0
    batchname = None
    output_path = os.path.join(config.root_path, "output", "deforum")
    # Find the correct folder based on timestring
    # for folder_name in os.listdir(output_path):
    #     if timestring in folder_name:
    #         batchname = folder_name
    #         break
    #
    # if batchname is None:
    #     print(f"No folder found with timestring {timestring}")
    #     return None  # or handle the error as appropriate

    # folder_path = os.path.join(output_path, batchname)

    # Image file extensions to be considered
    valid_extensions = {'.png', '.jpg', '.jpeg'}

    for item in os.listdir(resume_path):
        extension = os.path.splitext(item)[1].lower()

        if extension in valid_extensions:
            # Additional filename checks can be reintroduced here if necessary
            frame_count += 1
            if DEBUG_MODE:
                print(f"\033[36mResuming:\033[0m File: {item}")

    print(f"\033[36mResuming:\033[0m Current frame count: {frame_count}")

    last_frame = frame_count - (frame_count % cadence)
    prev_frame = last_frame - cadence
    next_frame = frame_count - 1

    prev_img_path = os.path.join(resume_path, f"{timestring}_{prev_frame:09}.png")
    next_img_path = os.path.join(resume_path, f"{timestring}_{next_frame:09}.png")
    prev_img = cv2.imread(prev_img_path)
    next_img = cv2.imread(next_img_path)

    print(f"\033[36mResuming:\033[0m Last frame: {prev_frame} - Next frame: {next_frame}")

    return batchname, prev_frame, next_frame, prev_img, next_img

