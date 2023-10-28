import numpy as np
from PIL import Image

from ..models.FILM.inference import FilmModel


class FILMInterpolator:

    def __init__(self):
        self.film = FilmModel()

    def __call__(self,
                 images,
                 interp_frames,
                 *args,
                 **kwargs):
        images = [np.array(image) for image in images]

        interpolated = [images[0]]
        for i in range(len(images) - 1):  # We subtract 1 to avoid out-of-index errors
            image1 = images[i]
            image2 = images[i + 1]

            # Assuming self.film returns a list of interpolated frames
            interpolated_frames = self.film.inference(image1, image2, interp_frames)
            # interpolated_frames.pop(0)
            interpolated_frames.pop(-1)
            # Append the interpolated frames to the interpolated list
            interpolated.extend(interpolated_frames)
        interpolated.append(images[-1])
        interpolated = [Image.fromarray(image) for image in interpolated]

        return interpolated
