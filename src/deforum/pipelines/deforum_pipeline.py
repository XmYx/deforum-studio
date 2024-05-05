import importlib
import os
from glob import glob

from deforum.utils.model_download import (
    get_filename_from_url,
    download_file,
    download_from_civitai,
)

from deforum.utils.logging_config import logger
from deforum.utils.constants import config

class DeforumBase:
    """
    Base class for Deforum inference.
    """

    @classmethod
    def from_file(
        cls,
        model_path: str = "",
        generator_name: str = "ComfyDeforumGenerator",
        lcm: bool = False,
        trt: bool = False,
    ) -> "DeforumBase":
        """
        Class method to initialize a Deforum pipeline from pretrained model path.
        """

        # assert model_path is not None, "Model path should be a string."
        assert isinstance(model_path, str), "Model path should be a string."

        # assert pretrained_model_name_or_path is a file
        assert os.path.isfile(
            model_path
        ), "Instantiating from pretrained model requires a valid file path."

        # import the generator and pipeline
        deforum_module = importlib.import_module(cls.__module__.split(".")[0])
        generator_attrs = {
            "model_path": model_path,
            "lcm": lcm,
            "trt": trt,
        }
        pipeline_attrs = {}
        generator_class = getattr(deforum_module, generator_name)
        pipeline_class = getattr(deforum_module, cls.__name__)

        # create and return pipeline object
        return cls.factory(
            pipeline_class, pipeline_attrs, generator_class, generator_attrs
        )

    @classmethod
    def from_url(
        cls,
        url: str = "",
        cache_dir: str = config.model_dir,
        force_download: bool = False,
        generator_name: str = "ComfyDeforumGenerator",
        lcm: bool = False,
        trt: bool = False,
    ) -> "DeforumBase":
        """
        Class method to initialize a Deforum pipeline from download url.
        """

        # assert url is not None
        assert url is not None, "URL should be a string."

        # get filename
        filename = get_filename_from_url(url)

        # try download model from url
        try:
            filename = download_file(
                download_url=url,
                destination=cache_dir,
                filename=filename,
                force_download=force_download,
            )
            model_path = os.path.join(cache_dir, filename)
        except:
            model_path = None

        # assert model_path is not None
        assert model_path is not None, "Instantiating from URL requires a valid URL."

        # import the generator and pipeline
        deforum_module = importlib.import_module(cls.__module__.split(".")[0])
        generator_attrs = {"model_path": model_path, "lcm": lcm, "trt": trt}
        pipeline_attrs = {}
        generator_class = getattr(deforum_module, generator_name)
        pipeline_class = getattr(deforum_module, cls.__name__)

        # Create and return pipeline object
        return cls.factory(
            pipeline_class, pipeline_attrs, generator_class, generator_attrs
        )

    @classmethod
    def from_civitai(
        cls,
        model_id: str = "",
        cache_dir: str = config.model_dir,
        force_download: bool = False,
        generator_name: str = "ComfyDeforumGenerator",
        lcm: bool = False,
        trt: bool = False,
    ) -> "DeforumBase":
        """
        Class method to initialize a Deforum pipeline from civitai model id.
        """

        # assert modelid is not None, "Model ID should be a string."
        assert isinstance(model_id, str), "Model ID should be a string."
        # Find the largest 'safetensors' file in cache_dir
        safetensor_files = glob(os.path.join(cache_dir, '*safetensors*'))
        if safetensor_files:
            largest_file = max(safetensor_files, key=os.path.getsize)
            filename = os.path.basename(largest_file)
            logger.info(f"Using largest safetensor file found: {filename}")
        else:
            filename = None
        # try download model from civitai
        try:
            filename = download_from_civitai(
                model_id=model_id, destination=cache_dir, force_download=force_download
            )
            model_path = os.path.join(cache_dir, filename)
        except:
            filepath = os.path.join(cache_dir, filename)
            if os.path.exists(filepath):
                logger.warning("Couldn't download model from CivitAI, using cached copy.")
                model_path = filepath
            else:
                model_path = None

        # assert model_path is not None
        assert model_path is not None, (
            "Instantiating from CivitAI is only possible with a valid modelID, and when "
            "CivitAI's services are healthy."
        )

        # import the generator and pipeline
        deforum_module = importlib.import_module(cls.__module__.split(".")[0])
        generator_attrs = {"model_path": model_path, "lcm": lcm, "trt": trt}
        pipeline_attrs = {}
        generator_class = getattr(deforum_module, generator_name)
        pipeline_class = getattr(deforum_module, cls.__name__)

        # create and return pipeline object
        return cls.factory(
            pipeline_class, pipeline_attrs, generator_class, generator_attrs
        )

    @classmethod
    def factory(cls, pipeline_class, pipeline_attrs, generator_class, generator_attrs):
        return pipeline_class(
            generator=generator_class(**generator_attrs), **pipeline_attrs
        )
