import importlib
import os

import requests
from tqdm import tqdm

def fetch_and_download_model(modelId: str, destination: str = ""):
    # Fetch model details
    response = requests.get(f"https://civitai.com/api/v1/models/{modelId}")
    response.raise_for_status()
    model_data = response.json()

    download_url = model_data['modelVersions'][0]['downloadUrl']
    filename = model_data['modelVersions'][0]['files'][0]['name']

    dir_path = destination

    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)

    # Check if file already exists
    if os.path.exists(filepath):
        print(f"File {filename} already exists in models/checkpoints/")
        return filename

    # Download file in chunks with progress bar
    print(f"Downloading {filename}...")
    response = requests.get(f'{download_url}?token=a44763d416db87cfb4fdb6b70369f4a3', stream=True, headers={'Content-Disposition': 'attachment'})
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filepath, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong while downloading the file.")
    else:
        print(f"{filename} downloaded successfully!")

    return filename


class DeforumBase:
    """
    Base class for the Deforum animation processing.

    Provides methods for initializing the Deforum animation pipeline using specific generator and pipeline
    configurations.
    """

    @classmethod
    def from_civitai(cls,
                     modelid: str = "125703",
                     generator_name: str = "ComfyDeforumGenerator",
                     cache_dir: str = "",
                     lcm: bool = False,
                     trt: bool = False) -> 'DeforumBase':
        """
        Class method to initialize a Deforum animation pipeline using specific configurations.

        Args:
            modelid (str, optional): Identifier for the model to fetch from CivitAi. Defaults to None.
            generator_name (str, optional): The generator class tp use. Defaults to "ComfyDeforumGenerator".
            cache_dir (str, optional): Directory for caching models. Defaults to default_cache_folder.
            lcm (bool, optional): Flag to determine if low-complexity mode should be activated. Defaults to False.
            trt (bool, optional): Flag to enable TrT

        Returns:
            DeforumBase: Initialized Deforum animation pipeline object.

        Raises: AssertionError: Raised if the specified generator or pipeline is not available or if cache directory
        issues occur.
        """

        # from deforum import available_engines
        # assert generator in available_engines, f"Make sure to use one of the available engines: {available_engines}"
        #
        # from deforum import available_pipelines assert pipeline in available_pipelines, f"Make sure to use one of
        # the available pipelines: {available_pipelines}" Ensure cache directory exists
        if cache_dir == '':
            cache_dir = 'models'

        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        assert os.path.isdir(
            cache_dir), "Could not create the requested cache dir, make sure the application has permissions"
        # Download model from CivitAi if specified
        if modelid is not None:
            # from ..utils.file_dl_util import download_file_to
            # from ..utils.civitai_model_dl_util import get_civitai_link_from_modelid
            filename = fetch_and_download_model(modelId=modelid, destination="models")
            model_path = os.path.join("models", filename)
            # model_params = get_civitai_link_from_modelid(modelId=modelid)
            # filename = model_params.filename
            # model_path = os.path.join(cache_dir, filename)
            # print(model_params.url, model_params.filename)
            # download_file_to(url=model_params.url, destination_dir=cache_dir, filename=model_params.filename)
        else:
            model_path = None
        assert model_path is not None, ("Instantiating from CivitAI is only possible with a valid modelID, and when "
                                        "CivitAI's services are healthy.")
        deforum_module = importlib.import_module(cls.__module__.split(".")[0])
        generator_attrs = {
            "model_path": model_path,
            "lcm": lcm,
            "trt": trt
        }
        pipeline_attrs = {}
        # Import the generator and pipeline
        generator_class = getattr(deforum_module, generator_name)
        pipeline_class = getattr(deforum_module, cls.__name__)
        # Create and return pipeline object
        return cls.factory(pipeline_class, pipeline_attrs, generator_class, generator_attrs)

    @classmethod
    def from_single_file(cls,
                         pretrained_model_repo_or_path: str = "",
                         generator_name: str = "ComfyDeforumGenerator",
                         ) -> 'DeforumBase':
        """

        Args:
            generator_name: (str) The selected generator. Defaults to (ComfyDeforumGenerator)
            pretrained_model_repo_or_path: (str) The desired model's path

        Returns:
           DeforumBase: Initialized Deforum animation pipeline object.

        Raises:
            AssertionError: Raised if the specified model cannot be found or loaded.

        """

        assert os.path.isfile(pretrained_model_repo_or_path), "The given model path must exist"
        deforum_module = importlib.import_module(cls.__module__.split(".")[0])
        generator_attrs = {
            "model_path": pretrained_model_repo_or_path,
            "lcm": False,
            "trt": False
        }
        pipeline_attrs = {}
        # Import the generator and pipeline
        generator_class = getattr(deforum_module, generator_name)
        pipeline_class = getattr(deforum_module, cls.__name__)
        # Create and return pipeline object
        return cls.factory(pipeline_class, pipeline_attrs, generator_class, generator_attrs)

    @classmethod
    def factory(cls, pipeline_class, pipeline_attrs, generator_class, generator_attrs):
        return pipeline_class(generator=generator_class(**generator_attrs), **pipeline_attrs)
