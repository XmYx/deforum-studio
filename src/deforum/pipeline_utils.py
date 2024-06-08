import itertools
import json
import os
import random
import secrets
import time
import uuid
from deforum.utils.logging_config import logger

from .pipelines.deforum_animation.animation_params import (DeforumArgs, DeforumAnimArgs,
                                                           ParseqArgs, LoopArgs, RootArgs,
                                                           DeforumOutputArgs, DeforumAnimPrompts, areas)
from .utils.constants import config


def next_seed(args, root):
    if args.seed_behavior == 'iter':
        args.seed += 1 if root.seed_internal % args.seed_iter_N == 0 else 0
        root.seed_internal += 1
    elif args.seed_behavior == 'ladder':
        args.seed += 2 if root.seed_internal == 0 else -1
        root.seed_internal = 1 if root.seed_internal == 0 else 0
    elif args.seed_behavior == 'alternate':
        args.seed += 1 if root.seed_internal == 0 else -1
        root.seed_internal = 1 if root.seed_internal == 0 else 0
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2 ** 32 - 1)
    return args.seed


# Add pairwise implementation here not to upgrade
# the whole python to 3.10 just for one function
def pairwise_repl(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def isJson(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True


def extract_values(args):
    return {key: value['value'] for key, value in args.items()}


def load_settings(settings_file_path: str) -> dict:
    """
    Load settings from a provided file.

    Args:
        settings_file_path (str): Path to the settings file.

    Returns:
        dict: Loaded settings data.

    Raises:
        ValueError: If the provided file type is unsupported.
    """
    if not os.path.isfile(settings_file_path):
        raise FileNotFoundError(f"Settings file not found: {settings_file_path}")

    file_ext = os.path.splitext(settings_file_path)[1]
    if file_ext == '.json':
        with open(settings_file_path, 'r') as f:
            data = json.load(f)
    elif file_ext == '.txt':
        with open(settings_file_path, 'r') as f:
            content = f.read()
            data = json.loads(content)
    else:
        raise ValueError("Unsupported file type")

    return data


class DeforumDataObject:
    """
    Class representing the data object for Deforum animations.

    This class contains all data output of a generation, and can be reused to store and return any data.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the generation object with default values and any provided arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        # Set all provided keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, attribute, default=None):
        """
        Retrieve the value of a specified attribute or a default value if not present.

        Args:
            attribute (str): Name of the attribute to retrieve.
            default (any, optional): Default value to return if attribute is not present.

        Returns:
            any: Value of the attribute or the default value.
        """
        return getattr(self, attribute, default)

    def to_dict(self) -> dict:
        """
        Convert all instance attributes to a dictionary.

        Returns:
            dict: Dictionary containing all instance attributes.
        """
        return self.__dict__.copy()

    def update_from_kwargs(self, **kwargs):
        """
        Update object attributes using provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_json_compatible_dict(self):
        """
        Convert all instance attributes to a JSON-compatible dictionary.

        Returns:
            dict: Dictionary containing all instance attributes that are JSON-compatible.
        """
        def is_jsonable(x):
            try:
                json.dumps(x)
                return True
            except (TypeError, OverflowError):
                return False

        def convert(obj):
            if isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif hasattr(obj, 'to_dict'):
                return convert(obj.to_dict())
            elif is_jsonable(obj):
                return obj
            else:
                # If it's not JSON serializable, skip it
                return None

        json_compatible_dict = {}
        for key, value in self.__dict__.items():
            try:
                # Convert the value if possible, otherwise it will be None
                converted_value = convert(value)
                if converted_value is not None:
                    json_compatible_dict[key] = converted_value
            except TypeError as e:
                # If there's a TypeError, it means conversion isn't possible; skip this attribute
                logger.error(f"Skipping attribute '{key}': {e}")

        return json_compatible_dict

    def save_as_json(self, file_path):
        """
        Save the JSON-compatible dictionary to a text file.

        Args:
            file_path (str): The path to the file where the JSON should be saved.
        """
        json_compatible_dict = self.to_json_compatible_dict()
        with open(file_path, 'w') as file:
            json.dump(json_compatible_dict, file, indent=4)

class DeforumGenerationObject(DeforumDataObject):
    """
    Class representing the generation object for Deforum animations.

    This class contains all the required attributes and methods for defining, managing, and manipulating the
    animation generation object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Initializes the generation object with default values and any provided arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        # Extract default values from various argument classes
        base_args = extract_values(DeforumArgs())
        anim_args = extract_values(DeforumAnimArgs())
        parseg_args = extract_values(ParseqArgs())
        loop_args = extract_values(LoopArgs())
        root = RootArgs()
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        merged_args = {**base_args, **anim_args, **parseg_args, **loop_args, **output_args_dict, **root}
        self.diffusion_cadence = 1
        self.seed = -1
        # Set all default values as attributes
        for key, value in merged_args.items():
            setattr(self, key, value)

        self.parseq_manifest = None
        animation_prompts = DeforumAnimPrompts()
        self.animation_prompts = json.loads(animation_prompts)
        self.resume_timestring = None
        self.resume_from_timestring = False

        self.timestring = kwargs.get('timestring', time.strftime('%Y%m%d%H%M%S'))
        self.batch_name = kwargs.get('batch_name', f"deforum_")
        full_base_folder_path = config.output_dir
        # outdir is computed here, but may be overridden by client code if 'outdir' is specified in the settings.
        # This ability to override is essential to ensure predictable output paths for the client.
        # The override happens in DeforumGenerationObject.from_settings_file().
        self.outdir = os.path.join(full_base_folder_path, f"{self.batch_name}_{self.timestring}")
        

        # Handle seed initialization
        if self.seed == -1 or self.seed == "-1":
            setattr(self, "seed", secrets.randbelow(999999999999999999))
            setattr(self, "raw_seed", int(self.seed))
            setattr(self, "seed_internal", 0)
        else:
            self.seed = int(self.seed)

        self.scheduler = "normal"
        self.sampler_name = "DPM++ 2M Karras"

        # Further attribute initializations
        self.prompts = None
        self.frame_interpolation_engine = None
        self.prev_img = None
        self.color_match_sample = None
        self.start_frame = 0
        self.frame_idx = 0
        self.last_diffused_frame = None
        self.next_frame_to_diffuse = None
        self.flow = None
        self.prev_flow = None
        self.image = None
        self.store_frames_in_ram = None
        self.turbo_prev_image, self.turbo_prev_frame_idx = None, 0
        self.turbo_next_image, self.turbo_next_frame_idx = None, 0
        self.contrast = 1.0
        self.hybrid_use_full_video = True
        self.hybrid_use_first_frame_as_init_image = False
        self.turbo_steps = self.diffusion_cadence
        self.img = None
        self.opencv_image = None
        self.use_areas = False
        self.areas = areas
        self.operation_id = uuid.uuid4().hex
        self.depth = None
        self.skip_hybrid_paths = False
        self.inputfiles = None
        self.amount = 0
        self.noise = 0.002
        self.skip_video_creation = False
        self.color_match_at = 'post'
        self.dry_run = False
        self.animation_prompts_positive = ""
        self.animation_prompts_negative = ""
        self.audio_path = ""
        self.enable_ad_pass = False
        self.adiff_pass_params = {}

        # Set all provided keyword arguments as attributes
        for key, value in kwargs.items():
            if key == 'animation_prompts':
                logger.info(f"SETTING GENERATION OBJECT: {key}, {value}")

            setattr(self, key, value)

    @classmethod
    def from_settings_file(cls, settings_file_path: str = None) -> 'DeforumGenerationObject':
        """
        Create an instance of the generation object using settings from a provided file.

        Args:
            settings_file_path (str, optional): Path to the settings file.

        Returns:
            DeforumGenerationObject: Initialized generation object instance.

        Raises:
            FileNotFoundError: If the settings file is not found.
            ValueError: If the provided file type is unsupported.
        """
        instance = cls()

        # Load data from provided file
        if settings_file_path:
            logger.info(f"settings_file_path: {settings_file_path}")
            data = load_settings(settings_file_path)

            # Update instance attributes using loaded data
            for key, value in data.items():
                setattr(instance, key, value)

        # Additional attribute updates based on loaded data
        if hasattr(instance, "diffusion_cadence"):
            instance.turbo_steps = int(instance.diffusion_cadence)
        if hasattr(instance, "using_video_init") and instance.using_video_init:
            instance.turbo_steps = 1
        if instance.prompts is not None:
            instance.animation_prompts = instance.prompts

        return instance

class DeforumKeyFrame(DeforumDataObject):
    """
    Class representing the key frame for Deforum animations.

    This class contains attributes that define a specific frame's characteristics in the Deforum animation process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Initializes the Keyframe object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

    @classmethod
    def from_keys(cls, keys, frame_idx) -> 'DeforumKeyFrame':
        """
        Create an instance of the key frame object using settings from provided keys and frame index.

        Args:
            keys: Object containing animation schedule series attributes.
            frame_idx (int): Index of the frame to retrieve settings for.

        Returns:
            DeforumKeyFrame: Initialized key frame object instance.
        """
        instance = cls()
        instance.noise = keys.noise_schedule_series[frame_idx]
        instance.strength = keys.strength_schedule_series[frame_idx]
        instance.scale = keys.cfg_scale_schedule_series[frame_idx]
        instance.contrast = keys.contrast_schedule_series[frame_idx]
        instance.kernel = int(keys.kernel_schedule_series[frame_idx])
        instance.sigma = keys.sigma_schedule_series[frame_idx]
        instance.amount = keys.amount_schedule_series[frame_idx]
        instance.threshold = keys.threshold_schedule_series[frame_idx]
        instance.cadence_flow_factor = keys.cadence_flow_factor_schedule_series[frame_idx]
        instance.redo_flow_factor = keys.redo_flow_factor_schedule_series[frame_idx]
        instance.hybrid_comp_schedules = {
            "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
            "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
            "mask_auto_contrast_cutoff_low": int(
                keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
            "mask_auto_contrast_cutoff_high": int(
                keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]),
            "flow_factor": keys.hybrid_flow_factor_schedule_series[frame_idx]
        }
        instance.scheduled_sampler_name = None
        instance.scheduled_clipskip = None
        instance.scheduled_noise_multiplier = None
        instance.scheduled_ddim_eta = None
        instance.scheduled_ancestral_eta = None
        return instance
