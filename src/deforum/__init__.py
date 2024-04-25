__version__ = "0.01.8.dev1"

import os, sys
from deforum.utils.logging_config import logger

rel_path = os.path.join(os.path.dirname(__file__), "models", "depth_models")

sys.path.extend([rel_path])

logger.info(f"Extended path with: {rel_path}; full path: {sys.path}")

from .commands import (start_deforum_cli)
from .generators import (ComfyDeforumGenerator,
                        DeforumDiffusersGenerator,
                         FILMInterpolator,
                         ImageRNGNoise)

from .models import (FilmModel)

from .pipelines import (DeforumBase,
                        DeforumAnimationPipeline,
                        DeforumAnimateDiffPipeline,
                        DeforumAnimateDifforumPipeline)