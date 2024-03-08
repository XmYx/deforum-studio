__version__ = "0.01.8.dev1"

import os, sys

rel_path = os.path.join(os.path.dirname(__file__), "models", "depth_models")

print("EXTENDING PATH WITH", rel_path)

sys.path.extend([rel_path])

from .commands import (start_deforum_cli)
from .generators import (ComfyDeforumGenerator,
                         FILMInterpolator,
                         ImageRNGNoise)

from .models import (FilmModel)

from .pipelines import (DeforumBase,
                        DeforumAnimationPipeline,
                        DeforumAnimateDiffPipeline,
                        DeforumAnimateDifforumPipeline)