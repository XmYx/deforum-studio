import os
from unittest.mock import Mock

import pytest

from deforum import DeforumAnimationPipeline
from deforum.commands.deforum_e2e_test_helpers import get_video_properties
from deforum.utils.constants import config


##
# Uses dry run to confirm we generate diffusion and interpolation frames
# at the correct positions, for various values of the diffusion cadence.
##
class TestCadence:

    @pytest.fixture(autouse=True)
    def inject_request(self, request):
        self.test_name = request.node.name
        self.interpolated_frames = []
        self.diffused_frames = []


    def build_options_to_test(self, cadence):
        outdir = os.path.join(config.output_dir, "tests", "e2etests", self.test_name)
        return {
            # config item under test:
            "diffusion_cadence": cadence,

            # to ensure transforms are visible (5 degree rotation per frame)
            "angle":"(0): 5",

            # lock in for subsequent validation
            "outdir": outdir,
            "batch_name": self.test_name,
            "max_frames": 10,
            "fps": 2,

            # to speed up the test run:
            "dry_run": True,
            "animation_mode": "2D",
            "use_depth_warping": False,
            "color_coherence": "None",
        }


    def run_deforum_with(self, options):
        deforum = DeforumAnimationPipeline(Mock())
        return deforum(None, **options, callback=self.frame_tracking_callback)


    def frame_tracking_callback(self, data):
        frame = data.get("frame_idx")
        if (data.get("is_interpolated")):
            self.interpolated_frames.append(frame)
        else:
            self.diffused_frames.append(frame)


    @pytest.mark.parametrize("cadence, expected_diffused, expected_interpolated", [
        (1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], []),
        (2, [0, 2, 4, 6, 8], [1, 3, 5, 7, 9]),
        (3, [0, 3, 6, 9], [1, 2, 4, 5, 7, 8]),
        (4, [0, 4, 8], [1, 2, 3, 5, 6, 7, 9]),
        (5, [0, 5], [1, 2, 3, 4, 6, 7, 8, 9]),
        (6, [0, 6], [1, 2, 3, 4, 5, 7, 8, 9]),
        (7, [0, 7], [1, 2, 3, 4, 5, 6, 8, 9]),
        (8, [0, 8], [1, 2, 3, 4, 5, 6, 7, 9]),
        (9, [0, 9], [1, 2, 3, 4, 5, 6, 7, 8]),
        (10, [0], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (11, [0], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    ])
    def test_cadence(self, cadence, expected_diffused, expected_interpolated):
        # set up
        options = self.build_options_to_test(cadence=cadence)

        # act
        result = self.run_deforum_with(options)

        # assert
        total_frames, fps = get_video_properties(result.video_path)
        assert self.diffused_frames == expected_diffused
        assert self.interpolated_frames == expected_interpolated
        assert total_frames == options.get("max_frames")
        assert fps == options.get("fps")

