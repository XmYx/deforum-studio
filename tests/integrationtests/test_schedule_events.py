
import os
from unittest.mock import Mock

import pytest

from deforum import DeforumAnimationPipeline
from deforum.commands.deforum_e2e_test_helpers import get_video_properties
from deforum.utils.constants import config


##
# Uses dry run to confirm we generate schedules with variables injected as expected
##
class TestScheduleVariables:

    @pytest.fixture(autouse=True)
    def inject_request(self, request):
        self.test_name = request.node.name
        self.interpolated_frames = []
        self.diffused_frames = []


    def build_options_to_test(self):
        outdir = os.path.join(config.output_dir, "tests", "e2etests", self.test_name)
        return {
            # config item under test:
            "diffusion_cadence": 2,

            # lock in for subsequent validation
            "outdir": outdir,
            "batch_name": self.test_name,
            "max_frames": 5,
            "fps": 2,

            # to speed up the test run:
            "dry_run": True,
            "animation_mode": "2D",
            "use_depth_warping": False,
            "color_coherence": "None",
        }


    
    def run_deforum_with(self, options):        
        deforum = DeforumAnimationPipeline(Mock())
        return deforum(None, **options)
    

    def test_frame_number(self):
        # set up
        options = self.build_options_to_test()
        options["angle"] = "(0): t"

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.to_list() == [0, 1, 2, 3, 4]


    def test_bpm(self):
        # set up
        options = self.build_options_to_test()
        options["angle"] = "(0): bpm"

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.to_list() == [120, 120, 120, 120, 120]


    def test_fps(self):
        # set up
        options = self.build_options_to_test()
        options["angle"] = "(0): fps"

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.to_list() == [2, 2, 2, 2, 2]


    def test_max_frames(self):
        # set up
        options = self.build_options_to_test()
        options["angle"] = "(0): max_f"

        # act
        result = self.run_deforum_with(options)

        # assert - max_f is deliberately last frame idx rather than number of frames
        assert result.keys.angle_series.to_list() == [4, 4, 4, 4, 4]


    def test_beat(self):
        # set up
        options = self.build_options_to_test()
        options["bpm"] = 120
        options["fps"] = 10   
        options["angle"] = "(0): beat"

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.round(2).to_list() == [0, 0.2, 0.4, 0.6, 0.8]


    def test_beat_with_offset(self):
        # set up
        options = self.build_options_to_test()
        options["bpm"] = 120
        options["fps"] = 10
        options["beat_offset"] = 0.1
        options["angle"] = "(0): beat"

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.round(2).to_list() == [-0.1, 0.1, 0.3, 0.5, 0.7]


    def test_frames_to_beat(self):
        # set up
        options = self.build_options_to_test()
        options["bpm"] = 120
        options["fps"] = 10
        options["beat_offset"] = 0.1
        options["angle"] = "(0): beat"

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.round(2).to_list() == [-0.1, 0.1, 0.3, 0.5, 0.7]


    def test_frames_until_next_event(self):
        # set up
        options = self.build_options_to_test()
        options["bpm"] = 120
        options["fps"] = 10
        options["max_frames"] = 10
        options["angle"] = "(0): frames_until_next_event"
        options["schedule_events"] = [
            {
                "time": 0.2,

            },
            {
                "time": 0.55,
            },
            {
                "time": 0.8,
            },            
        ]

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.round(2).to_list() == [2, 1, 0, 2, 1, 0, 2, 1, 0, 0]


    def test_frames_since_prev_event(self):
        # set up
        options = self.build_options_to_test()
        options["bpm"] = 120
        options["fps"] = 10
        options["max_frames"] = 10
        options["angle"] = "(0): frames_since_prev_event"
        options["schedule_events"] = [
            {
                "time": 0.2,

            },
            {
                "time": 0.55,
            },
            {
                "time": 0.8,
            },            
        ]

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.round(2).to_list() == [0, 1, 2, 1, 2, 3, 1, 2, 3, 1]


    def test_events_conditional(self):
        # set up
        options = self.build_options_to_test()
        options["bpm"] = 120
        options["fps"] = 10
        options["angle"] = "(0): where((sec>0.2) & (frames_until_next_event<1), 1, -1)"
        options["schedule_events"] = [
            {
                "time": 0.2,

            },
            {
                "time": 0.3,
            },
        ]

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.angle_series.round(2).to_list() == [-1.0, -1.0, -1.0, 1.0, 1.0]



    def test_strength_conditional(self):
        # set up
        options = self.build_options_to_test()
        options["bpm"] = 120
        options["fps"] = 10
        options["strength_schedule"] = "(0): where(frames_until_next_event<1, 0.2, 0.8)",
        options["schedule_events"] = [
            {
                "time": 0.2,
            }
        ]

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.strength_schedule_series.round(2).to_list() == [0.8, 0.8, 0.2, 0.8, 0.2]

    def test_seed_reference(self):
        # set up
        options = self.build_options_to_test()
        options["bpm"] = 120
        options["fps"] = 10
        options["max_frames"] = 10
        options["seed"] = -1
        options["seed_behaviour"] = "schedule"
        options["seed_schedule"] = "(0): unique+events_passed",
        options["schedule_events"] = [
            {
                "time": 0.2,

            },
            {
                "time": 0.5,
            },
        ]

        # act
        result = self.run_deforum_with(options)

        # assert
        assert result.keys.seed_schedule_series[9]-result.keys.seed_schedule_series[3] == 1
        assert result.keys.seed_schedule_series[3]-result.keys.seed_schedule_series[0] == 1



 