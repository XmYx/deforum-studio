from unittest.mock import Mock, patch

import pytest

from deforum.pipelines.deforum_animation.animation_helpers import generate_interpolated_frames
from deforum.pipelines.deforum_animation.pipeline_deforum_animation import DeforumAnimationPipeline


class TestPipelineDeforumAnimation:

    @classmethod
    def build_minimal_mock_generator(cls):
        gen = Mock()
        gen.diffusion_redo = 0
        gen.max_frames = 10
        gen.frame_interpolation_engine = "None"
        gen.use_init = False
        gen.resume_from_timestring = False
        return gen

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.deforum = DeforumAnimationPipeline(Mock())
        self.deforum.gen = TestPipelineDeforumAnimation.build_minimal_mock_generator()

    def use_cadence(self, cadence:int) -> None:
        # diffusion_cadence is obained from gen.get, so we need to provide it via a mock get method:
        self.deforum.gen.get.side_effect = lambda arg, default=None: cadence if arg == 'diffusion_cadence' else default


    @patch('deforum.pipelines.deforum_animation.pipeline_deforum_animation.os')  # mock out file system operations
    def test_cadence_post_function_correctly_added(self, _):
        # Setup
        self.use_cadence(2)

        # Act
        self.deforum.setup()

        # Assert
        assert generate_interpolated_frames in self.deforum.shoot_fns


    @patch('deforum.pipelines.deforum_animation.pipeline_deforum_animation.os')  # mock out file system operations
    def test_cadence_post_function_correctly_omitted(self, _):
        # Setup
        self.use_cadence(1)

        # Act
        self.deforum.setup()

        # Assert
        assert generate_interpolated_frames not in self.deforum.shoot_fns


    @patch('deforum.pipelines.deforum_animation.pipeline_deforum_animation.os')  # mock out file system operations
    def test_cadence_post_function_not_affectect_by_optical_flow_cadence(self, _):
        # Setup
        self.use_cadence(2)
        self.deforum.gen.optical_flow_cadence = "None"

        # Act
        self.deforum.setup()

        # Assert
        assert generate_interpolated_frames in self.deforum.shoot_fns
