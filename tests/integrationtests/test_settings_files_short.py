import os
import shutil

import pytest

from deforum import DeforumAnimationPipeline
from deforum.commands.deforum_e2e_test_helpers import VideoComparisonResult, get_frames, get_video_properties
from deforum.pipeline_utils import load_settings
from deforum.utils.constants import config
from deforum.utils.logging_config import logger


# TODO: this should be whittled down to a smaller number of salient settings tests.
settings_files = [
    'Classic.txt',
    'Classic-30s.txt',
    'Classic-3D-Motion-2-30s.txt',
    'Classic-3D-Motion-2.txt',
    'Classic-3D-Motion-3-30s.txt',
    'Classic-3D-Motion-3.txt',
    'Classic-3D-Motion-30s.txt',
    'Classic-3D-Motion-4-30s.txt',
    'Classic-3D-Motion-4.txt',
    'Classic-3D-Motion.txt',
    'Classic-Stop-Motion-2-30s.txt',
    'Classic-Stop-Motion-2.txt',
    'Classic-Stop-Motion-30s.txt',
    'Classic-Stop-Motion.txt',
    'Classic-Zoom-In-30s.txt',
    'Classic-Zoom-In.txt',
    'Classic-Zoom-Out-30s.txt',
    'Classic-Zoom-Out.txt',
    'Dolly-Zoom-Out-30s.txt',
    'Dolly-Zoom-Out.txt',
    'Elastic-Vision-30s.txt',
    'Elastic-Vision.txt',
    'Evolve-Fast-30s.txt',
    'Evolve-Fast.txt',
    'Evolve-Pulse-2-30s.txt',
    'Evolve-Pulse-2.txt',
    'Evolve-Pulse-30s.txt',
    'Evolve-Pulse-Glitch-30s.txt',
    'Evolve-Pulse-Glitch.txt',
    'Evolve-Pulse.txt',
    'Evolve-Slow-2-30s.txt',
    'Evolve-Slow-2.txt',
    'Evolve-Slow-30s.txt',
    'Evolve-Slow.txt',
    'Evolve-Zoom-Slow-30s.txt',
    'Evolve-Zoom-Slow.txt',
    'Flashbacks-30s.txt',
    'Flashbacks.txt',
    'Fly-Through-2-30s.txt',
    'Fly-Through-2.txt',
    'Fly-Through-30s.txt',
    'Fly-Through-Spin-30s.txt',
    'Fly-Through-Spin.txt',
    'Fly-Through.txt',
    'Grids-30s.txt',
    'Grids.txt',
    'Look-Around-30s.txt',
    'Look-Around.txt',
    'Move-Around-30s.txt',
    'Move-Around.txt',
    'Move-Down-30s.txt',
    'Move-Down.txt',
    'Move-Float-30s.txt',
    'Move-Float.txt',
    'Move-In-out-30s.txt',
    'Move-In-out.txt',
    'Move-Left-30s.txt',
    'Move-Left.txt',
    'Move-Right-30s.txt',
    'Move-Right.txt',
    'Move-Up-30s.txt',
    'Move-Up.txt',
    'Move-Warp-2-30s.txt',
    'Move-Warp-2.txt',
    'Move-Warp-30s.txt',
    'Move-Warp.txt',
    'Quick-Change-30s.txt',
    'Quick-Change.txt',
    'Revolve-30s.txt',
    'Revolve.txt',
    'Scene-Change-2-30s.txt',
    'Scene-Change-2.txt',
    'Scene-Change-30s.txt',
    'Scene-Change.txt',
    'Shapes-Circles-30s.txt',
    'Shapes-Circles.txt',
    'Shapes-Hexagons-30s.txt',
    'Shapes-Hexagons.txt',
    'Shapes-Kaleidoscope-30s.txt',
    'Shapes-Kaleidoscope.txt',
    'Shapes-Squares-30s.txt',
    'Shapes-Squares.txt',
    'Shapes-Stars-30s.txt',
    'Shapes-Stars.txt',
    'Spacewalk-30s.txt',
    'Spacewalk.txt',
    'Transitions.txt',
    'Zoom-In-Out-30s.txt',
    'Zoom-In-Out.txt',
    'Zoom-To-New-30s.txt',
    'Zoom-To-New.txt',
    # TODO: this one seems to have a lot of variance run-to-run. Need to find out why.
    # 'Evolve-Glitch-2-30s.txt',
    # 'Evolve-Glitch-2.txt',
    # 'Evolve-Glitch-30s.txt',
    # 'Evolve-Glitch.txt',

]

##
# Runs full end-to-end generations of a small number of frames on many settings files.
# Validates frame-by-frame parameters and final video properties. Compares the output
# video to a baseline.
##
class TestSettingsFilesShort:

    MAX_FRAMES = 10

    @pytest.fixture(scope="class", autouse=True)
    def class_setup_teardown(cls):
        # Apparently this helps force sequential test execution which is necessary when using constrained resources like a GPU.
        # TODO: validate whether this is necessary.
        yield


    @pytest.fixture
    def deforum(cls):
        deforum = DeforumAnimationPipeline.from_civitai(model_id="125703")
        deforum.generator.optimize = True
        return deforum


    @pytest.fixture
    def test_name(self, request):
        return request.node.name


    @pytest.fixture
    def baseline_video_path(self, test_name):
        test_script_path = os.path.abspath(__file__)
        baseline_directory = os.path.join(os.path.dirname(test_script_path), "__baseline_videos__")
        os.makedirs(baseline_directory, exist_ok=True)
        return os.path.join(baseline_directory, f"{test_name}__baseline.mp4")


    @pytest.mark.parametrize("settings_filename", settings_files)
    def test_settings_file_short(self, test_name, deforum, settings_filename, snapshot, update_baseline_videos, baseline_video_path):

        if not os.path.exists(baseline_video_path) and not update_baseline_videos:
            pytest.fail(f"Baseline video not found at {baseline_video_path}. To create baseline videos, run pytest with --update-baseline-videos.")

        # Setup
        outdir = os.path.join(config.output_dir, "tests", "e2etests", test_name)
        preset_dir = os.path.join(config.presets_path, "settings")
        settings_filepath = os.path.join(preset_dir, settings_filename)
        if not os.path.exists(settings_filepath):
            pytest.fail(f"Settings file not found at {settings_filepath}. Please ensure you have cloned the settings files as per the README.")

        extra_args = {
            "settings_file": settings_filepath
        }
        settings_data = load_settings(settings_filepath)
        settings_data.update({
            "outdir": outdir,
            "batch_name": test_name,
            "max_frames": self.MAX_FRAMES,
            "hybrid_use_full_video": False,
            "video_init_path": os.path.join(preset_dir, settings_data.get("video_init_path", "")),
            "prompts": {"0": "A solo delorean speeding on an ethereal highway through time jumps, like in the iconic movie back to the future."},
            "deforum_save_gen_info_as_srt": True,

            # remove all randomness from seeds and seed schedules by removing all occurrences of -1
            "seed": 420,
            "seed_schedule": str(settings_data.get("seed_schedule", "(0):420")).replace("(-1)", "(420)"),
            "subseed":1,
            "subseed_schedule": str(settings_data.get("subseed_schedule", "(0):1")).replace("(-1)", "1"),
        })

        # Act
        result = deforum(**extra_args, **settings_data)

        # Validate final fps and frame count after FILM/RIFE
        expected_frame_count, expected_fps = get_expected_frame_properties(settings_data)
        total_frames, fps = get_video_properties(result.video_path)
        assert total_frames == expected_frame_count
        assert fps == expected_fps

        # assert frame-by-frame parameters via SRT snapshot comparison
        with open(result.srt_filename, "r") as f:
            result_srt = f.read()
            assert result_srt == snapshot

        # assert video similarity to baseline (TODO convert this to a syrupy custom snapshot comparison)
        if (update_baseline_videos):
            shutil.copyfile(result.video_path, baseline_video_path)
            pytest.skip(f"Updated baseline file: {baseline_video_path}. Re-run the tests without --update-baseline-videos to confirm pass.")
        else:
            baseline_frames = get_frames(baseline_video_path)
            result_frames = get_frames(result.video_path)
            comparison = VideoComparisonResult.of(baseline_frames, result_frames)
            if (comparison.average_ssim < VideoComparisonResult.ssim_threshold or comparison.average_psnr < VideoComparisonResult.psnr_threshold):
                scores = list(zip(range(0, len(comparison.ssim_scores)), comparison.ssim_scores, comparison.psnr_scores))
                scores.insert(0, ("Frame", "SSIM", "PSNR"))
                logger.warning(f"Scores per frame: {scores}")
                pytest.fail(f"Video comparison failed. [{baseline_video_path} vs {result.video_path}]. SSIM: {comparison.average_ssim} / PSNR: {comparison.average_psnr}. To overwrite baseline videos, run pytest with --update-baseline-videos.")


# TODO - move to a test utils module
def get_expected_frame_properties(settings_data):
        uninterpolated_frame_count = settings_data.get("max_frames")
        uninterpolated_fps = settings_data.get("fps")

        standard_interpolation_factor = 1
        slow_mo_interpolation_factor = 1
        if settings_data.get("frame_interpolation_engine", "None") != "None":
            standard_interpolation_factor = settings_data.get("frame_interpolation_x_amount", 1)
            if settings_data.get("frame_interpolation_slow_mo_enabled"):
              slow_mo_interpolation_factor = settings_data.get("frame_interpolation_slow_mo_amount", 1)

        expected_frame_count = uninterpolated_frame_count * standard_interpolation_factor * slow_mo_interpolation_factor
        expected_fps = uninterpolated_fps * standard_interpolation_factor

        return expected_frame_count, expected_fps
