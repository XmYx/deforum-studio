import time


def test_loading_unloading_deforum():
    from deforum import DeforumAnimationPipeline

    deforum = DeforumAnimationPipeline.from_civitai()

    deforum.cleanup()

def test_run_deforum():
    from deforum import DeforumAnimationPipeline

    deforum = DeforumAnimationPipeline.from_civitai()

    _ = deforum(W=768,
                H=768,
                max_frames=1,
                video_init_path="")


    deforum.cleanup()

