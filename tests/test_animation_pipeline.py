from deforum import DeforumAnimationPipeline

pipeline = DeforumAnimationPipeline.from_civitai("125703")

args = {"width": 1024, "height": 1024, "max_frames": 5}
pipeline(**args)