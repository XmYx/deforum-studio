from deforum import DeforumAnimationPipeline

pipeline = DeforumAnimationPipeline.from_civitai("125703")

args = {"W": 1024, "H": 1024}
pipeline(**args)