from deforum import DeforumAnimationPipeline

pipeline = DeforumAnimationPipeline.from_civitai("125703", generator_name="DeforumDiffusersGenerator")

args = {"width": 1024, "height": 1024, "prompts": {0:"cat sushi"}}
pipeline(**args)