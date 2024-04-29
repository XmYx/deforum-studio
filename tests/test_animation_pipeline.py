import os
os.environ['COMFY_PATH'] = 'src/ComfyUI'

from deforum import DeforumAnimationPipeline

pipeline = DeforumAnimationPipeline.from_civitai("125703")

args = {'settings_file':'presets/Grids.txt'}
anim = pipeline(**args)