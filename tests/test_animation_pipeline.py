import time

from deforum import DeforumAnimationPipeline

pipeline = DeforumAnimationPipeline.from_civitai("125703")

args = {"settings_file":"presets/Zoom-To-New.txt"}

start_time = time.time()
pipeline(**args)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")