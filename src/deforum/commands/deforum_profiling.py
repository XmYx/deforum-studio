from line_profiler import LineProfiler

def start_deforum_test():
    def profile_deforum():
        from deforum import DeforumAnimationPipeline
        deforum = DeforumAnimationPipeline.from_civitai()
        options = {"settings_file": 'tests/gpt_bird.txt'}
        _ = deforum(**options)
        # Your code here

    profiler = LineProfiler()
    profiler.add_function(profile_deforum)
    with profiler:
        profile_deforum()

    profiler.print_stats()