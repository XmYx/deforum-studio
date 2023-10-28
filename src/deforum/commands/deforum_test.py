"""
Simple test protocol that runs all tests that require instantiating the pipeline from scratch,
as well as consecutively running ones.
"""


def start_deforum_test():
    from deforum import DeforumAnimationPipeline
    test_files = ['tests/gpt_bird.txt', 'tests/gpt_bird.txt']
    consecutive_test_files = ['tests/gpt_bird.txt', 'tests/gpt_bird.txt']
    for file in test_files:
        options = {"settings_file": file}
        deforum = DeforumAnimationPipeline.from_civitai()
        _ = deforum(**options)
        del _
        del deforum

    deforum = DeforumAnimationPipeline.from_civitai()

    for file in consecutive_test_files:
        options = {"settings_file": file}
        _ = deforum(**options)
        del _
