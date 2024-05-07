import os
import subprocess


def install_sfast():
    os.environ['MAX_JOBS'] = str(8)

    subprocess.run(
        [
            "python",
            "-m" "pip",
            "install",
            "ninja",
        ]
    )

    subprocess.run(
        [
            "python",
            "-m" "pip",
            "install",
            "-v",
            "-U",
            "git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast",
        ]
    )
