#@markdown **Environment Setup**
import os
import subprocess
import sys
import time

# Specify requirements in a list for easy editing
requirements = ['einops', 'timm', 'basicsr', 'accelerate', 'transformers', 'torchsde']

def setup_environment(reqs):
    # Check if in google colab
    try:
        ipy = get_ipython()
        if 'google.colab' not in str(ipy):
            print('Not running on Google Colab. Environment setup aborted.\n')
            print('```')
            print('Environment: Non Google Colab')
            print('Error: This notebook is not running in Google Colab.')
            print('```')
            print("Please ensure this notebook runs in Google Colab.")
            print("If this problem persists, please open an issue at: https://github.com/deforum-studio/deforum/issues")
            return
    except Exception:
        print("Error! Not running in an IPython environment. Environment setup aborted.\n")
        print('```')
        print('Environment: Non IPython')
        print('Error: This notebook is not running in an IPython environment.')
        print('```')
        print("If this problem persists, please open an issue at: https://github.com/deforum-studio/deforum/issues")
        return

    # Start the timer
    start_time = time.time()

    try:
        # Clone the deforum repo
        process = subprocess.Popen(['git', 'clone', 'https://github.com/deforum-studio/deforum.git'])
        process.wait()

        # Install required packages in one go
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *reqs])

        # Append the path
        sys.path.append('./deforum/src/')
    except Exception as e:
        print("An error occurred during the setup.\n")
        print('```')
        print('Error:', str(e))
        print('```')
        print("Please copy the above text and open an issue at: https://github.com/deforum-studio/deforum/issues")
        return
      
    # Report the total time
    total_time = time.time() - start_time
    print('Environment setup completed successfully. Total time: {:.2f} seconds'.format(total_time))

# Call the function with the requirements list as argument
setup_environment(requirements)