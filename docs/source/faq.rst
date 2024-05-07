FAQ
===

Here are some frequently asked questions about installing and using Deforum, with a focus on the proper use of virtual environments to avoid conflicts.

**Q: I'm new to Python. What do I need to install Deforum?**
------------------------------------------------------------------

**A:** To install Deforum, you will need Python installed on your system. Python 3.10 is recommended. You can download it from `python.org <https://www.python.org/downloads/>`. After installing Python, you should set up a virtual environment to avoid conflicts with other Python projects. See below for instructions on how to create a virtual environment.

**Q: How do I create a virtual environment for Python?**
------------------------------------------------------

**A:** A virtual environment is an isolated environment that keeps the dependencies required by different projects separate. It's highly recommended to use a virtual environment when installing software like Deforum to prevent conflicts. Here’s how to create one:

**Using virtualenv**:

.. code-block:: bash

   pip install virtualenv
   virtualenv venv -p 3.10
   source venv/bin/activate

**Using conda**:

.. code-block:: bash

   conda create -n deforum python=3.10
   conda activate deforum

Always activate your virtual environment before installing or running Deforum.

**Q: What should I do if I accidentally installed Deforum in the system environment?**
--------------------------------------------------------------------------------------

**A:** If Deforum was installed in the system Python environment accidentally, it’s best to uninstall it and reinstall it within a virtual environment to avoid potential conflicts with other Python applications on your system:

.. code-block:: bash

   pip uninstall deforum

Then, create a virtual environment (as shown above) and reinstall Deforum there.

**Q: How can I verify that Deforum is installed correctly?**
-----------------------------------------------------------

**A:** After setting up your virtual environment and installing Deforum, you can verify the installation by running sample tests included in the package:

.. code-block:: bash

   COMFY_PATH=src/ComfyUI python tests/test_animation_pipeline.py

Ensure you are within your virtual environment when running tests. This command should execute without errors, confirming that Deforum is installed correctly.

**Q: What should I do if I get an error during installation from PyPI?**
----------------------------------------------------------------------

**A:** Ensure you are in your virtual environment when installing from PyPI. Errors during installation can be due to network issues, outdated pip versions, or incorrect permissions. First, update pip within your virtual environment:

.. code-block:: bash

   pip install --upgrade pip

If the problem persists, check your internet connection, or consider downloading Deforum from the source as described in the installation instructions.

**Q: The installation completed but I can't start the WebUI. What's wrong?**
---------------------------------------------------------------------------

**A:** Verify that you have set `COMFY_PATH` correctly and that you are running the command within your virtual environment. Here is the correct command sequence:

.. code-block:: bash

   export COMFY_PATH=src/ComfyUI
   deforum webui

Check the console output for any error messages if the WebUI fails to start, and address the specific errors mentioned.

**Q: How do I update Deforum to the latest version?**
---------------------------------------------------

**A:** To update Deforum, activate your virtual environment where Deforum is installed, and then run:

.. code-block:: bash

   pip install --upgrade deforum

If you installed from source, update with these commands:

.. code-block:: bash

   cd deforum
   git pull
   pip install -e .["cli"]
