Installation
============

Deforum is a state-of-the-art Animation Diffusion framework implemented in PyTorch and TRT. This section provides detailed instructions on different methods to install Deforum.

Virtual Environment
-------------------

**Using virtualenv**:

.. code-block:: bash

   pip install virtualenv
   virtualenv venv -p 3.10
   source venv/bin/activate

**Using conda**:

.. code-block:: bash

   conda create -n deforum python=3.10
   conda activate deforum

From Source
-----------

To install Deforum directly from the source:

.. code-block:: bash

   git clone https://github.com/deforum-studio/deforum
   cd deforum
   pip install -e .["cli"]

For Developers
--------------

For developers, install with developer dependencies:

.. code-block:: bash

   git clone https://github.com/deforum-studio/deforum
   cd deforum
   pip install -e .["dev"]

Testing
-------

After installation, you can test the installation by running various tests.

**Test the animation pipeline**:

.. code-block:: bash

   COMFY_PATH=src/ComfyUI python tests/test_animation_pipeline.py

**Test the diffusers pipeline**:

.. code-block:: bash

   python tests/test_diffusers_pipeline.py

**Test all presets**:

.. code-block:: bash

   deforum runpresets --options randomize_files=True

WebUI
-----

To launch the WebUI, use the following command:

.. code-block:: bash

   COMFY_PATH=src/ComfyUI deforum webui

**Instructions for WebUI**:

Upload the `deforum.txt` file from the Presets folder into the WebUI. Make sure to check "Use Settings File" to load and edit parameters directly through the UI.

Stable-Fast
-----------

Install accelerated inference libraries:

.. code-block:: bash

   deforum setup

qtpy GUI
--------

To launch the Qt GUI:

.. code-block:: bash

   deforum ui

License
-------

Deforum is licensed under the GNU General Public License v3.0.

For more details, please refer to the LICENSE file in the source repository.
