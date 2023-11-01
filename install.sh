#!/bin/bash

# create a virtual environment
python -m venv venv
source venv/bin/activate

# install dependencies
pip install -e .["dev"]