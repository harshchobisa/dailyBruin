#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda config --add channels conda-forge
conda config --add channels menpo
conda config --add channels pytorch
conda create --name DBenvs -y
conda init
conda activate DBenvs
conda install  --file aws_requirements.txt -y
pip install omegaconf
pip install hydra-core
pip install tdqm
