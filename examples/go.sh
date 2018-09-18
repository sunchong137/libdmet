#! /bin/bash

source ~/.bashrc_libdmet
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
python Hub2dBCS_Localized.py 
