#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

import psutil
import torch

def print_memory_usage():
    try:
        # Print CPU Memory Usage
        cpu_memory = psutil.virtual_memory()
        print("CPU Memory Usage:")
        print(f"Total: {cpu_memory.total / (1024.0 ** 3):.2f} GB")
        print(f"Available: {cpu_memory.available / (1024.0 ** 3):.2f} GB")
        print(f"Used: {cpu_memory.used / (1024.0 ** 3):.2f} GB")
        print(f"Percentage Used: {cpu_memory.percent}%")
    except Exception as e:
        print(f"Error while fetching CPU memory usage: {e}")

    # try:
    #     # Print GPU Memory Usage using GPUtil
    #     print("\nGPU Memory Usage (GPUtil):")
    #     GPUtil.showUtilization()
    # except Exception as e:
    #     print(f"Error while fetching GPU memory usage with GPUtil: {e}")

    try:
        # Print GPU Memory Usage using PyTorch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("\nGPU Memory Usage (PyTorch):")
            print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3):.2f} GB")
            print(f"Allocated Memory: {torch.cuda.memory_allocated(device) / (1024.0 ** 3):.2f} GB")
            print(f"Reserved Memory: {torch.cuda.memory_reserved(device) / (1024.0 ** 3):.2f} GB")
        else:
            print("\nCUDA is not available. Please check your GPU setup.")
    except Exception as e:
        print(f"Error while fetching GPU memory usage with PyTorch: {e}")

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


