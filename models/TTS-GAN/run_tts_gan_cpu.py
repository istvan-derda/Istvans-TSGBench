# import torch
# 
# # Monkey patch to disable CUDA usage
# torch.cuda.is_available = lambda: False
# torch.cuda.FloatTensor = torch.FloatTensor

import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), 'TTS_GAN')) # fix relative paths in TTS_GAN

import TTS_GAN.JumpingGAN_Train