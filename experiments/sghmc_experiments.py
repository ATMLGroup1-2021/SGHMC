import os
import sys
import matplotlib.pyplot as plt
import numpy as np
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from samplers.sghmc import SGHMC

'''
Perform testing for SGHMC and HMC using conjugate prior
distributions, inorder to compare with analytical posterior
'''