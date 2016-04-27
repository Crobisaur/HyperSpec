__author__ = "Christo Robison"

import numpy as np
import h5py
from PIL import Image
import os
import collections

'''This program reads in BSQ datacubes into an HDF file'''

def loadBSQ(path = '/home/crob/HyperSpec_Data', debug=False):
    a = []
    l = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".bsq"):

