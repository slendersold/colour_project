import numpy as np
from colour import XYZ_to_Lab, delta_E
import tifffile
import zarr
from colour_lib.utils.CustomCCTF import CustomCCTF

from colour_lib.utils.rawparser import RawDataParser
from colour_lib.utils.circlelib import *
from colour_lib.utils.CustomCCTF import CustomCCTF

CCTF = CustomCCTF()


def image_read(img, level, type):
    store = tifffile.imread(img, aszarr=True)
    zarr_pyramids = zarr.open(store, mode="r")
    image = np.array(zarr_pyramids[level]) / 255
    image_revert = CCTF.apply_CCTF(mode="decode", cctf_type=type, image=image)
    return image_revert


def calculate_delta_E(observe, reference):
    observe = XYZ_to_Lab(observe)
    reference = XYZ_to_Lab(reference)
    deltas = np.zeros((observe.shape[0], 1))
    for i in range(observe.shape[0]):
        a = reference[i, :]
        b = observe[i, :]
        deltas[i] = delta_E(a, b, method="CIE 2000")
    return deltas
