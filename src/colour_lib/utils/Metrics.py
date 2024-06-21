import numpy as np
from colour import XYZ_to_Lab, delta_E

def calculate_delta_E(observe, reference):
    observe = XYZ_to_Lab(observe)
    reference = XYZ_to_Lab(reference)
    deltas = np.zeros((observe.shape[0], 1))
    for i in range(observe.shape[0]):
        a = reference[i, :]
        b = observe[i, :]
        deltas[i] = delta_E(a, b, method="CIE 2000")
    return deltas