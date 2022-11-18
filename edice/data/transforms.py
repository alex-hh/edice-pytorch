import torch
import numpy as np


TRANSFORMATIONS = ["arcsinh"]


def apply_transformation(transform, values):
    if transform is None:
        return values
    elif transform == 'arcsinh':
        return np.arcsinh(values)
    else:
        raise ValueError('wrong transform value')
