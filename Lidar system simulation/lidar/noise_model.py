import numpy as np

def add_noise(points, std=0.02):
    return points + np.random.normal(scale=std, size=points.shape)
