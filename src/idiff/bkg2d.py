'''
Module: idiff.bkg2d
-------------------

Background subtraction for 2D-arrays/images.
'''


import skimage as sk
import numpy as np
import onnxruntime as ort


def rolling_ball(arr, radius=20):
    '''
    Subtract background from an array using *rolling ball* algorithm.

    Parameters
    ----------
    arr : numpy array
        Original array.
        Usually 2D-array supplied from package stemdiff.
    radius : int, optional, default is 20
        Radius of the rolling ball

    Returns
    -------
    arr_bcorr : numpy array
        The array with the subtracted background.
    '''
    # Get background from RollingBall algorithm in sk = skimage
    background = sk.restoration.rolling_ball(arr, radius=radius)
    # Subtract background from original array
    arr_bcorr = arr - background
    # Return array with subtracted background
    return(arr_bcorr)


class NeuralNetwork:
    def __init__(self, path):
        self.model = ort.InferenceSession(path)

    def predict(self, x):
        if len(x.shape) == 2:
            x = x[None, None]
        elif len(x.shape) == 3:
            x = x[:, None]
        
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        clean, bkg = self.model.run(None, {"x": x})

        return clean.squeeze(), bkg.squeeze()