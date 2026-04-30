'''
Module: idiff.bkg2d
-------------------

Background subtraction for 2D-arrays/images.
'''


import skimage as sk
from scipy.ndimage import white_tophat
from skimage.measure import label, regionprops
from skimage.morphology import disk
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

def tophat(arr, thr=40, area_size=5, radius=2):
    # Apply white top-hat morphological operation
    tophat_result = white_tophat(arr, footprint=disk(radius))
    
    # Threshold the result
    mask = tophat_result > thr
    
    # Label connected components
    labeled_mask = label(mask, connectivity=1)  # 1 = 4-connectivity
    
    refined_output = np.zeros_like(arr)
    
    # Iterate over connected regions
    for region in regionprops(labeled_mask, intensity_image=tophat_result):
        if region.area >= area_size:
            coords = tuple(zip(*region.coords))
            refined_output[coords] = tophat_result[coords]
    
    return refined_output

class NeuralNetwork:
    def __init__(self, path: str):
        self.model = ort.InferenceSession(path)

    def predict(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape

        if len(x.shape) == 2:
            x = x[None, None]
        elif len(x.shape) == 3:
            x = x[:, None]
        
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        clean = self.model.run(None, {"x": x})

        if x.dtype != np.float32:
            clean = np.round(clean).astype(x.dtype)

        return np.reshape(clean, original_shape)