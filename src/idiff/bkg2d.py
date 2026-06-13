'''
Module: idiff.bkg2d
-------------------

Background subtraction for 2D-arrays/images.
'''


import skimage as sk
from scipy.ndimage import white_tophat
from skimage.measure import label, regionprops
from skimage.morphology import disk
import cv2
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
    '''
    Subtract background from an array using a *white top-hat* morphological
    operation.

    Parameters
    ----------
    arr : numpy array
        Original array.
        Usually 2D-array supplied from package stemdiff.
    thr : float, optional, default is 40
        Intensity threshold for keeping significant features.
    area_size : int, optional, default is 5
        Minimum area (in pixels) for a connected component to be retained.
    radius : int, optional, default is 2
        Radius of the disk-shaped structuring element.

    Returns
    -------
    arr_bcorr : numpy array
        The array with subtracted background and small noise removed.
    '''
    # Apply white top-hat morphological operation
    result = white_tophat(arr, footprint=disk(radius))
    
    return _remove_small_components(result, thr, area_size)


def gaussian(arr, thr=20, area_size=5, sigma=2, normalize=False):
    '''
    Subtract background from an array using a Gaussian blur filter and 
    remove small connected components.

    This function first estimates the background using a Gaussian blur, 
    subtracts it from the original array, and then filters out small 
    connected components that fall below a given intensity threshold and 
    area size.

    Parameters
    ----------
    arr : numpy array
        Original array. Usually a 2D array supplied from package stemdiff.
    thr : float, optional, default is 20
        Intensity threshold for the background-subtracted array. 
        Connected components with peak intensities below this value 
        are considered background and removed.
    area_size : int, optional, default is 5
        Minimum area (in pixels) for a connected component to be retained. 
        Smaller components are removed regardless of intensity.
    sigma : float, optional, default is 2
        Standard deviation for the Gaussian kernel used to estimate 
        the background.
    normalize : bool, optional, default is False
        If True, the background-subtracted array is normalized by its 
        mean before small component removal. The original scale is 
        restored in the final output. This can help in thresholding 
        features relative to the local signal strength.

    Returns
    -------
    arr_bcorr : numpy array
        The array with background subtracted and small noise/components
        removed.

    Notes
    -----
    - If `normalize` is True, the mean of the background-subtracted array
      is used for normalization. If the mean is near zero (< 1e-4), 
      normalization is skipped to avoid division by zero, and the 
      unnormalized result is returned before small component removal.
    - The small component removal uses 4-connectivity (up, down, left, right).
    - The `thr` parameter is applied to the background-subtracted data, 
      not the original raw data.
    - If the array's dtype is not np.float32 or np.float64, it is converted
      to np.float32.
    '''
    if not (arr.dtype == np.float32 or arr.dtype == np.float64):
        arr = arr.astype(np.float32)  

    b = cv2.GaussianBlur(arr, (0, 0), sigma)
    b = np.clip(b, 0, arr)
    arr = arr - b

    if normalize:
        mean = np.mean(arr)
        if mean < 1e-4:
            return arr
        arr = arr / mean
    
    result = _remove_small_components(arr, thr, area_size)

    # Undo normalization
    if normalize:
        result *= mean

    return result


def _remove_small_components(arr, thr, area_size):
    '''
    Filter out small connected components based on intensity threshold and area
    size.

    Parameters
    ----------
    arr : numpy array
        Background-subtracted array.
    thr : float
        Intensity threshold; pixels below this value are treated as background.
    area_size : int
        Minimum number of connected pixels required to keep a feature.

    Returns
    -------
    refined_output : numpy array
        Array with only significant connected components retained.
    '''

    # Threshold the array
    mask = arr > thr
    
    # Label connected components
    labeled_mask = label(mask, connectivity=1)  # 1 = 4-connectivity
    
    refined_output = np.zeros_like(arr)
    
    # Iterate over connected regions
    for region in regionprops(labeled_mask, intensity_image=arr):
        if region.area >= area_size:
            coords = tuple(zip(*region.coords))
            refined_output[coords] = arr[coords]
    
    return refined_output


class NeuralNetwork:
    ''' Neural network for subtracting background.
    This class is a wrapper for ONNX Runtime inference session that runs
    the neural network.
    '''
    def __init__(self, path: str, **kwargs):
        '''
        Initialize the ONNX Runtime inference session. This runs the 
        neural network model for subtracting background.

        Parameters
        ----------
        path : str
            File path to the ONNX model.
        **kwargs : dict
            Additional arguments passed to `onnxruntime.InferenceSession`.

        Returns
        -------
        None
            Initializes the model session as an instance attribute.
        '''
        self.model = ort.InferenceSession(path, **kwargs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        '''
        Run the ONNX model on the input array to subtract the background.

        Parameters
        ----------
        x : numpy array
            Input array, typically 2D or 3D image data (multiple images).
            Will be reshaped and converted to `np.float32` if necessary.

        Returns
        -------
        clean_array : numpy array
            Array with subtracted background, reshaped to match the original
            input dimensions.
        '''
        # Save original shape
        original_shape = x.shape

        # Convert input to required shape and type
        if len(x.shape) == 2:
            x = x[None, None]
        elif len(x.shape) == 3:
            x = x[:, None]
        
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # Get name of the input argument
        input_name = self.model.get_inputs()[0].name
        clean = self.model.run(None, {input_name: x})[0]

        # Reshape to original shape and return result
        return np.reshape(clean, original_shape)
