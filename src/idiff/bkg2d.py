'''
Module: idiff.bkg2d
-------------------

Background subtraction for 2D-arrays/images.

The module provides several background-removal approaches:

- rolling-ball background subtraction;
- white top-hat morphological filtering;
- Gaussian-blur background estimation;
- connected-component cleanup;
- ONNX neural-network inference wrapper.

Typical input is a 2D diffraction/image array, for example
an image loaded from STEMDIFF/IDIFF-related workflows.
'''


import skimage as sk
from scipy.ndimage import white_tophat
from skimage.measure import label, regionprops
from skimage.morphology import disk
import cv2
import numpy as np
import onnxruntime as ort


def rolling_ball(arr, radius=20):
    """
    Subtract smooth background using the rolling-ball algorithm.

    The rolling-ball algorithm estimates a slowly varying background
    by rolling a ball-shaped structuring element underneath the image
    intensity surface. The estimated background is then subtracted
    from the original image.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D image/array. Usually a diffraction pattern or microscopy
        image. The array is converted to ``np.float32`` if it is not already
        a floating type.

    radius : int, optional
        Radius of the rolling ball, in pixels. Larger values remove
        broader, more slowly varying background. Smaller values remove
        more local background structure. Default is 20.

    Returns
    -------
    arr_bcorr : np.ndarray
        Background-subtracted array.

    Raises
    ------
    ValueError
        If ``arr`` is not 2D or if ``radius`` is not positive.
    """
    # Get background from RollingBall algorithm in sk = skimage
    background = sk.restoration.rolling_ball(arr, radius=radius)

    # Subtract background from original array
    arr_bcorr = arr - background

    # Return array with subtracted background
    return(arr_bcorr)


def tophat(arr, thr=40, area_size=5, radius=2):
    """
    Subtract background using a white top-hat morphological operation.

    White top-hat filtering extracts bright features that are smaller
    than the chosen structuring element. It is useful when diffraction
    spots/features are bright and compact relative to a smoother
    background.

    After top-hat filtering, small connected components are removed using
    ``_remove_small_components``.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D image/array.

    thr : float, optional
        Intensity threshold applied after top-hat filtering. Pixels with
        values less than or equal to this threshold are treated as background.
        Default is 40.

    area_size : int, optional
        Minimum connected-component area, in pixels, required for a feature
        to be retained. Default is 5.

    radius : int, optional
        Radius of the disk-shaped structuring element used by the top-hat
        operation. Default is 2.

    Returns
    -------
    arr_bcorr : np.ndarray
        Top-hat filtered array with small/noisy connected components removed.

    Raises
    ------
    ValueError
        If ``arr`` is not 2D, ``radius`` is not positive, or ``area_size`` is
        negative.
    """
    # Apply white top-hat morphological operation
    result = white_tophat(arr, footprint=disk(radius))
    
    return _remove_small_components(result, thr, area_size)


def gaussian(arr, thr=20, area_size=5, sigma=2, normalize=False):
    """
    Subtract background using Gaussian blur and remove small components.

    This method estimates the smooth background by Gaussian blurring
    the input image. The estimated background is clipped so that it
    does not exceed the original image pixel-by-pixel, then subtracted.
    This avoids creating negative values in ordinary non-negative images.

    After subtraction, small connected components are removed with
    ``_remove_small_components``.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D image/array. Converted to ``np.float32`` unless already
        ``np.float32`` or ``np.float64``.

    thr : float, optional
        Intensity threshold applied after background subtraction. Pixels
        with values less than or equal to this threshold are considered
        background. Default is 20.

    area_size : int, optional
        Minimum connected-component area, in pixels, required for a feature
        to be retained. Default is 5.

    sigma : float, optional
        Standard deviation of the Gaussian blur kernel, in pixels. Larger
        values estimate a smoother, broader background. Default is 2.

    normalize : bool, optional
        If True, divide the background-subtracted array by its mean before
        thresholding, then restore the original scale afterward. This can
        make thresholding less sensitive to absolute intensity scale. Default
        is False.

    Returns
    -------
    arr_bcorr : np.ndarray
        Background-subtracted array with small components removed.

    Raises
    ------
    ValueError
        If ``arr`` is not 2D, ``sigma`` is not positive, or ``area_size``
        is negative.

    Notes
    -----
    If ``normalize=True`` and the mean of the background-subtracted image
    is too close to zero, normalization is skipped and the unfiltered
    background-subtracted image is returned. This avoids division by zero.
    """
    # Use floating point arithmetic for subtraction and filtering.
    if not (arr.dtype == np.float32 or arr.dtype == np.float64):
        arr = arr.astype(np.float32)

    # Estimate smooth background by Gaussian blur.
    b = cv2.GaussianBlur(arr, (0, 0), sigma)

    # Prevent the estimated background from being larger than
    # the original  signal at each pixel. For non-negative images, this
    # keeps arr - background non-negative.
    b = np.clip(b, 0, arr)

    # Subtract estimated background.
    arr = arr - b

    if normalize:
        mean = np.mean(arr)

        # Use absolute value here so negative near-zero means are
        # also handled.
        if mean < 1e-4:
            return arr

        # Normalize before thresholding.
        arr = arr / mean
        result = _remove_small_components(arr, thr, area_size)

        # Restore original intensity scale.
        result *= mean

        return result

    return _remove_small_components(arr, thr, area_size)


def _remove_small_components(arr, thr, area_size):
    """
    Remove small connected components from a thresholded image.

    Pixels above ``thr`` are treated as foreground. Connected foreground
    regions are labeled using 4-connectivity. Only regions with area
    greater than or equal to ``area_size`` are retained.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array, usually after background subtraction.

    thr : float
        Threshold for foreground detection. Pixels satisfying ``arr > thr``
        are considered part of candidate components.

    area_size : int
        Minimum connected-component area, in pixels, required for retention.

    Returns
    -------
    refined_output : np.ndarray
        Array with only sufficiently large connected components retained.
        Pixels outside retained components are set to zero.

    Raises
    ------
    ValueError
        If ``arr`` is not 2D or ``area_size`` is negative.
    """
    # Binary foreground mask.
    mask = arr > thr

    # Label connected foreground regions.
    # connectivity=1 means 4-connectivity in 2D.
    labeled_mask = label(mask, connectivity=1)  # 1 = 4-connectivity

    # Output starts as zeros; only accepted components are copied back.
    refined_output = np.zeros_like(arr)

    # Iterate over connected regions. The intensity image lets regionprops
    # associate each component with values from the original array.
    for region in regionprops(labeled_mask, intensity_image=arr):
        if region.area >= area_size:
            # region.coords has shape (N, 2), where columns are row, col.
            # Transpose to make it usable for NumPy advanced indexing.
            coords = tuple(zip(*region.coords))
            refined_output[coords] = arr[coords]
    
    return refined_output


class NeuralNetwork:
    """
    ONNX Runtime wrapper for neural-network background subtraction.

    The class loads an ONNX model and provides a simple ``predict`` method
    for running inference on 2D or 3D NumPy arrays.

    Expected input shapes
    ---------------------
    ``(H, W)``
        Single 2D image. Internally reshaped to ``(1, 1, H, W)``.

    ``(N, H, W)``
        Batch/stack of 2D images. Internally reshaped to ``(N, 1, H, W)``.

    The returned output is reshaped back to the original input shape.
    """
    def __init__(self, path: str, **kwargs):
        """
        Initialize the ONNX Runtime inference session.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ONNX model file.

        **kwargs
            Additional keyword arguments passed directly to
            ``onnxruntime.InferenceSession``. This can be used to specify
            providers, session options, etc.

        Returns
        -------
        None
        """
        # Load ONNX model into an inference session.
        self.model = ort.InferenceSession(path, **kwargs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run neural-network background subtraction.

        The input is converted to ``np.float32`` and reshaped into the common
        ONNX image tensor format ``(N, C, H, W)`` with one channel. After model
        inference, the output is reshaped back to the original input shape.

        Parameters
        ----------
        x : np.ndarray
            Input image or image stack.

            Supported shapes are:

            - ``(H, W)`` for one image;
            - ``(N, H, W)`` for a stack of images.

        Returns
        -------
        clean_array : np.ndarray
            Model output reshaped to the same shape as the original input.

        Raises
        ------
        ValueError
            If the input is not 2D or 3D, or if the model output cannot be
            reshaped back to the original input shape.
        """
        # Save original shape
        original_shape = x.shape

        # Convert to NCHW format with one channel.
        if len(x.shape) == 2:
            # (H, W) -> (1, 1, H, W)
            x = x[None, None]

        elif len(x.shape) == 3:
            # (N, H, W) -> (N, 1, H, W)
            x = x[:, None]

        # ONNX models usually expect float32 input.
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        # Get the model's first input tensor name.
        input_name = self.model.get_inputs()[0].name
        clean = self.model.run(None, {input_name: x})[0]

        # Reshape to original shape and return result
        return np.reshape(clean, original_shape)
