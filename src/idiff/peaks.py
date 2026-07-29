'''
Module: idiff.peaks
-------------------

Find diffraction peaks in 2D-arrays/images/diffractograms.

Moved from brach david:

- log
- doh
- pcbr

MSER not moved, because it struggled to find any peaks
(Could be caused by using wrong combination of parameters,
however, multiple combinations were tested).
'''

from scipy import stats
from skimage.feature import (
    blob_log, blob_doh, peak_local_max, hessian_matrix_eigvals, hessian_matrix)
from skimage.transform import resize
from skimage.measure import label, regionprops
import numpy as np
from typing import *

def _estimate_noise(img):
    '''Computes robust sigma using MAD (normal scale factor),
       with safe fallback if MAD is zero/NaN.

    Parameters
    ----------
    img : Numpy 2D array
        Input image.

    Returns
    -------
    float
        Noise sigma.
    '''

    median_val = np.nanmedian(img)
    if np.isnan(median_val):
        return 1.0
    
    mad = stats.median_abs_deviation(img, axis=None, scale='normal', 
                                     nan_policy='omit')

    if np.isnan(mad) or mad == 0:
        noise_sigma = 1.0
    else:
        noise_sigma = mad

    return noise_sigma

def run_log(img, downsample_factor=4, threshold_factor=3.5, peak_min_sigma=6,
             peak_max_sigma=60, num_sigma_steps=30, **kwargs):
    '''Runs LoG detector and returns a Dirac-delta image.

    Parameters
    ----------
    img : Numpy 2D array
        Input image for peak detection.
    downsample_factor : int, optional
        Optional speed-up by processing downsampled image and
        mapping detections back, by default 4
    threshold_factor : float, optional
        Detection threshold scale (used with estimated noise sigma),
        by default 3.5
    peak_min_sigma : float, optional
        by default 6
    peak_max_sigma : float, optional
        by default 60
    num_sigma_steps : int, optional
        by default 30
    **kwargs
        Additional keyword arguments passed to
        :func:`calculate_integrated_intensities` (e.g. ``sigma_factor``,
        ``use_sigma_factor``, ``fixed_radius``).

    Returns
    -------
    Numpy 2D array
        2D float Dirac image (same shape as input):
        each detected peak is a single non-zero pixel equal to
        integrated intensity.
    '''
    original_shape = img.shape
    noise_sigma = _estimate_noise(img)
    abs_threshold = noise_sigma * threshold_factor
    abs_threshold = max(abs_threshold, 1e-6)  # Ensure positive

    if downsample_factor > 1:
        new_shape = (original_shape[0] // downsample_factor,
                     original_shape[1] // downsample_factor)

        if new_shape[0] < 1 or new_shape[1] < 1:
            print("Warning: Downsampled image too small. Skipping.")
            return np.zeros_like(img, dtype=float)

        img_small = resize(img, new_shape, anti_aliasing=True, order=1,
                           preserve_range=True)

        # Scale sigma values
        # Min sigma typically >= 0.5 for skimage
        min_sigma_small = max(0.5, peak_min_sigma / downsample_factor)
        max_sigma_small = max(min_sigma_small + 0.5,
                              peak_max_sigma / downsample_factor)

        if min_sigma_small >= max_sigma_small:
            print("Warning: Invalid sigma range after downscaling",
                  f"({min_sigma_small:.2f}-{max_sigma_small:.2f}).",
                  "Skipping LoG.")
            return np.zeros_like(img, dtype=float)

        blobs_small = blob_log(img_small,
                               min_sigma=min_sigma_small,
                               max_sigma=max_sigma_small,
                               num_sigma=num_sigma_steps,
                               threshold=abs_threshold)

        if blobs_small.shape[0] == 0:
            return np.zeros_like(img, dtype=float)
        # Scale results back to original coordinates/scale
        rows_orig = blobs_small[:, 0] * downsample_factor
        cols_orig = blobs_small[:, 1] * downsample_factor
        scores_orig = blobs_small[:, 2] * downsample_factor  # Scale sigma

        rows, cols, scores = (rows_orig, cols_orig, scores_orig)
    else:
        blobs = blob_log(img, min_sigma=peak_min_sigma,
                         max_sigma=peak_max_sigma,
                         num_sigma=num_sigma_steps,
                         threshold=abs_threshold)

        if blobs.shape[0] == 0:
            return np.zeros_like(img, dtype=float)
        rows, cols, scores = blobs[:, 0], blobs[:, 1], blobs[:, 2]

    # Calculate integrated intensities (use_sigma_factor=True for LoG)
    intensities = calculate_integrated_intensities(
        img, rows, cols, scores,
        use_sigma_factor=kwargs.get('use_sigma_factor', True),
        sigma_factor=kwargs.get('sigma_factor', 3),
        fixed_radius=kwargs.get('fixed_radius', 5)
    )

    return dirac_delta_image(img, rows, cols, intensities)
    
def run_doh(img, downsample_factor=4, threshold_factor=3.5, peak_min_sigma=6,
             peak_max_sigma=60, num_sigma_steps=30, log_scale=False, **kwargs):
    '''Runs DoH detector and returns a Dirac-delta image.

    Parameters
    ----------
    img : Numpy 2D array
        Input image for peak detection.
    downsample_factor : int, optional
        Optional speed-up by processing downsampled image and
        mapping detections back, by default 4
    threshold_factor : float, optional
        Detection threshold scale (used with estimated noise sigma),
        by default 3.5
    peak_min_sigma : float, optional
        by default 6
    peak_max_sigma : float, optional
        by default 60
    num_sigma_steps : int, optional
        by default 30
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base 10.
        If not, linear interpolation is used.
    **kwargs
        Additional keyword arguments passed to
        :func:`calculate_integrated_intensities` (e.g. ``sigma_factor``,
        ``use_sigma_factor``, ``fixed_radius``).

    Returns
    -------
    Numpy 2D array
        2D float Dirac image (same shape as input):
        each detected peak is a single non-zero pixel equal to
        integrated intensity.
    '''
    original_shape = img.shape
    # DoH threshold needs careful tuning, especially with downsampling.
    # This is a heuristic.
    noise_sigma = _estimate_noise(img)
    abs_threshold = noise_sigma * threshold_factor * 0.1
    abs_threshold = max(abs_threshold, 1e-6)

    if downsample_factor > 1:
        new_shape = (original_shape[0] // downsample_factor,
                     original_shape[1] // downsample_factor)
        if new_shape[0] < 1 or new_shape[1] < 1:
            print("Warning: Downsampled image too small. Skipping.")
            return np.zeros_like(img, dtype=float)

        img_small = resize(img, new_shape, anti_aliasing=True, order=1,
                           preserve_range=True)

        min_sigma_small = max(0.5, peak_min_sigma / downsample_factor)
        max_sigma_small = max(min_sigma_small + 0.5,
                              peak_max_sigma / downsample_factor)

        if min_sigma_small >= max_sigma_small:
            print("Warning: Invalid sigma range after downscaling",
                  f"({min_sigma_small:.2f}-{max_sigma_small:.2f}).",
                  "Skipping DoH.")
            return np.zeros_like(img, dtype=float)

        blobs_small = blob_doh(img_small, min_sigma=min_sigma_small,
                               max_sigma=max_sigma_small,
                               num_sigma=num_sigma_steps,
                               threshold=abs_threshold,
                               log_scale=log_scale)

        if blobs_small.shape[0] == 0:
            return np.zeros_like(img, dtype=float)
        rows, cols, scores = (
            blobs_small[:, 0] * downsample_factor,
            blobs_small[:, 1] * downsample_factor,
            blobs_small[:, 2] * downsample_factor
        )
    else:
        blobs = blob_doh(img, min_sigma=peak_min_sigma,
                         max_sigma=peak_max_sigma,
                         num_sigma=num_sigma_steps,
                         threshold=abs_threshold,
                         log_scale=log_scale)

        if blobs.shape[0] == 0:
            return np.zeros_like(img, dtype=float)
        rows, cols, scores = blobs[:, 0], blobs[:, 1], blobs[:, 2]

    # Calculate integrated intensities (use_sigma_factor=True for DoH)
    intensities = calculate_integrated_intensities(
        img, rows, cols, scores,
        use_sigma_factor=kwargs.get('use_sigma_factor', True),
        sigma_factor=kwargs.get('sigma_factor', 3),
        fixed_radius=kwargs.get('fixed_radius', 5)
    )

    return dirac_delta_image(img, rows, cols, intensities)
    
def run_pcbr(img, sigma=3.0, lambda_thresh=0.5, response_thresh_rel=0.1,
              min_distance=5, **kwargs):
    '''Principal-curvature-based response (Hessian eigenvalue + local maxima)
    and returns a Dirac-delta image.

    Parameters
    ----------
    img : Numpy 2D array
        Input image for peak detection.
    sigma : float, optional
        Hessian scale., by default 3.0
    lambda_thresh : float, optional
        Negative-eigenvalue threshold for ridge-like response, by default 0.5
    response_thresh_rel : float, optional
        Relative threshold for local maxima extraction in response map,
        by default 0.1
    min_distance : int, optional
        Minimum spacing between PCBR local maxima, by default 5
    **kwargs
        Additional keyword arguments passed to
        :func:`calculate_integrated_intensities` (e.g. ``use_sigma_factor``,
        ``fixed_radius``).

    Returns
    -------
    Numpy 2D array
        2D float Dirac image (same shape as input):
        each detected peak is a single non-zero pixel equal to
        integrated intensity.
    '''

    try:
        Hrr, Hrc, Hcc = hessian_matrix(img,
            sigma=sigma,
            use_gaussian_derivatives=True,
            mode='nearest',
            order='rc')
    except Exception as e:
        print(f"      ERROR calculating Hessian matrix: {e}. Skipping PCBR.")
        return np.zeros_like(img, dtype=float)

    try:
        if not np.all(np.isfinite(Hrr)) or \
            not np.all(np.isfinite(Hrc)) or \
            not np.all(np.isfinite(Hcc)):
            print("Warning: Non-finite values found in Hessian components. " \
                "Attempting nan_to_num.")
            Hrr = np.nan_to_num(Hrr)
            Hrc = np.nan_to_num(Hrc)
            Hcc = np.nan_to_num(Hcc)
        lambda1, lambda2 = hessian_matrix_eigvals([Hrr, Hrc, Hcc])
    except Exception as e:
        print(f"ERROR calculating Hessian eigenvalues: {e}. Skipping PCBR.")
        return np.zeros_like(img, dtype=float)

    response_map = np.zeros_like(img)
    valid_lambda = np.isfinite(lambda1)
    mask = valid_lambda & (lambda1 < -lambda_thresh)

    if np.any(mask):
        response_map[mask] = -lambda1[mask]
    else:
        print("PCBR: No pixels met the lambda threshold condition.")
        return np.zeros_like(img, dtype=float)

    coordinates = peak_local_max(response_map,
                                  min_distance=min_distance,
                                  threshold_rel=response_thresh_rel,
                                  exclude_border=False)
    if coordinates.shape[0] == 0:
        print("      PCBR: No local maxima found above threshold.")
        return np.zeros_like(img, dtype=float)

    rows, cols = coordinates[:, 0], coordinates[:, 1]
    scores = response_map[rows, cols]

    # Calculate integrated intensities (use_sigma_factor=False for PCBR)
    intensities = calculate_integrated_intensities(
        img, rows, cols, scores,
        use_sigma_factor=kwargs.get('use_sigma_factor', False),
        sigma_factor=kwargs.get('sigma_factor', 3),
        fixed_radius=kwargs.get('fixed_radius', 5)
    )

    return dirac_delta_image(img, rows, cols, intensities)


def calculate_integrated_intensities(img, peak_rows, peak_cols, peak_scores,
                                     use_sigma_factor=True,
                                     sigma_factor=3, fixed_radius=5):
    '''Calculate Integrated Intensities via Aperture Sum.
    For each surviving detection, intensity is integrated from
    background-subtracted image in a circular aperture:
        - LoG/DoH: radius ≈ `sigma_factor * sigma_score`
        - Other detectors: fixed `fixed_radius`
    Intensities are summed inside the aperture mask (local ROI for efficiency).

    Parameters
    ----------
    img : Numpy 2D array
        Input image.
    peak_rows : Numpy 1D array
        X coordinates of peaks.
    peak_cols : Numpy 1D array
        Y coordinates of peaks.
    peak_scores : Numpy 1D array
        Peak scores.
    use_sigma_factor : bool, optional
        Controls if sigma factor or fixed radius are used, by default True.
        Sigma factor should be used for LoG/DoH.
    sigma_factor : float, optional
        Aperture radius multiplier (sigma-based radius), by default 3.
    fixed_radius : int, optional
        Fixed aperture radius, by default 5.

    Returns
    -------
    Numpy 1D array
        Integrated intensities.
    '''

    integrated_intensities = np.zeros(len(peak_rows), dtype=float)
    img_h, img_w = img.shape

    if len(peak_rows) > 0:
        # Pre-calculate meshgrid for faster distance calculations
        # (optional optimization)
        # yy, xx = np.mgrid[:img_h, :img_w] # Could be large

        for i in range(len(peak_rows)):
            r_float, c_float = peak_rows[i], peak_cols[i]
            score = peak_scores[i]

            # Determine integration radius
            radius = 0.0
            if use_sigma_factor:
                # Use sigma score (ensure score is positive and reasonable)
                # Use max(0.5, ...) as sigma=0 is invalid for radius calc
                radius = sigma_factor * max(0.5, score)
            else:
                # Use fixed radius for other methods
                radius = float(fixed_radius)

            radius = max(1.0, radius)  # Ensure radius is at least 1 pixel
            radius_sq = radius ** 2  # Use squared radius for distance check

            # Define a bounding box around the peak for efficiency
            # Use ceil for max bounds to ensure aperture fits
            radius_ceil = int(np.ceil(radius))
            r_center_int = int(round(r_float)) # Use rounded center for slicing
            c_center_int = int(round(c_float))

            r_min = max(0, r_center_int - radius_ceil)
            r_max = min(img_h, r_center_int + radius_ceil + 1)
            c_min = max(0, c_center_int - radius_ceil)
            c_max = min(img_w, c_center_int + radius_ceil + 1)

            # Check if bounding box is valid
            if r_min >= r_max or c_min >= c_max:
                # Cannot calculate sum if box is empty
                integrated_intensities[i] = 0.0
                continue

            # Extract patch coordinates and data
            patch_rr, patch_cc = np.mgrid[r_min:r_max, c_min:c_max]
            patch_data = img[r_min:r_max, c_min:c_max]  # Indexing with slices

            # Calculate distance squared from the *float* center to each 
            # pixel *center* in the patch
            # Add 0.5 to patch indices to represent pixel centers for more 
            # accuracy? Optional.
            dist_sq = (patch_rr - r_float) ** 2 + (patch_cc - c_float) ** 2

            # Create mask for pixels within the circular aperture
            mask = dist_sq <= radius_sq

            # Sum intensities within the mask
            # Apply mask to patch_data before summing
            intensity_sum = np.sum(patch_data[mask])
            integrated_intensities[i] = intensity_sum

    return integrated_intensities

def dirac_delta_image(img, peak_rows, peak_cols, peak_intensities):
    '''Dirac-delta image construction
       Peaks are stored as `(row, col, integrated_intensity, score)`, 
       sorted by intensity.
       Output image is zeros; 
       each peak contributes intensity to one rounded pixel.
       Overlaps are accumulated using `np.add.at`.

    Parameters
    ----------
    img : Numpy 2D array
        Input image.
    peak_rows : Numpy 1D array
        X coordinates of peaks.
    peak_cols : Numpy 1D array
        Y coordinates of peaks.
    peak_intensities : Numpy 1D array
        Intensities of peaks

    Returns
    -------
    Numpy 2D array
        2D float Dirac image (same shape as input):
        each detected peak is a single non-zero pixel equal to 
        integrated intensity.
    '''

    # Store results (with float coords), sorted by intensity
    if len(peak_rows) > 0:
        peaks_data = sorted(list(zip(peak_rows, peak_cols, peak_intensities)),
                                    key=lambda p: p[2], reverse=True)
    else:
        peaks_data = []

    output_dirac = np.zeros_like(img, dtype=float)
    if peaks_data:
        peak_rows_float = np.array([p[0] for p in peaks_data])
        peak_cols_float = np.array([p[1] for p in peaks_data])
        peak_intensities_all = np.array([p[2] for p in peaks_data])

        rows_idx = np.round(peak_rows_float).astype(int)
        cols_idx = np.round(peak_cols_float).astype(int)
        rows_idx = np.clip(rows_idx, 0, output_dirac.shape[0] - 1)
        cols_idx = np.clip(cols_idx, 0, output_dirac.shape[1] - 1)

        # Sum intensities at overlapping pixels
        np.add.at(output_dirac, (rows_idx, cols_idx), peak_intensities_all)

    return output_dirac

def run_regions(image, threshold=1):
    '''
    Detects peaks by finding connected region with values equal to or higher
    than `threshold`.

    Parameters
    ----------
    image : Numpy 2D array
        Input image.
    threshold : int, optional, default is 1
        Values less than `threshold` will be ignored.

    Returns
    -------
    Numpy 2D array
        2D array with every peak replaced by a single pixel at its centroid,
        intensity equals to sum of peaks intensities.
    '''
    
    binary_image = (image >= threshold).astype(np.uint8)

    labeled_image = label(binary_image)
    new_image = np.zeros_like(image, dtype=np.float32)
    regions = regionprops(labeled_image, intensity_image=image)
    for region in regions:
        if region.area >= 1:
            coords = region.coords
            sum_intensity = image[coords[:, 0], coords[:, 1]].sum()
            centroid = region.centroid_weighted
            new_image[round(centroid[0]), round(centroid[1])] = sum_intensity

    arr = new_image

    return arr