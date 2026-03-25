'''
Module: idiff.peaks
-------------------

Find diffraction peaks in 2D-arrays/images/diffractograms.
'''

# Functions from 


from scipy import stats, spatial
from skimage.feature import (
    blob_log, blob_doh, peak_local_max, hessian_matrix_eigvals, hessian_matrix)
import cv2
from collections import defaultdict
from skimage.transform import resize
import numpy as np


def _estimate_noise(img):
    median_val = np.nanmedian(img)
    if np.isnan(median_val):
        print("\nWarning: Median is NaN.")
        return 1.0
    
    mad = stats.median_abs_deviation(img, axis=None, scale='normal', 
                                     nan_policy='omit')

    if np.isnan(mad) or mad == 0:
        print("\nWarning: MAD is zero/NaN.")
        noise_sigma = 1.0
    else:
        noise_sigma = mad

    return noise_sigma

def _run_log(img, downsample_factor=4, threshold_factor=3.5, peak_min_sigma=6, 
             peak_max_sigma=60, num_sigma_steps=30):
    """Runs LoG detector, with optional downsampling."""
    original_shape = img.shape
    noise_sigma = _estimate_noise(img)
    abs_threshold = noise_sigma * threshold_factor
    abs_threshold = max(abs_threshold, 1e-6) # Ensure positive

    if downsample_factor > 1:
        new_shape = (original_shape[0] // downsample_factor, 
                     original_shape[1] // downsample_factor)
        
        if new_shape[0] < 1 or new_shape[1] < 1:
            print("Warning: Downsampled image too small. Skipping.")
            return np.array([]), np.array([]), np.array([])
        
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
            return np.array([]), np.array([]), np.array([])
        
        blobs_small = blob_log(img_small, 
                               min_sigma=min_sigma_small, 
                               max_sigma=max_sigma_small,
                               num_sigma=num_sigma_steps, 
                               threshold=abs_threshold)

        if blobs_small.shape[0] == 0: 
            return np.array([]), np.array([]), np.array([])
        # Scale results back to original coordinates/scale
        rows_orig = blobs_small[:, 0] * downsample_factor
        cols_orig = blobs_small[:, 1] * downsample_factor
        scores_orig = blobs_small[:, 2] * downsample_factor # Scale sigma score
        return rows_orig, cols_orig, scores_orig

    else:
        blobs = blob_log(img, min_sigma=peak_min_sigma,
                         max_sigma=peak_max_sigma,
                         num_sigma=num_sigma_steps, 
                         threshold=abs_threshold)
        
        if blobs.shape[0] == 0: return np.array([]), np.array([]), np.array([])
        return blobs[:, 0], blobs[:, 1], blobs[:, 2]
    
def _run_doh(img, downsample_factor=4, threshold_factor=3.5, peak_min_sigma=6, 
             peak_max_sigma=60, num_sigma_steps=30, log_scale=False):
    """Runs DoH detector, with optional downsampling."""
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
            return np.array([]), np.array([]), np.array([])
    
        img_small = resize(img, new_shape, anti_aliasing=True, order=1, 
                           preserve_range=True)

        min_sigma_small = max(0.5, peak_min_sigma / downsample_factor)
        max_sigma_small = max(min_sigma_small + 0.5, 
                              peak_max_sigma / downsample_factor)
        
        if min_sigma_small >= max_sigma_small: 
            print("Warning: Invalid sigma range after downscaling",
                  f"({min_sigma_small:.2f}-{max_sigma_small:.2f}).",
                  "Skipping DoH.")
            return np.array([]), np.array([]), np.array([])

        print(f"DoH (downsampled) params: min_sig={min_sigma_small:.2f},",
              f"max_sig={max_sigma_small:.2f}, thresh={abs_threshold:.3f}",
              "(NEEDS TUNING)")
        
        blobs_small = blob_doh(img_small, min_sigma=min_sigma_small,
                               max_sigma=max_sigma_small,
                               num_sigma=num_sigma_steps, 
                               threshold=abs_threshold, 
                               log_scale=log_scale)

        if blobs_small.shape[0] == 0: 
            return np.array([]), np.array([]), np.array([])
        rows_orig = blobs_small[:, 0] * downsample_factor
        cols_orig = blobs_small[:, 1] * downsample_factor
        scores_orig = blobs_small[:, 2] * downsample_factor
        return rows_orig, cols_orig, scores_orig

    else:
        print(f"DoH params: min_sig={peak_min_sigma},",
              f"max_sig={peak_max_sigma},",
              f"thresh={abs_threshold:.3f} (NEEDS TUNING)")
        blobs = blob_doh(img, min_sigma=peak_min_sigma, 
                         max_sigma=peak_max_sigma,
                         num_sigma=num_sigma_steps, 
                         threshold=abs_threshold, 
                         log_scale=log_scale)
        
        if blobs.shape[0] == 0: 
            return np.array([]), np.array([]), np.array([])
        return blobs[:, 0], blobs[:, 1], blobs[:, 2]
    
def _run_mser(img, delta=5, min_area=60, max_area=14400):

    if 'cv2' not in globals(): 
        raise ImportError("MSER requires OpenCV (cv2)")

    min_val, max_val = np.min(img), np.max(img)
    if max_val > min_val:
        img_uint8 = (img - min_val) / (max_val - min_val) * 255
        img_uint8 = img_uint8.astype(np.uint8) 
    else:
        img_uint8 = np.zeros_like(img, dtype=np.uint8)

    mser = cv2.MSER_create(_delta=delta, 
                           _min_area=min_area, 
                           _max_area=max_area)
    
    regions, bboxes = mser.detectRegions(img_uint8)
    if not regions:
        return np.array([]), np.array([]), np.array([])
    rows, cols, scores = [], [], []
    for i, pts in enumerate(regions):
        try:
            moments = cv2.moments(pts)
            if moments["m00"] != 0:
                rows.append(int(moments["m01"] / moments["m00"]))
                cols.append(int(moments["m10"] / moments["m00"]))
                scores.append(cv2.contourArea(pts))
        except Exception:
            pass
        return np.array(rows), np.array(cols), np.array(scores)
    
def _run_pcbr(img, sigma=3.0, lambda_thresh=0.5, response_thresh_rel=0.1, 
              min_distance=5):
    try:
        Hrr, Hrc, Hcc = hessian_matrix(img,
            sigma=sigma,
            use_gaussian_derivatives=True,  # Default is True, explicit here
            mode='nearest',  # Specify boundary handling
            order='rc')  # Specify order if needed (rc is default)
    except Exception as e:
        print(f"      ERROR calculating Hessian matrix: {e}. Skipping PCBR.")
        return np.array([]), np.array([]), np.array([])

    try:
        if not np.all(np.isfinite(Hrr)) or \
            not np.all(np.isfinite(Hrc)) or \
            not np.all(np.isfinite(Hcc)):
            print("Warning: Non-finite values found in Hessian components. " \
                "Attempting nan_to_num.")
            Hrr = np.nan_to_num(Hrr)
            Hrc = np.nan_to_num(Hrc)
            Hcc = np.nan_to_num(Hcc)
        lambda1, lambda2 = hessian_matrix_eigvals(Hrr, Hrc, Hcc)
    except Exception as e:
        print(f"ERROR calculating Hessian eigenvalues: {e}. Skipping PCBR.")
        return np.array([]), np.array([]), np.array([])

    response_map = np.zeros_like(img)
    valid_lambda = np.isfinite(lambda1)
    mask = valid_lambda & (lambda1 < -lambda_thresh)

    if np.any(mask):
        response_map[mask] = -lambda1[mask]
    else:
        print("PCBR: No pixels met the lambda threshold condition.")


    if np.any(response_map > 0):
        print(f"    Finding peaks in PCBR response map...")
        coordinates = peak_local_max(response_map,
                                        min_distance=min_distance,
                                        threshold_rel=response_thresh_rel,
                                        exclude_border=False)
        if coordinates.shape[0] == 0:
            print("      PCBR: No local maxima found above threshold.")
            return np.array([]), np.array([]), np.array([])

        rows, cols = coordinates[:, 0], coordinates[:, 1]
        scores = response_map[rows, cols]
        return rows, cols, scores
    else:
        print("      PCBR: Response map is empty or zero. No peaks found.")
        return np.array([]), np.array([]), np.array([])
    

# default uses all algs with default parameters
def _run_vote(img, min_votes=2, vote_radius=5.0, methods={}):
    all_detections = []
    if not methods:
        methods = {_run_log: {}, _run_doh: {}, _run_mser: {},_run_pcbr: {}}

    for alg, args in methods.items():
        try:
            rows, cols, scores = alg(img, **args)
            for r, c in zip(rows, cols):
                all_detections.append((r, c, str(alg)))
        except ImportError as e: 
            print(f"Skipping detector {str(alg)} due to missing library: {e}")
        except Exception as e: 
            print(f"ERROR running detector {str(alg)}: {e}")

    if not all_detections:
        print("Voting: No detections found by any contributing method.")
        return np.array([]), np.array([]), np.array([])
    
    all_points = np.array([(d[0], d[1]) for d in all_detections])
    detector_names = [d[2] for d in all_detections]
    if all_points.shape[0] < min_votes:
        print(f"Voting: Insufficient points ({all_points.shape[0]})",
              f"for min_votes ({min_votes}).")
        return np.array([]), np.array([]), np.array([])
    
    kdtree = spatial.KDTree(all_points)
    pairs = kdtree.query_pairs(r=vote_radius)
    adjacency_list = defaultdict(list)
    for i, j in pairs:
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)

    num_points = len(all_points)
    visited = np.zeros(num_points, dtype=bool)
    consensus_peaks = []
    for i in range(num_points):
        if not visited[i]:
            component_indices = []
            q = [i]
            visited[i] = True
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                component_indices.append(u)
                [q.append(v) for v in adjacency_list[u]
                 if not visited[v] and not visited.__setitem__(v, True)]
            component_detectors = \
                set(detector_names[idx] for idx in component_indices)
            num_votes = len(component_detectors)
            if num_votes >= min_votes:
                component_points = all_points[component_indices]
                avg_row, avg_col = np.mean(component_points, axis=0)
                consensus_peaks.append((avg_row, avg_col, num_votes))
    
    if not consensus_peaks:
        return np.array([]), np.array([]), np.array([])
    peak_rows = np.array([p[0] for p in consensus_peaks])
    peak_cols = np.array([p[1] for p in consensus_peaks])
    peak_scores = np.array([p[2] for p in consensus_peaks])
    return peak_rows, peak_cols, peak_scores
    
def filter_central_region(img, peak_rows, peak_cols, peak_scores, 
                          ignore_radius_pixels):
    if len(peak_rows) > 0:
        center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
        center_dist_sq = (peak_rows - center_y) ** 2 + \
                         (peak_cols - center_x) ** 2
        center_ignore_radius_sq = ignore_radius_pixels ** 2
        keep_indices = center_dist_sq > center_ignore_radius_sq

        return (peak_rows[keep_indices], 
                peak_cols[keep_indices], 
                peak_scores[keep_indices])


# Calculate Integrated Intensities via Aperture Sum
# TODO: mention sigma score is for log and doh methods
def calculate_integrated_intensities(img, peak_rows, peak_cols, peak_scores,
                                     use_sigma_factor=True,
                                     sigma_factor=3, fixed_radius=5):
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

def detect_peaks(img, alg, **kwargs):
    # run detector
    rows, cols, scores = alg(img, **kwargs)
    rows, cols, scores = filter_central_region(img, rows, cols, scores)

    # Calculate Integrated Intensities via Aperture Sum ---
    integrated_intensities = calculate_integrated_intensities(img, rows, cols, 
                                                              scores)
    return rows, cols, integrated_intensities, scores

