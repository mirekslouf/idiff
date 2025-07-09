import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.transform import Rotation
import random
from skimage.draw import disk
import cv2


class STEMDiffractionGenerator:
    """
    A class for generating physically accurate STEM diffraction patterns.
    """

    def __init__(self,
                 size=256,
                 crystal_structure='fcc',
                 lattice_params=None,
                 elements=None,
                 atomic_positions=None,
                 voltage=200,  # kV
                 camera_length=300,  # mm
                 convergence_angle=1.5,  # mrad
                 zone_axis=None,
                 deviation_parameter=0.1,  # 1/nm
                 noise_level=0.4,
                 background_variation=0.01,
                 detector_defects=True,
                 sample_thickness=50,  # nm
                 temperature_factor=0.8,  # Debye-Waller factor
                 direct_beam_intensity=0.7,
                 spot_intensity_factor=0.4,
                 kikuchi_intensity_factor=0.03,
                 holz_intensity=0.02,
                 spot_size_factor=1.2):

        # Set default values for mutable parameters
        if lattice_params is None:
            lattice_params = {'a': 3.615, 'b': 3.615, 'c': 3.615, 'alpha': 90, 'beta': 90, 'gamma': 90}

        if elements is None:
            elements = ['Al']

        if atomic_positions is None:
            atomic_positions = [{'element': 'Al', 'position': [0, 0, 0]}]

        if zone_axis is None:
            zone_axis = [0, 0, 1]
        """
        Initialize the STEM diffraction pattern generator with physical parameters.

        Parameters:
        -----------
        size : int
            Size of the output image (size x size pixels)
        crystal_structure : str
            Type of crystal structure ('fcc', 'bcc', 'hcp', 'diamond', 'rocksalt', etc.)
        lattice_params : dict
            Dictionary containing lattice parameters (a,b,c in Angstroms and alpha,beta,gamma in degrees)
        elements : list
            List of elements in the crystal
        atomic_positions : list of dict
            List of dictionaries containing element and position for each atom in the unit cell
        voltage : float
            Electron beam voltage in kV
        camera_length : float
            Camera length in mm
        convergence_angle : float
            Beam convergence semi-angle in mrad
        zone_axis : list
            Zone axis direction [h,k,l]
        deviation_parameter : float
            Excitation error (deviation parameter) in 1/nm
        noise_level : float
            Level of noise to add to the image
        background_variation : float
            Level of non-uniform background variation
        detector_defects : bool
            Whether to add detector defects (e.g., dead pixels)
        sample_thickness : float
            Sample thickness in nm
        temperature_factor : float
            Debye-Waller factor (0-1) to account for thermal vibrations
        direct_beam_intensity : float
            Intensity factor for the direct beam (0-1)
        spot_intensity_factor : float
            Scaling factor for diffraction spot intensities (0-1)
        kikuchi_intensity_factor : float
            Intensity factor for Kikuchi lines (0-1)
        holz_intensity : float
            Intensity factor for HOLZ rings (0-1)
        spot_size_factor : float
            Controls the size/sharpness of diffraction spots
        """
        # Store all parameters as instance variables
        self.size = size
        self.crystal_structure = crystal_structure
        self.lattice_params = lattice_params
        self.elements = elements
        self.atomic_positions = atomic_positions
        self.voltage = voltage
        self.camera_length = camera_length
        self.convergence_angle = convergence_angle
        self.zone_axis = zone_axis
        self.deviation_parameter = deviation_parameter
        self.noise_level = noise_level
        self.background_variation = background_variation
        self.detector_defects = detector_defects
        self.sample_thickness = sample_thickness
        self.temperature_factor = temperature_factor
        self.direct_beam_intensity = direct_beam_intensity
        self.spot_intensity_factor = spot_intensity_factor
        self.kikuchi_intensity_factor = kikuchi_intensity_factor
        self.holz_intensity = holz_intensity
        self.spot_size_factor = spot_size_factor

        # Initialize derived parameters
        self._init_derived_parameters()

        # Initialize scattering factors table
        self._init_scattering_factors()

    def _init_derived_parameters(self):
        """Initialize parameters derived from the basic inputs."""
        # Physical constants
        self.m0 = 9.1094e-31  # electron rest mass in kg
        self.e = 1.6022e-19  # elementary charge in coulombs
        self.h = 6.6261e-34  # Planck's constant in J·s
        self.c = 2.9979e8  # speed of light in m/s

        # Calculate electron wavelength
        V = self.voltage * 1000  # Convert kV to V
        self.wavelength = (self.h / np.sqrt(2 * self.m0 * self.e * V *
                                            (1 + self.e * V / (2 * self.m0 * self.c * self.c)))) * 1e10  # in Angstroms

        # Calculate reciprocal lattice parameters
        a, b, c = self.lattice_params['a'], self.lattice_params['b'], self.lattice_params['c']
        alpha = np.radians(self.lattice_params['alpha'])
        beta = np.radians(self.lattice_params['beta'])
        gamma = np.radians(self.lattice_params['gamma'])

        # Calculate volume of the unit cell
        self.volume = a * b * c * np.sqrt(
            1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))

        # Calculate reciprocal lattice vectors
        self.a_recip = 2 * np.pi * b * c * np.sin(alpha) / self.volume
        self.b_recip = 2 * np.pi * a * c * np.sin(beta) / self.volume
        self.c_recip = 2 * np.pi * a * b * np.sin(gamma) / self.volume

        self.alpha_recip = np.arccos((np.cos(beta) * np.cos(gamma) - np.cos(alpha)) /
                                     (np.sin(beta) * np.sin(gamma)))
        self.beta_recip = np.arccos((np.cos(alpha) * np.cos(gamma) - np.cos(beta)) /
                                    (np.sin(alpha) * np.sin(gamma)))
        self.gamma_recip = np.arccos((np.cos(alpha) * np.cos(beta) - np.cos(gamma)) /
                                     (np.sin(alpha) * np.sin(beta)))

        # Create coordinate system for the image
        x, y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        self.x = x
        self.y = y
        self.center_x = self.size // 2
        self.center_y = self.size // 2

    def _init_scattering_factors(self):
        """Initialize the atomic scattering factors table."""
        # Values at sin(θ)/λ = 0 for 200 kV electrons - simplified lookup
        self.scattering_factors = {
            'H': 1.0, 'He': 1.5, 'Li': 2.5, 'Be': 3.5, 'B': 4.0, 'C': 4.8,
            'N': 5.7, 'O': 6.6, 'F': 7.5, 'Ne': 8.5, 'Na': 9.5, 'Mg': 10.5,
            'Al': 11.5, 'Si': 12.5, 'P': 13.5, 'S': 14.5, 'Cl': 15.5, 'Ar': 16.5,
            'K': 17.5, 'Ca': 18.5, 'Ti': 20.5, 'V': 21.5, 'Cr': 22.5, 'Mn': 23.5,
            'Fe': 24.5, 'Co': 25.5, 'Ni': 26.5, 'Cu': 27.5, 'Zn': 28.5, 'Ga': 29.5,
            'Ge': 30.5, 'As': 31.5, 'Se': 32.5, 'Br': 33.5, 'Kr': 34.5,
            'Au': 79.0
        }

    def _is_allowed_reflection(self, h, k, l):
        """
        Determine if reflection is allowed based on structure factor rules.

        Parameters:
        -----------
        h, k, l : int
            Miller indices of the reflection

        Returns:
        --------
        bool
            True if the reflection is allowed, False otherwise
        """
        structure = self.crystal_structure.lower()

        if structure == 'fcc':
            # For FCC, h, k, l must be all even or all odd
            return h % 2 == k % 2 == l % 2
        elif structure == 'bcc':
            # For BCC, h + k + l must be even
            return (h + k + l) % 2 == 0
        elif structure == 'sc':
            # For simple cubic, all reflections are allowed
            return True
        elif structure == 'diamond':
            # For diamond, h,k,l all even and h+k+l=4n, or h,k,l all odd
            if h % 2 == k % 2 == l % 2 == 0:
                return (h + k + l) % 4 == 0
            else:
                return h % 2 == k % 2 == l % 2 == 1
        elif structure == 'hcp':
            # For HCP, l even: all h,k allowed; l odd: h+2k≠3n
            if l % 2 == 0:
                return True
            else:
                return (h + 2 * k) % 3 != 0
        elif structure == 'rocksalt':
            # For rocksalt (NaCl), h,k,l all odd or all even, but not mixed
            # and h+k+l≠odd
            same_parity = (h % 2 == k % 2 == l % 2)
            return same_parity and (h + k + l) % 2 == 0
        elif structure == 'perovskite':
            # For perovskite (ABO3), different conditions for different peaks
            # Simplified rule: all three indices must be all odd or all even
            return (h % 2 == k % 2 == l % 2)
        elif structure == 'zincblende':
            # For zincblende, all indices odd or all even
            # If all even, then h+k+l=4n or if all odd, then h+k+l=odd
            if h % 2 == k % 2 == l % 2 == 0:
                return (h + k + l) % 4 == 0
            elif h % 2 == k % 2 == l % 2 == 1:
                return (h + k + l) % 2 == 1
            else:
                return False
        else:
            # Default: allow all reflections
            return True

    def _line(self, y0, x0, y1, x1):
        """
        Bresenham's line algorithm for drawing Kikuchi lines.

        Parameters:
        -----------
        y0, x0 : int
            Starting coordinates
        y1, x1 : int
            Ending coordinates

        Returns:
        --------
        numpy array
            Arrays of y and x coordinates for the line
        """
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        line_points_x = []
        line_points_y = []

        x, y = x0, y0
        while True:
            line_points_x.append(x)
            line_points_y.append(y)

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 >= dy:
                if x == x1:
                    break
                err += dy
                x += sx
            if e2 <= dx:
                if y == y1:
                    break
                err += dx
                y += sy

        return np.array(line_points_y), np.array(line_points_x)

    def _generate_reflections(self, max_index=12):
        """
        Generate all possible reflections within max_index.

        Parameters:
        -----------
        max_index : int
            Maximum Miller index to consider

        Returns:
        --------
        list
            List of allowed reflections with their properties
        """
        reflections = []

        # Convert zone axis to unit vector
        zone_norm = np.sqrt(sum(z ** 2 for z in self.zone_axis))
        uz = [z / zone_norm for z in self.zone_axis]

        # Generate all possible reflections within max_index
        for h in range(-max_index, max_index + 1):
            for k in range(-max_index, max_index + 1):
                for l in range(-max_index, max_index + 1):
                    # Skip the origin
                    if h == 0 and k == 0 and l == 0:
                        continue

                    # Check if reflection is allowed based on systematic absences
                    if self._is_allowed_reflection(h, k, l):
                        # Skip reflections with effectively zero magnitude
                        hkl_mag = np.sqrt(h ** 2 + k ** 2 + l ** 2)
                        if hkl_mag < 1e-10:
                            continue

                        # Calculate excitation error (deviation from exact Bragg condition)
                        s = self.deviation_parameter * (abs(h) + abs(k) + abs(l)) / hkl_mag

                        # Calculate diffraction vector magnitude (1/d where d is d-spacing)
                        g_magnitude = np.sqrt((h * self.a_recip) ** 2 +
                                              (k * self.b_recip) ** 2 +
                                              (l * self.c_recip) ** 2)

                        # Calculate structure factor
                        structure_factor = 0
                        for atom in self.atomic_positions:
                            if atom['element'] in self.scattering_factors:
                                f = self.scattering_factors[atom['element']]
                                px, py, pz = atom['position']
                                phase = 2 * np.pi * (h * px + k * py + l * pz)
                                structure_factor += f * np.exp(1j * phase)

                        # Calculate intensity - kinematical approximation with safeguards
                        sf_abs = abs(structure_factor)
                        epsilon = 1e-10  # Small value to prevent division by zero

                        # Calculate extinction distance safely
                        denom = self.wavelength * g_magnitude * sf_abs
                        if denom < epsilon:
                            # Skip reflections that would cause division by zero
                            continue

                        extinction_distance = 1.0 / denom

                        # Calculate sin term safely
                        sin_term = np.sin(np.pi * self.sample_thickness / extinction_distance)

                        # Protect against division by zero in s term
                        pi_s = np.pi * s
                        if abs(pi_s) < epsilon:
                            # Skip reflections with tiny s
                            continue

                        # Calculate intensity with all safeguards in place
                        intensity = (sf_abs ** 2) * (sin_term ** 2) / (pi_s ** 2)

                        # Apply Debye-Waller temperature factor
                        B = -np.log(self.temperature_factor) * 8 * np.pi ** 2  # Convert to B-factor
                        intensity *= np.exp(-B * (g_magnitude / (4 * np.pi)) ** 2)

                        # Scale down intensity
                        intensity = np.abs(intensity) / 100 * self.spot_intensity_factor

                        # Dot product between reflection and zone axis
                        dot_product = h * uz[0] + k * uz[1] + l * uz[2]

                        if abs(dot_product) < 0.1:  # Tolerance for reflections to appear
                            reflections.append({
                                'h': h, 'k': k, 'l': l,
                                'intensity': float(intensity),
                                'g_magnitude': float(g_magnitude)
                            })

        # Sort and normalize reflections
        reflections = sorted(reflections, key=lambda x: x['intensity'], reverse=True)

        # Group reflections by similar d-spacing and normalize intensities
        if reflections:
            max_intensity = reflections[0]['intensity']

            # Group by d-spacing
            grouped_by_d = {}
            for ref in reflections:
                d_key = round(ref['g_magnitude'] * 10) / 10
                if d_key not in grouped_by_d:
                    grouped_by_d[d_key] = []
                grouped_by_d[d_key].append(ref)

            # Normalize within groups
            for d_key, group in grouped_by_d.items():
                group_max = max(ref['intensity'] for ref in group)
                if group_max > 0:
                    for ref in group:
                        ref['intensity'] = 0.3 * (ref['intensity'] / group_max) * max_intensity

        return reflections

    def generate(self, generate_mask=False, mask_type="binary", intensity_threshold=0.01, return_clean=False):
        """
        Generate a STEM diffraction pattern based on the initialized parameters.
        Optionally generate a mask of diffraction peak positions and/or return a clean pattern without noise and background.

        Parameters:
        -----------
        generate_mask : bool
            If True, generates a mask identifying diffraction peaks
        mask_type : str
            Type of mask to generate:
            - "binary": Binary mask with 1s at peak locations and 0s elsewhere
            - "intensity": Continuous mask preserving the relative intensity of peaks
            - "adaptive": Mask with peaks sized proportionally to their intensity
        intensity_threshold : float
            Minimum intensity threshold for including a spot in the mask (0-1)
        return_clean : bool
            If True, returns a clean version of the diffraction pattern without noise and background

        Returns:
        --------
        If generate_mask=False and return_clean=False:
            numpy.ndarray
                2D array representing the diffraction pattern

        If generate_mask=True and return_clean=False:
            tuple
                (diffraction_pattern, mask)
                - diffraction_pattern: 2D array of the diffraction pattern
                - mask: 2D array identifying diffraction peaks

        If generate_mask=False and return_clean=True:
            tuple
                (diffraction_pattern, clean_pattern)
                - diffraction_pattern: 2D array of the diffraction pattern with noise and background
                - clean_pattern: 2D array of the diffraction pattern without noise and background

        If generate_mask=True and return_clean=True:
            tuple
                (diffraction_pattern, mask, clean_pattern)
                - diffraction_pattern: 2D array of the diffraction pattern with noise and background
                - mask: 2D array identifying diffraction peaks
                - clean_pattern: 2D array of the diffraction pattern without noise and background
        """
        # Create empty array for the diffraction pattern
        diffraction_pattern = np.zeros((self.size, self.size))

        # If we want a clean pattern, we'll create a separate array for it
        if return_clean:
            clean_pattern = np.zeros((self.size, self.size))

        # Create a completely separate mask from scratch
        reflection_positions = []

        # Add direct beam position
        reflection_positions.append((self.center_x, self.center_y))

        # Generate allowed reflections to get their positions
        reflections = self._generate_reflections(max_index=6)

        # Filter to include only significant reflections
        visible_reflections = []
        for reflection in reflections:
            if reflection['intensity'] > intensity_threshold:
                visible_reflections.append(reflection)

        # Limit reflections for clarity
        max_reflections = 300
        visible_reflections = visible_reflections[:max_reflections]

        # Find maximum intensity for normalization
        max_intensity = 0
        for ref in visible_reflections:
            if ref['intensity'] > max_intensity:
                max_intensity = ref['intensity']

        # Calculate scale factor for pixel coordinates
        scale_factor = self.size / (4 * self.wavelength * self.camera_length)

        # Calculate unit vectors perpendicular to zone axis

        if abs(self.zone_axis[2]) < 0.9:
            u1 = np.cross(self.zone_axis, [0, 0, 1])
        else:
            u1 = np.cross(self.zone_axis, [1, 0, 0])
        u1 = u1 / np.linalg.norm(u1)

        u2 = np.cross(self.zone_axis, u1)
        u2 = u2 / np.linalg.norm(u2)

        # Get positions of all reflections
        for reflection in visible_reflections:
            h, k, l = reflection['h'], reflection['k'], reflection['l']

            # Project g-vector onto diffraction plane
            g_vector = [h * self.a_recip, k * self.b_recip, l * self.c_recip]
            proj_u1 = np.dot(g_vector, u1)
            proj_u2 = np.dot(g_vector, u2)

            # Convert to pixel coordinates
            spot_x = self.center_x + proj_u1 * scale_factor
            spot_y = self.center_y + proj_u2 * scale_factor

            # Only add if within image boundaries
            if 0 <= spot_x < self.size and 0 <= spot_y < self.size:
                reflection_positions.append((spot_x, spot_y))

        # Create direct beam with realistic shape
        beam_radius = self.size * self.convergence_angle / (25 * self.wavelength)
        direct_beam = np.exp(-((self.x - self.center_x) ** 2 + (self.y - self.center_y) ** 2) /
                             (2 * (beam_radius) ** 2))
        direct_beam = direct_beam / np.max(direct_beam) * self.direct_beam_intensity

        # Add direct beam to diffraction pattern
        diffraction_pattern += direct_beam
        if return_clean:
            clean_pattern += direct_beam

        # Add diffraction spots
        spot_size = self.size / 200 * self.spot_size_factor

        for reflection in reflections:
            h, k, l = reflection['h'], reflection['k'], reflection['l']

            # Project g-vector onto diffraction plane
            g_vector = [h * self.a_recip, k * self.b_recip, l * self.c_recip]
            proj_u1 = np.dot(g_vector, u1)
            proj_u2 = np.dot(g_vector, u2)

            # Convert to pixel coordinates
            spot_x = self.center_x + proj_u1 * scale_factor
            spot_y = self.center_y + proj_u2 * scale_factor

            # Check if spot is within boundaries
            if 0 <= spot_x < self.size and 0 <= spot_y < self.size:
                r = np.sqrt((self.x - spot_x) ** 2 + (self.y - spot_y) ** 2)

                # Create realistic spot profile
                spot_core = np.exp(-r ** 2 / (2 * (spot_size * 0.5) ** 2))
                spot_tail = 1 / (1 + (r / spot_size) ** 4)
                spot = 0.8 * spot_core + 0.2 * spot_tail

                # Add spot to pattern
                diffraction_pattern += spot * reflection['intensity']
                if return_clean:
                    clean_pattern += spot * reflection['intensity']

        # Add Kikuchi lines
        for reflection in reflections:
            if np.random.random() < 0.05:
                h, k, l = reflection['h'], reflection['k'], reflection['l']
                g_magnitude = reflection['g_magnitude']

                # Calculate Bragg angle
                theta = np.arcsin(self.wavelength * g_magnitude / 2)

                # Project g-vector onto diffraction plane
                g_vector = [h * self.a_recip, k * self.b_recip, l * self.c_recip]
                proj_u1 = np.dot(g_vector, u1)
                proj_u2 = np.dot(g_vector, u2)

                # Normalize
                g_proj_norm = np.sqrt(proj_u1 ** 2 + proj_u2 ** 2)
                if g_proj_norm < 1e-6:
                    continue

                g_proj_u1 = proj_u1 / g_proj_norm
                g_proj_u2 = proj_u2 / g_proj_norm

                # Calculate shift due to Bragg angle
                shift = self.size * np.tan(2 * theta) / (2 * self.wavelength * self.camera_length)

                # Draw Kikuchi lines (excess and deficiency)
                for sign in [1, -1]:
                    # Vector perpendicular to g
                    perp_u1 = -g_proj_u2
                    perp_u2 = g_proj_u1

                    # Line endpoints
                    start_x = self.center_x - perp_u1 * self.size / 2
                    start_y = self.center_y - perp_u2 * self.size / 2
                    end_x = self.center_x + perp_u1 * self.size / 2
                    end_y = self.center_y + perp_u2 * self.size / 2

                    # Shift by Bragg angle
                    shift_x = sign * shift * g_proj_u1
                    shift_y = sign * shift * g_proj_u2

                    start_x += shift_x
                    start_y += shift_y
                    end_x += shift_x
                    end_y += shift_y

                    # Create line mask
                    line_mask = np.zeros((self.size, self.size))
                    rr, cc = self._line(int(start_y), int(start_x), int(end_y), int(end_x))
                    valid_idx = (rr >= 0) & (rr < self.size) & (cc >= 0) & (cc < self.size)

                    if np.any(valid_idx):
                        line_mask[rr[valid_idx], cc[valid_idx]] = 1
                        line_mask = ndimage.gaussian_filter(line_mask, sigma=self.size / 50)

                        intensity_factor = self.kikuchi_intensity_factor * reflection['intensity']
                        if sign > 0:  # Excess line
                            diffraction_pattern += line_mask * intensity_factor
                            if return_clean:
                                clean_pattern += line_mask * intensity_factor
                        else:  # Deficiency line
                            diffraction_pattern -= line_mask * intensity_factor
                            if return_clean:
                                clean_pattern -= line_mask * intensity_factor

        # Add HOLZ rings
        if np.random.random() < 0.2:
            HOLZ_radius = self.size * self.wavelength * np.sqrt(4 ** 2) / 4
            r = np.sqrt((self.x - self.center_x) ** 2 + (self.y - self.center_y) ** 2)
            HOLZ_ring = np.exp(-(r - HOLZ_radius) ** 2 / (2 * (self.size / 50) ** 2))
            diffraction_pattern += HOLZ_ring * self.holz_intensity
            if return_clean:
                clean_pattern += HOLZ_ring * self.holz_intensity

        # At this point, clean_pattern is complete if we're returning it
        if return_clean:
            # Apply radial intensity decay to clean pattern (this is physical, not noise)
            r = np.sqrt((self.x - self.center_x) ** 2 + (self.y - self.center_y) ** 2)
            radial_decay = np.exp(-r / (self.size / 3))
            clean_pattern *= radial_decay

            # Ensure clean pattern values are between 0 and 1
            clean_pattern = np.clip(clean_pattern, 0, 1)

            # Apply logarithmic intensity scaling to clean pattern
            clean_pattern = np.log1p(clean_pattern * 5) / np.log(6)

            # Final normalization for clean pattern
            if np.max(clean_pattern) > 0:
                clean_pattern = clean_pattern / np.max(clean_pattern)

        # Now add noise and background to the regular diffraction pattern

        # Add non-uniform background
        y_gradient, x_gradient = np.meshgrid(
            np.linspace(-1, 1, self.size),
            np.linspace(-1, 1, self.size)
        )
        background = self.background_variation * (
                1 + np.sin(x_gradient * np.pi * 1.5) * 0.5 +
                np.cos(y_gradient * np.pi * 2.3) * 0.5
        )

        background = ndimage.gaussian_filter(background, sigma=self.size / 10)
        diffraction_pattern += background

        # Add detector defects
        if self.detector_defects:
            # Dead pixels
            num_dead_pixels = int(self.size * self.size * 0.0005)
            for _ in range(num_dead_pixels):
                px, py = np.random.randint(0, self.size, 2)
                diffraction_pattern[py, px] = 0

            # Hot pixels
            num_hot_pixels = int(self.size * self.size * 0.0002)
            for _ in range(num_hot_pixels):
                px, py = np.random.randint(0, self.size, 2)
                diffraction_pattern[py, px] = 1

            # Clusters of bad pixels
            num_clusters = np.random.randint(1, 3)
            for _ in range(num_clusters):
                cx, cy = np.random.randint(0, self.size, 2)
                cluster_size = np.random.randint(2, 5)
                cluster_value = np.random.choice([0, 1])

                for i in range(-cluster_size // 2, cluster_size // 2 + 1):
                    for j in range(-cluster_size // 2, cluster_size // 2 + 1):
                        if 0 <= cx + i < self.size and 0 <= cy + j < self.size:
                            if np.random.random() < 0.6:
                                diffraction_pattern[cy + j, cx + i] = cluster_value

        # Add Poisson noise
        diffraction_pattern = np.maximum(0, diffraction_pattern)
        signal_level = np.mean(diffraction_pattern) * 500
        signal_level = max(signal_level, 1e-10)
        diffraction_pattern_counts = np.random.poisson(diffraction_pattern * signal_level) / signal_level

        # Add readout noise
        diffraction_pattern_counts += np.random.normal(0, self.noise_level, (self.size, self.size))

        # Apply detector response function
        epsilon = 1e-10
        diffraction_pattern = np.power(np.maximum(diffraction_pattern_counts, epsilon), 0.9)

        # Apply radial intensity decay
        r = np.sqrt((self.x - self.center_x) ** 2 + (self.y - self.center_y) ** 2)
        radial_decay = np.exp(-r / (self.size / 3))

        diffraction_pattern *= radial_decay

        # Ensure values are between 0 and 1
        diffraction_pattern = np.clip(diffraction_pattern, 0, 1)

        # Apply logarithmic intensity scaling
        diffraction_pattern = np.log1p(diffraction_pattern * 5) / np.log(6)

        # Final normalization
        if np.max(diffraction_pattern) > 0:
            diffraction_pattern = diffraction_pattern / np.max(diffraction_pattern)

        # Create mask only if requested
        if generate_mask:
            # Always initialize with zeros
            mask = np.zeros((self.size, self.size), dtype=np.float32)

            # For binary masks, simply draw circles at each reflection position

            if generate_mask:
                # Always initialize with zeros (uint8 for OpenCV)
                mask = np.zeros((self.size, self.size), dtype=np.uint8)

                if mask_type == "binary":
                    spot_mask_radius = int(spot_size * 2)  # Or tweak as needed

                    for spot_x, spot_y in reflection_positions:
                        # Round and convert spot positions to int for OpenCV
                        spot_x_int, spot_y_int = int(round(spot_x)), int(round(spot_y))

                        # Draw anti-aliased filled white circle
                        cv2.circle(mask, (spot_x_int, spot_y_int), spot_mask_radius,
                                   color=255, thickness=-1, lineType=cv2.LINE_AA)

                    # Convert to float32 binary mask (0.0 or 1.0)
                    mask = (mask > 127).astype(np.float32)
            elif mask_type == "intensity":
                # For intensity mask, we need to create spot profiles with intensities
                for i, reflection in enumerate(visible_reflections):
                    h, k, l = reflection['h'], reflection['k'], reflection['l']

                    # Project g-vector onto diffraction plane (same as in pattern generation)
                    g_vector = [h * self.a_recip, k * self.b_recip, l * self.c_recip]
                    proj_u1 = np.dot(g_vector, u1)
                    proj_u2 = np.dot(g_vector, u2)

                    # Convert to pixel coordinates
                    spot_x = self.center_x + proj_u1 * scale_factor
                    spot_y = self.center_y + proj_u2 * scale_factor

                    # Check if spot is within boundaries
                    if 0 <= spot_x < self.size and 0 <= spot_y < self.size:
                        r = np.sqrt((self.x - spot_x) ** 2 + (self.y - spot_y) ** 2)

                        # Create intensity-weighted spot profile
                        spot_profile = np.exp(-r ** 2 / (2 * (spot_size * 1.0) ** 2))
                        intensity_factor = reflection['intensity'] / self.direct_beam_intensity
                        mask += spot_profile * intensity_factor

                # Add direct beam to intensity mask
                db_profile = np.exp(-((self.x - self.center_x) ** 2 + (self.y - self.center_y) ** 2) /
                                    (2 * (beam_radius) ** 2))
                mask += db_profile

                # Normalize
                if np.max(mask) > 0:
                    mask /= np.max(mask)

            elif mask_type == "adaptive":
                # For adaptive masks, spot size scales with intensity
                for i, reflection in enumerate(visible_reflections):
                    h, k, l = reflection['h'], reflection['k'], reflection['l']

                    # Project g-vector onto diffraction plane
                    g_vector = [h * self.a_recip, k * self.b_recip, l * self.c_recip]
                    proj_u1 = np.dot(g_vector, u1)
                    proj_u2 = np.dot(g_vector, u2)

                    # Convert to pixel coordinates
                    spot_x = self.center_x + proj_u1 * scale_factor
                    spot_y = self.center_y + proj_u2 * scale_factor

                    # Check if spot is within boundaries
                    if 0 <= spot_x < self.size and 0 <= spot_y < self.size:
                        # Calculate relative intensity
                        rel_intensity = reflection['intensity'] / max_intensity if max_intensity > 0 else 0

                        # Scale spot size based on intensity (larger spots for stronger reflections)
                        adaptive_spot_size = spot_size * (0.5 + rel_intensity)

                        r = np.sqrt((self.x - spot_x) ** 2 + (self.y - spot_y) ** 2)
                        spot_profile = np.exp(-r ** 2 / (2 * (adaptive_spot_size) ** 2))

                        # Add to mask with intensity scaling
                        mask += spot_profile * min(1.0, rel_intensity * 3)  # Enhance contrast

                # Add direct beam
                db_profile = np.exp(-((self.x - self.center_x) ** 2 + (self.y - self.center_y) ** 2) /
                                    (2 * (beam_radius) ** 2))
                mask += db_profile

                # Normalize
                if np.max(mask) > 0:
                    mask /= np.max(mask)

            # Return appropriate tuple based on parameters
            if return_clean:
                return diffraction_pattern, mask, clean_pattern
            else:
                return diffraction_pattern, mask
        else:
            # Return appropriate value or tuple based on parameters
            if return_clean:
                return diffraction_pattern, clean_pattern
            else:
                return diffraction_pattern

    def save(self, diffraction_pattern=None, filename='stem_diffraction.png', colormap='viridis', dpi=300):
        """
        Save the diffraction pattern as an image file.

        Parameters:
        -----------
        diffraction_pattern : 2D numpy array, optional
            Diffraction pattern to save. If None, will generate a new pattern.
        filename : str
            Output filename
        colormap : str
            Matplotlib colormap to use
        dpi : int
            Resolution of output image
        """
        if diffraction_pattern is None:
            diffraction_pattern = self.generate()

        plt.figure(figsize=(10, 10))
        plt.imshow(diffraction_pattern, cmap=colormap)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def display(self, diffraction_pattern=None, colormap='viridis', title='STEM Diffraction Pattern'):
        """
        Display the diffraction pattern.

        Parameters:
        -----------
        diffraction_pattern : 2D numpy array, optional
            Diffraction pattern to display. If None, will generate a new pattern.
        colormap : str
            Matplotlib colormap to use
        title : str
            Title for the plot
        """
        if diffraction_pattern is None:
            diffraction_pattern = self.generate()

        plt.figure(figsize=(8, 8))
        plt.imshow(diffraction_pattern, cmap=colormap)
        plt.title(title)
        plt.axis('off')
        plt.colorbar(label='Intensity')
        plt.tight_layout()
        plt.show()

    @classmethod
    @classmethod
    def generate_random(cls, size=256, material=None, generate_mask=False, mask_type="binary",
                        intensity_threshold=0.01, return_clean=False):
        """
        Generate a random diffraction pattern with physically meaningful parameter variations.
        Optionally generate a mask of diffraction peak positions and/or return a clean pattern without noise and background.

        This method creates a diffraction pattern generator with randomly selected but physically
        meaningful parameters. It can either select a random material or use a specified one,
        and will randomize orientation, imaging conditions, and sample parameters.

        Parameters:
        -----------
        size : int
            Size of the output image
        material : str, optional
            Specific material to use. If None, a random material will be selected.
        generate_mask : bool
            If True, generates a mask identifying diffraction peaks
        mask_type : str
            Type of mask to generate:
            - "binary": Binary mask with 1s at peak locations and 0s elsewhere
            - "intensity": Continuous mask preserving the relative intensity of peaks
            - "adaptive": Mask with peaks sized proportionally to their intensity
        intensity_threshold : float
            Minimum intensity threshold for including a spot in the mask (0-1)
        return_clean : bool
            If True, returns a clean version of the diffraction pattern without noise and background

        Returns:
        --------
        If generate_mask=False and return_clean=False:
            tuple
                (diffraction_pattern, generator, parameters_dict)
                - diffraction_pattern: The generated diffraction pattern
                - generator: The STEMDiffractionGenerator instance used
                - parameters_dict: Dictionary with all the randomized parameters for reference

        If generate_mask=True and return_clean=False:
            tuple
                (diffraction_pattern, peak_mask, generator, parameters_dict)
                - diffraction_pattern: The generated diffraction pattern
                - peak_mask: Binary or intensity mask where diffraction peaks are located
                - generator: The STEMDiffractionGenerator instance used
                - parameters_dict: Dictionary with all the randomized parameters for reference

        If generate_mask=False and return_clean=True:
            tuple
                (diffraction_pattern, clean_pattern, generator, parameters_dict)
                - diffraction_pattern: The generated diffraction pattern with noise and background
                - clean_pattern: The clean diffraction pattern without noise and background
                - generator: The STEMDiffractionGenerator instance used
                - parameters_dict: Dictionary with all the randomized parameters for reference

        If generate_mask=True and return_clean=True:
            tuple
                (diffraction_pattern, peak_mask, clean_pattern, generator, parameters_dict)
                - diffraction_pattern: The generated diffraction pattern with noise and background
                - peak_mask: Binary or intensity mask where diffraction peaks are located
                - clean_pattern: The clean diffraction pattern without noise and background
                - generator: The STEMDiffractionGenerator instance used
                - parameters_dict: Dictionary with all the randomized parameters for reference

        Usage:
            # Generate a completely random diffraction pattern
            pattern, generator, params = STEMDiffractionGenerator.generate_random()

            # Generate a pattern with its diffraction peak mask
            pattern, mask, generator, params = STEMDiffractionGenerator.generate_random(
                material='Au',
                generate_mask=True,
                mask_type='adaptive'
            )

            # Generate a pattern with its clean version (no noise or background)
            pattern, clean_pattern, generator, params = STEMDiffractionGenerator.generate_random(
                return_clean=True
            )

            # Generate a pattern with both mask and clean version
            pattern, mask, clean_pattern, generator, params = STEMDiffractionGenerator.generate_random(
                generate_mask=True,
                return_clean=True
            )
        """

        # Get list of all available materials
        materials_list = [
            # Simple structures
            'Al', 'Cu', 'Fe', 'Ni', 'Au', 'Si', 'Ge', 'Mo', 'W', 'Pt',
            # Complex materials
            'Fe3O4', 'TiO2_rutile', 'TiO2_anatase', 'SrTiO3', 'GaAs',
            'ZnO', 'Y2O3', 'MgAl2O4', 'CaCO3_calcite', 'NiO', 'CeO2', 'ZrO2', 'LaB6'
        ]

        # Simple structures data dictionary
        simple_materials = {
            'Al': {'structure': 'fcc', 'lattice': 4.046, 'element': 'Al'},
            'Cu': {'structure': 'fcc', 'lattice': 3.615, 'element': 'Cu'},
            'Fe': {'structure': 'bcc', 'lattice': 2.866, 'element': 'Fe'},
            'Ni': {'structure': 'fcc', 'lattice': 3.524, 'element': 'Ni'},
            'Au': {'structure': 'fcc', 'lattice': 4.078, 'element': 'Au'},
            'Si': {'structure': 'diamond', 'lattice': 5.431, 'element': 'Si'},
            'Ge': {'structure': 'diamond', 'lattice': 5.658, 'element': 'Ge'},
            'Mo': {'structure': 'bcc', 'lattice': 3.147, 'element': 'Mo'},
            'W': {'structure': 'bcc', 'lattice': 3.165, 'element': 'W'},
            'Pt': {'structure': 'fcc', 'lattice': 3.924, 'element': 'Pt'},
        }

        # Common zone axes for different structures
        zone_axes_by_structure = {
            'fcc': [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [1, 1, 3], [0, 1, 2], [0, 1, 3], [1, 2, 3]],
            'bcc': [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [0, 1, 2], [1, 1, 3], [2, 2, 3]],
            'hcp': [[0, 0, 0, 1], [1, 0, -1, 0], [1, -1, 0, 0], [1, 1, -2, 0], [1, 0, -1, 1], [2, -1, -1, 0]],
            'diamond': [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 5]],
            # Catch-all for complex structures
            'complex': [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 2],
                        [1, 2, 1], [2, 1, 1]]
        }

        # Select material (or use provided one)
        if material is None:
            material = random.choice(materials_list)
        elif material not in materials_list:
            raise ValueError(f"Material '{material}' not found in available materials list")

        # Prepare parameters dict to track all random choices
        params_dict = {'material': material}

        # Parameters with physically realistic random ranges
        # Voltage range (kV)
        voltage = random.uniform(80, 300)
        params_dict['voltage'] = round(voltage, 1)

        # Sample thickness (nm) - depends on voltage and material
        min_thickness = 20  # Minimum thickness for any sample
        max_thickness = 10 + voltage * 0.2  # Scales with voltage
        # Adjust based on material (some are typically studied as thinner sections)
        if material in ['Si', 'Ge', 'GaAs', 'ZnO']:
            max_thickness = min(max_thickness, 60)
        elif material in ['Fe3O4', 'TiO2_rutile', 'Y2O3']:
            max_thickness = min(max_thickness, 80)
        sample_thickness = random.uniform(min_thickness, max_thickness)
        params_dict['sample_thickness'] = round(sample_thickness, 1)

        # Convergence angle (mrad) - varies based on typical TEM operating conditions
        convergence_angle = random.uniform(0.5, 5.0)
        params_dict['convergence_angle'] = round(convergence_angle, 2)

        # Camera length (mm) - affects pattern spread
        camera_length = random.choice([300, 400, 500, 600, 800, 1000])
        params_dict['camera_length'] = camera_length

        # Image quality parameters with realistic correlations
        noise_level = random.uniform(0.005, 0.03)
        params_dict['noise_level'] = round(noise_level, 3)

        # Higher voltage generally gives less background variation
        background_variation = random.uniform(0.005, 0.03) * (300 / (voltage + 50))
        params_dict['background_variation'] = round(background_variation, 3)

        # Detector defects - more common in older detectors
        detector_defects = random.choice([True, True, True, True, False])
        params_dict['detector_defects'] = detector_defects

        # Temperature factor - decreases with sample thickness
        temperature_factor = random.uniform(0.7, 0.95) * (1 - sample_thickness / 200)
        params_dict['temperature_factor'] = round(temperature_factor, 2)

        # Direct beam intensity - affected by convergence angle and beam conditions
        direct_beam_intensity = random.uniform(0.5, 0.8) * (1 + convergence_angle / 10)
        params_dict['direct_beam_intensity'] = round(min(direct_beam_intensity, 0.9), 2)

        # Spot intensity/size factors - trade-off between them
        spot_intensity_factor = random.uniform(0.06, 0.18)
        params_dict['spot_intensity_factor'] = round(spot_intensity_factor, 2)

        spot_size_factor = random.uniform(0.8, 1) * (1.2 - spot_intensity_factor)
        params_dict['spot_size_factor'] = round(spot_size_factor, 2)

        # Kikuchi and HOLZ intensities - stronger in thicker samples
        thickness_factor = min(sample_thickness / 50, 1.0)
        kikuchi_intensity_factor = random.uniform(0.01, 0.05) * thickness_factor
        params_dict['kikuchi_intensity_factor'] = round(kikuchi_intensity_factor, 3)

        holz_intensity = random.uniform(0.01, 0.04) * thickness_factor
        params_dict['holz_intensity'] = round(holz_intensity, 3)

        # Determine zone axis based on material type
        if material in simple_materials:
            # For simple materials, use the structure to determine zone axis options
            structure = simple_materials[material]['structure']
            available_axes = zone_axes_by_structure.get(structure, zone_axes_by_structure['complex'])

            # Adjust for special cases like hexagonal structures
            if structure == 'hcp':
                # Convert 4-index to 3-index for internal processing
                selected_zone_axis = random.choice(available_axes)
                if len(selected_zone_axis) == 4:
                    h, k, i, l = selected_zone_axis
                    # Convert from Miller-Bravais (4-index) to Miller (3-index)
                    zone_axis = [h, k, l]
                    # Keep original for display
                    zone_axis_str = f"{h},{k},{i},{l}"
                else:
                    zone_axis = selected_zone_axis
                    zone_axis_str = ','.join(map(str, zone_axis))
            else:
                zone_axis = random.choice(available_axes)
                zone_axis_str = ','.join(map(str, zone_axis))

            # Create generator for simple material
            a = simple_materials[material]['lattice']
            element = simple_materials[material]['element']

            generator_params = {
                'size': size,
                'crystal_structure': structure,
                'lattice_params': {'a': a, 'b': a, 'c': a, 'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': [element],
                'atomic_positions': [{'element': element, 'position': [0, 0, 0]}],
                'voltage': voltage,
                'camera_length': camera_length,
                'convergence_angle': convergence_angle,
                'zone_axis': zone_axis,
                'sample_thickness': sample_thickness,
                'noise_level': noise_level,
                'background_variation': background_variation,
                'detector_defects': detector_defects,
                'temperature_factor': temperature_factor,
                'direct_beam_intensity': direct_beam_intensity,
                'spot_intensity_factor': spot_intensity_factor,
                'spot_size_factor': spot_size_factor,
                'kikuchi_intensity_factor': kikuchi_intensity_factor,
                'holz_intensity': holz_intensity
            }

            # Special case for diamond structure (Si, Ge)
            if structure == 'diamond':
                generator_params['atomic_positions'] = [
                    {'element': element, 'position': [0, 0, 0]},
                    {'element': element, 'position': [0.25, 0.25, 0.25]},
                ]

            # Special case for hcp (different c/a ratio and gamma angle)
            if structure == 'hcp':
                c_a_ratio = 1.633 if element not in ['Zn', 'Cd'] else 1.856
                generator_params['lattice_params'] = {'a': a, 'b': a, 'c': a * c_a_ratio,
                                                      'alpha': 90, 'beta': 90, 'gamma': 120}
                generator_params['atomic_positions'] = [
                    {'element': element, 'position': [0, 0, 0]},
                    {'element': element, 'position': [1 / 3, 2 / 3, 1 / 2]},
                ]

            generator = cls(**generator_params)

        else:
            # For complex materials, use the from_material class method
            # Select a random zone axis from the complex list
            complex_axes = zone_axes_by_structure['complex']
            zone_axis = random.choice(complex_axes)
            zone_axis_str = ','.join(map(str, zone_axis))

            # Create generator for complex material
            generator = cls.from_material(
                material_name=material,
                size=size,
                voltage=voltage,
                camera_length=camera_length,
                convergence_angle=convergence_angle,
                zone_axis=zone_axis,
                sample_thickness=sample_thickness,
                noise_level=noise_level,
                background_variation=background_variation,
                detector_defects=detector_defects,
                temperature_factor=temperature_factor,
                direct_beam_intensity=direct_beam_intensity,
                spot_intensity_factor=spot_intensity_factor,
                spot_size_factor=spot_size_factor,
                kikuchi_intensity_factor=kikuchi_intensity_factor,
                holz_intensity=holz_intensity
            )

        # Store zone axis info
        params_dict['zone_axis'] = zone_axis
        params_dict['zone_axis_str'] = f"[{zone_axis_str}]"

        # Add mask parameters to params_dict if generating mask
        if generate_mask:
            params_dict['mask_type'] = mask_type
            params_dict['intensity_threshold'] = intensity_threshold

        # Add clean pattern flag to params_dict
        if return_clean:
            params_dict['return_clean'] = return_clean

        # Generate the diffraction pattern with appropriate options
        if generate_mask and return_clean:
            diffraction_pattern, mask, clean_pattern = generator.generate(
                generate_mask=True,
                mask_type=mask_type,
                intensity_threshold=intensity_threshold,
                return_clean=True
            )
            return diffraction_pattern, mask, clean_pattern, generator, params_dict
        elif generate_mask:
            diffraction_pattern, mask = generator.generate(
                generate_mask=True,
                mask_type=mask_type,
                intensity_threshold=intensity_threshold
            )
            return diffraction_pattern, mask, generator, params_dict
        elif return_clean:
            diffraction_pattern, clean_pattern = generator.generate(
                generate_mask=False,
                return_clean=True
            )
            return diffraction_pattern, clean_pattern, generator, params_dict
        else:
            diffraction_pattern = generator.generate(generate_mask=False)
            return diffraction_pattern, generator, params_dict

    @classmethod
    def from_material(cls, material_name, size=256, voltage: float = 200, camera_length=300,
                      convergence_angle=1.5, zone_axis=None, **kwargs):
        """
        Create a diffraction pattern generator for a specific complex material.

        Parameters:
        -----------
        material_name : str
            Name of the material. Available options:
            - 'Fe3O4' (Magnetite)
            - 'TiO2_rutile' (Rutile titanium dioxide)
            - 'TiO2_anatase' (Anatase titanium dioxide)
            - 'SrTiO3' (Strontium titanate - perovskite)
            - 'GaAs' (Gallium arsenide)
            - 'ZnO' (Zinc oxide - wurtzite)
            - 'Y2O3' (Yttrium oxide)
            - 'MgAl2O4' (Magnesium aluminate spinel)
            - 'CaCO3_calcite' (Calcite calcium carbonate)
            - 'NiO' (Nickel oxide)
            - 'CeO2' (Cerium dioxide - fluorite)
            - 'ZrO2' (Zirconia - monoclinic)
            - 'LaB6' (Lanthanum hexaboride)
        size : int
            Size of the output image
        voltage : float
            Electron beam voltage in kV
        camera_length : float
            Camera length in mm
        convergence_angle : float
            Beam convergence semi-angle in mrad
        zone_axis : list, optional
            Zone axis direction [h,k,l]. If None, a default will be used for each material.
        **kwargs :
            Additional parameters to override defaults for the selected material

        Returns:
        --------
        STEMDiffractionGenerator
            Configured generator for the specified material

        Usage:
            # Generate a magnetite diffraction pattern with default settings
            generator = STEMDiffractionGenerator.from_material('Fe3O4')
            pattern = generator.generate()

            # Generate GaAs with a specific zone axis
            generator = STEMDiffractionGenerator.from_material('GaAs', zone_axis=[1,1,0])
            pattern = generator.generate()
        """
        # Dictionary of complex materials with their parameters
        materials = {
            # Magnetite (Iron(II,III) oxide) - Inverse spinel structure
            'Fe3O4': {
                'crystal_structure': 'spinel',
                'lattice_params': {'a': 8.396, 'b': 8.396, 'c': 8.396,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Fe', 'O'],
                'atomic_positions': [
                    {'element': 'Fe', 'position': [0, 0, 0]},  # Fe3+ at tetrahedral sites
                    {'element': 'Fe', 'position': [0.625, 0.625, 0.625]},  # Fe2+ and Fe3+ at octahedral sites
                    {'element': 'Fe', 'position': [0.625, 0.375, 0.375]},
                    {'element': 'Fe', 'position': [0.375, 0.625, 0.375]},
                    {'element': 'Fe', 'position': [0.375, 0.375, 0.625]},
                    {'element': 'O', 'position': [0.375, 0.375, 0.375]},
                    {'element': 'O', 'position': [0.375, 0.625, 0.625]},
                    {'element': 'O', 'position': [0.625, 0.375, 0.625]},
                    {'element': 'O', 'position': [0.625, 0.625, 0.375]},
                ],
                'default_zone_axis': [1, 1, 1],
                'sample_thickness': 40,
                'temperature_factor': 0.75,
            },

            # Rutile TiO2 - Tetragonal
            'TiO2_rutile': {
                'crystal_structure': 'rutile',
                'lattice_params': {'a': 4.594, 'b': 4.594, 'c': 2.959,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Ti', 'O'],
                'atomic_positions': [
                    {'element': 'Ti', 'position': [0, 0, 0]},
                    {'element': 'Ti', 'position': [0.5, 0.5, 0.5]},
                    {'element': 'O', 'position': [0.3, 0.3, 0]},
                    {'element': 'O', 'position': [0.7, 0.7, 0]},
                    {'element': 'O', 'position': [0.8, 0.2, 0.5]},
                    {'element': 'O', 'position': [0.2, 0.8, 0.5]},
                ],
                'default_zone_axis': [0, 0, 1],
                'sample_thickness': 30,
            },

            # Anatase TiO2 - Tetragonal
            'TiO2_anatase': {
                'crystal_structure': 'anatase',
                'lattice_params': {'a': 3.785, 'b': 3.785, 'c': 9.514,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Ti', 'O'],
                'atomic_positions': [
                    {'element': 'Ti', 'position': [0, 0, 0]},
                    {'element': 'Ti', 'position': [0, 0.5, 0.25]},
                    {'element': 'Ti', 'position': [0.5, 0, 0.25]},
                    {'element': 'Ti', 'position': [0.5, 0.5, 0]},
                    {'element': 'O', 'position': [0, 0, 0.2]},
                    {'element': 'O', 'position': [0, 0, 0.8]},
                    {'element': 'O', 'position': [0, 0.5, 0.05]},
                    {'element': 'O', 'position': [0, 0.5, 0.45]},
                ],
                'default_zone_axis': [0, 0, 1],
                'sample_thickness': 25,
            },

            # Strontium Titanate - Perovskite
            'SrTiO3': {
                'crystal_structure': 'perovskite',
                'lattice_params': {'a': 3.905, 'b': 3.905, 'c': 3.905,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Sr', 'Ti', 'O'],
                'atomic_positions': [
                    {'element': 'Sr', 'position': [0, 0, 0]},
                    {'element': 'Ti', 'position': [0.5, 0.5, 0.5]},
                    {'element': 'O', 'position': [0.5, 0.5, 0]},
                    {'element': 'O', 'position': [0.5, 0, 0.5]},
                    {'element': 'O', 'position': [0, 0.5, 0.5]},
                ],
                'default_zone_axis': [0, 0, 1],
                'sample_thickness': 35,
            },

            # Gallium Arsenide - Zincblende structure
            'GaAs': {
                'crystal_structure': 'zincblende',
                'lattice_params': {'a': 5.653, 'b': 5.653, 'c': 5.653,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Ga', 'As'],
                'atomic_positions': [
                    {'element': 'Ga', 'position': [0, 0, 0]},
                    {'element': 'Ga', 'position': [0, 0.5, 0.5]},
                    {'element': 'Ga', 'position': [0.5, 0, 0.5]},
                    {'element': 'Ga', 'position': [0.5, 0.5, 0]},
                    {'element': 'As', 'position': [0.25, 0.25, 0.25]},
                    {'element': 'As', 'position': [0.25, 0.75, 0.75]},
                    {'element': 'As', 'position': [0.75, 0.25, 0.75]},
                    {'element': 'As', 'position': [0.75, 0.75, 0.25]},
                ],
                'default_zone_axis': [1, 1, 0],
                'sample_thickness': 30,
            },

            # Zinc Oxide - Wurtzite structure
            'ZnO': {
                'crystal_structure': 'hcp',
                'lattice_params': {'a': 3.25, 'b': 3.25, 'c': 5.21,
                                   'alpha': 90, 'beta': 90, 'gamma': 120},
                'elements': ['Zn', 'O'],
                'atomic_positions': [
                    {'element': 'Zn', 'position': [0, 0, 0]},
                    {'element': 'Zn', 'position': [0.333, 0.667, 0.5]},
                    {'element': 'O', 'position': [0, 0, 0.382]},
                    {'element': 'O', 'position': [0.333, 0.667, 0.882]},
                ],
                'default_zone_axis': [0, 0, 1],
                'sample_thickness': 25,
            },

            # Yttrium Oxide - Cubic
            'Y2O3': {
                'crystal_structure': 'bcc',  # Simplified - actually body-centered cubic bixbyite
                'lattice_params': {'a': 10.604, 'b': 10.604, 'c': 10.604,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Y', 'O'],
                'atomic_positions': [
                    {'element': 'Y', 'position': [0, 0, 0]},
                    {'element': 'Y', 'position': [0.5, 0, 0.5]},
                    {'element': 'Y', 'position': [0.5, 0.5, 0]},
                    {'element': 'Y', 'position': [0, 0.5, 0.5]},
                    {'element': 'O', 'position': [0.25, 0.25, 0.25]},
                    {'element': 'O', 'position': [0.75, 0.25, 0.75]},
                    {'element': 'O', 'position': [0.75, 0.75, 0.25]},
                    {'element': 'O', 'position': [0.25, 0.75, 0.75]},
                ],
                'default_zone_axis': [1, 0, 0],
                'sample_thickness': 40,
            },

            # Magnesium Aluminate Spinel
            'MgAl2O4': {
                'crystal_structure': 'spinel',
                'lattice_params': {'a': 8.08, 'b': 8.08, 'c': 8.08,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Mg', 'Al', 'O'],
                'atomic_positions': [
                    {'element': 'Mg', 'position': [0, 0, 0]},
                    {'element': 'Mg', 'position': [0.5, 0.5, 0]},
                    {'element': 'Mg', 'position': [0.5, 0, 0.5]},
                    {'element': 'Mg', 'position': [0, 0.5, 0.5]},
                    {'element': 'Al', 'position': [0.125, 0.125, 0.125]},
                    {'element': 'Al', 'position': [0.375, 0.375, 0.125]},
                    {'element': 'Al', 'position': [0.375, 0.125, 0.375]},
                    {'element': 'Al', 'position': [0.125, 0.375, 0.375]},
                    {'element': 'O', 'position': [0.25, 0.25, 0.25]},
                    {'element': 'O', 'position': [0.75, 0.75, 0.25]},
                    {'element': 'O', 'position': [0.75, 0.25, 0.75]},
                    {'element': 'O', 'position': [0.25, 0.75, 0.75]},
                ],
                'default_zone_axis': [1, 1, 1],
                'sample_thickness': 35,
            },

            # Calcite (CaCO3) - Trigonal
            'CaCO3_calcite': {
                'crystal_structure': 'trigonal',
                'lattice_params': {'a': 4.99, 'b': 4.99, 'c': 17.06,
                                   'alpha': 90, 'beta': 90, 'gamma': 120},
                'elements': ['Ca', 'C', 'O'],
                'atomic_positions': [
                    {'element': 'Ca', 'position': [0, 0, 0]},
                    {'element': 'Ca', 'position': [0, 0, 0.5]},
                    {'element': 'C', 'position': [0, 0, 0.25]},
                    {'element': 'C', 'position': [0, 0, 0.75]},
                    {'element': 'O', 'position': [0.25, 0, 0.25]},
                    {'element': 'O', 'position': [0, 0.25, 0.25]},
                    {'element': 'O', 'position': [-0.25, -0.25, 0.25]},
                    {'element': 'O', 'position': [0.25, 0, 0.75]},
                    {'element': 'O', 'position': [0, 0.25, 0.75]},
                    {'element': 'O', 'position': [-0.25, -0.25, 0.75]},
                ],
                'default_zone_axis': [0, 0, 1],
                'sample_thickness': 30,
            },

            # Nickel Oxide - Rocksalt structure
            'NiO': {
                'crystal_structure': 'rocksalt',
                'lattice_params': {'a': 4.177, 'b': 4.177, 'c': 4.177,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Ni', 'O'],
                'atomic_positions': [
                    {'element': 'Ni', 'position': [0, 0, 0]},
                    {'element': 'Ni', 'position': [0, 0.5, 0.5]},
                    {'element': 'Ni', 'position': [0.5, 0, 0.5]},
                    {'element': 'Ni', 'position': [0.5, 0.5, 0]},
                    {'element': 'O', 'position': [0.5, 0, 0]},
                    {'element': 'O', 'position': [0.5, 0.5, 0.5]},
                    {'element': 'O', 'position': [0, 0.5, 0]},
                    {'element': 'O', 'position': [0, 0, 0.5]},
                ],
                'default_zone_axis': [1, 0, 0],
                'sample_thickness': 25,
            },

            # Cerium Dioxide (Ceria) - Fluorite structure
            'CeO2': {
                'crystal_structure': 'fluorite',
                'lattice_params': {'a': 5.411, 'b': 5.411, 'c': 5.411,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['Ce', 'O'],
                'atomic_positions': [
                    {'element': 'Ce', 'position': [0, 0, 0]},
                    {'element': 'Ce', 'position': [0, 0.5, 0.5]},
                    {'element': 'Ce', 'position': [0.5, 0, 0.5]},
                    {'element': 'Ce', 'position': [0.5, 0.5, 0]},
                    {'element': 'O', 'position': [0.25, 0.25, 0.25]},
                    {'element': 'O', 'position': [0.25, 0.75, 0.75]},
                    {'element': 'O', 'position': [0.75, 0.25, 0.75]},
                    {'element': 'O', 'position': [0.75, 0.75, 0.25]},
                    {'element': 'O', 'position': [0.75, 0.75, 0.75]},
                    {'element': 'O', 'position': [0.75, 0.25, 0.25]},
                    {'element': 'O', 'position': [0.25, 0.75, 0.25]},
                    {'element': 'O', 'position': [0.25, 0.25, 0.75]},
                ],
                'default_zone_axis': [1, 1, 0],
                'sample_thickness': 30,
            },

            # Zirconia - Monoclinic phase
            'ZrO2': {
                'crystal_structure': 'monoclinic',
                'lattice_params': {'a': 5.151, 'b': 5.212, 'c': 5.317,
                                   'alpha': 90, 'beta': 99.23, 'gamma': 90},
                'elements': ['Zr', 'O'],
                'atomic_positions': [
                    {'element': 'Zr', 'position': [0, 0, 0]},
                    {'element': 'Zr', 'position': [0.5, 0.5, 0]},
                    {'element': 'O', 'position': [0.25, 0.25, 0.07]},
                    {'element': 'O', 'position': [0.75, 0.25, 0.43]},
                    {'element': 'O', 'position': [0.25, 0.75, 0.57]},
                    {'element': 'O', 'position': [0.75, 0.75, 0.93]},
                ],
                'default_zone_axis': [0, 1, 0],
                'sample_thickness': 20,
            },

            # Lanthanum Hexaboride - Cubic
            'LaB6': {
                'crystal_structure': 'sc',  # Simple cubic
                'lattice_params': {'a': 4.156, 'b': 4.156, 'c': 4.156,
                                   'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': ['La', 'B'],
                'atomic_positions': [
                    {'element': 'La', 'position': [0, 0, 0]},
                    {'element': 'B', 'position': [0.2, 0.5, 0.5]},
                    {'element': 'B', 'position': [0.8, 0.5, 0.5]},
                    {'element': 'B', 'position': [0.5, 0.2, 0.5]},
                    {'element': 'B', 'position': [0.5, 0.8, 0.5]},
                    {'element': 'B', 'position': [0.5, 0.5, 0.2]},
                    {'element': 'B', 'position': [0.5, 0.5, 0.8]},
                ],
                'default_zone_axis': [1, 0, 0],
                'sample_thickness': 30,
            },
        }

        # Check if the material is supported
        if material_name not in materials:
            raise ValueError(f"Material '{material_name}' not found. Available materials: {list(materials.keys())}")

        # Get the material parameters
        params = materials[material_name].copy()

        # Use provided zone axis or default
        if zone_axis is None:
            zone_axis = params.pop('default_zone_axis')

        # Remove the default_zone_axis from parameters
        if 'default_zone_axis' in params:
            del params['default_zone_axis']

        # Add the common parameters
        params.update({
            'size': size,
            'voltage': voltage,
            'camera_length': camera_length,
            'convergence_angle': convergence_angle,
            'zone_axis': zone_axis
        })

        # Override with any custom parameters
        params.update(kwargs)

        # Create and return the generator instance
        return cls(**params)

    @classmethod
    def sample_orientations(cls, material, num_patterns=10, size=256, voltage=200, camera_length=300,
                            convergence_angle=1.5, sample_thickness=30, uniform=False, save_dir=None,
                            custom_orientations=None, return_generators=False, sampling_mode='common',
                            no_low_index_approximation=False, generate_mask=False, mask_type="binary",
                            intensity_threshold=0.01, return_clean=False):
        """
        Generate multiple diffraction patterns for the same material with different orientations.

        This method creates a set of diffraction patterns for a single material by sampling
        different crystallographic orientations while keeping detector parameters fixed.

        Parameters:
        -----------
        material : str
            Name of the material to use (must be one of the supported materials or elements)
        num_patterns : int
            Number of different orientations to generate
        size : int
            Size of the output images (size x size pixels)
        voltage : float
            Electron beam voltage in kV (fixed for all patterns)
        camera_length : float
            Camera length in mm (fixed for all patterns)
        convergence_angle : float
            Beam convergence semi-angle in mrad (fixed for all patterns)
        sample_thickness : float
            Sample thickness in nm (fixed for all patterns)
        sampling_mode : str
            Mode for orientation sampling:
            - 'common': Uses predefined common zone axes
            - 'uniform': Uniform random sampling over SO(3) rotation space
            - 'powder': High-density uniform sampling optimized for powder diffraction
            - 'custom': Uses orientations provided in custom_orientations parameter
        uniform : bool (deprecated, use sampling_mode instead)
            Legacy parameter for backward compatibility
        no_low_index_approximation : bool
            If True, keeps exact random orientations without approximating to low-index directions
            (Important for true powder diffraction simulation)
        save_dir : str, optional
            Directory to save generated patterns. If None, patterns are not saved.
        custom_orientations : list, optional
            List of custom zone axes to use instead of random sampling, e.g. [[0,0,1], [1,1,0]]
        return_generators : bool
            If True, returns the generator objects along with patterns
        generate_mask : bool
            If True, generates a mask identifying diffraction peaks for each pattern
        mask_type : str
            Type of mask to generate (binary, intensity, or adaptive)
        intensity_threshold : float
            Minimum intensity threshold for including a spot in the mask (0-1)
        return_clean : bool
            If True, returns clean versions of the diffraction patterns without noise and background

        Returns:
        --------
        dict
            Dictionary containing:
            - 'patterns': List of diffraction patterns
            - 'orientations': List of corresponding orientation information
            - 'generators': List of generator objects (if return_generators=True)
            - 'material': The material used
            - 'parameters': Dictionary of fixed parameters

            If generate_mask=True:
            - 'masks': List of masks corresponding to each diffraction pattern

            If return_clean=True:
            - 'clean_patterns': List of clean diffraction patterns without noise and background

        Usage:
            # Generate 10 patterns of Silicon with common orientations
            results = STEMDiffractionGenerator.sample_orientations('Si', num_patterns=10)

            # Generate patterns for powder diffraction simulation with clean patterns
            results = STEMDiffractionGenerator.sample_orientations(
                'Au',
                num_patterns=100,
                sampling_mode='powder',
                no_low_index_approximation=True,
                return_clean=True
            )

            # Generate patterns with specific orientations and masks
            results = STEMDiffractionGenerator.sample_orientations(
                'Fe3O4',
                sampling_mode='custom',
                custom_orientations=[[0,0,1], [1,1,0], [1,1,1]],
                generate_mask=True
            )
        """
        import random
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        # Support legacy uniform parameter
        if uniform:
            sampling_mode = 'uniform'

        # If custom orientations provided, force sampling_mode to 'custom'
        if custom_orientations:
            sampling_mode = 'custom'

        # Validate material
        materials_list = [
            # Simple elements
            'Al', 'Cu', 'Fe', 'Ni', 'Au', 'Si', 'Ge', 'Mo', 'W', 'Pt',
            # Complex materials
            'Fe3O4', 'TiO2_rutile', 'TiO2_anatase', 'SrTiO3', 'GaAs',
            'ZnO', 'Y2O3', 'MgAl2O4', 'CaCO3_calcite', 'NiO', 'CeO2', 'ZrO2', 'LaB6'
        ]

        if material not in materials_list:
            raise ValueError(f"Material '{material}' not found. Available materials: {materials_list}")

        # Create directory if specified
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Define common zone axes based on structure
        simple_materials = {
            'Al': {'structure': 'fcc', 'lattice': 4.046, 'element': 'Al'},
            'Cu': {'structure': 'fcc', 'lattice': 3.615, 'element': 'Cu'},
            'Fe': {'structure': 'bcc', 'lattice': 2.866, 'element': 'Fe'},
            'Ni': {'structure': 'fcc', 'lattice': 3.524, 'element': 'Ni'},
            'Au': {'structure': 'fcc', 'lattice': 4.078, 'element': 'Au'},
            'Si': {'structure': 'diamond', 'lattice': 5.431, 'element': 'Si'},
            'Ge': {'structure': 'diamond', 'lattice': 5.658, 'element': 'Ge'},
            'Mo': {'structure': 'bcc', 'lattice': 3.147, 'element': 'Mo'},
            'W': {'structure': 'bcc', 'lattice': 3.165, 'element': 'W'},
            'Pt': {'structure': 'fcc', 'lattice': 3.924, 'element': 'Pt'},
        }

        # Common zone axes for different structures
        zone_axes_by_structure = {
            'fcc': [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [1, 1, 3], [0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 1, 0],
                    [1, 3, 1]],
            'bcc': [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [0, 1, 2], [1, 1, 3], [2, 2, 3], [3, 1, 0], [2, 1, 1],
                    [3, 2, 1]],
            'hcp': [[0, 0, 0, 1], [1, 0, -1, 0], [1, -1, 0, 0], [1, 1, -2, 0], [1, 0, -1, 1], [2, -1, -1, 0],
                    [1, -1, 0, 2], [0, 0, 0, 2]],
            'diamond': [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 5], [0, 0, 2], [1, 3, 1],
                        [1, 3, 3], [2, 3, 1]],
            # Catch-all for complex structures
            'complex': [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 2],
                        [1, 2, 1], [2, 1, 1],
                        [2, 0, 1], [2, 1, 0], [2, 2, 0], [2, 2, 1], [3, 0, 1], [3, 1, 0], [3, 1, 1], [1, 0, 2],
                        [1, 2, 0], [1, 2, 2]]
        }

        # Function to generate uniform random orientations
        def random_quaternion():
            # Generate uniform random quaternion (using method from Ken Shoemake)
            u1, u2, u3 = np.random.random(3)
            q = [
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
                np.sqrt(u1) * np.cos(2 * np.pi * u3)
            ]
            return q

        def random_quaternion_gaussian():
            # Generate random quaternion using 4D gaussian (alternative method)
            q = [random.gauss(0, 1) for _ in range(4)]
            norm = np.sqrt(sum(x * x for x in q))
            return [x / norm for x in q]

        def quaternion_to_matrix(q):
            # Convert quaternion to rotation matrix
            w, x, y, z = q
            return np.array([
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
            ])

        def rotation_to_zone_axis(R):
            # Extract zone axis (z-direction) from rotation matrix
            # We use the 3rd column of the rotation matrix as our zone axis
            return list(R[:, 2])

        def find_nearby_low_index(za, max_index=5, angle_threshold=0.1):
            """Find the nearest low-index direction to the given zone axis"""
            if no_low_index_approximation:
                # Skip approximation and return original zone axis if flag is set
                return None

            za_magnitude = np.sqrt(sum(z * z for z in za))
            za_normalized = [z / za_magnitude for z in za]

            best_match = None
            min_dist = float('inf')

            for h in range(-max_index, max_index + 1):
                for k in range(-max_index, max_index + 1):
                    for l in range(-max_index, max_index + 1):
                        if h == 0 and k == 0 and l == 0:
                            continue
                        hkl = [h, k, l]
                        hkl_magnitude = np.sqrt(sum(idx * idx for idx in hkl))
                        hkl_normalized = [idx / hkl_magnitude for idx in hkl]

                        # Calculate angular distance
                        dot_product = sum(a * b for a, b in zip(za_normalized, hkl_normalized))
                        if dot_product > 1.0:  # Handle numerical precision issues
                            dot_product = 1.0
                        angle = np.arccos(dot_product)

                        if angle < min_dist and angle < angle_threshold:  # Within threshold
                            min_dist = angle
                            best_match = hkl

            return best_match

        # Lists to store results
        patterns = []
        orientations = []
        generators = []
        if generate_mask:
            masks = []
        if return_clean:
            clean_patterns = []

        # Determine orientations to use based on sampling mode
        if sampling_mode == 'custom':
            # Use provided orientations
            if not custom_orientations:
                raise ValueError("Custom orientations must be provided when using 'custom' sampling mode")
            orientation_list = custom_orientations

        elif sampling_mode == 'common':
            # Use predefined zone axes based on structure
            if material in simple_materials:
                structure = simple_materials[material]['structure']
                available_axes = zone_axes_by_structure.get(structure, zone_axes_by_structure['complex'])
            else:
                available_axes = zone_axes_by_structure['complex']

            # Select zone axes randomly without repeats if possible
            if num_patterns <= len(available_axes):
                orientation_list = random.sample(available_axes, num_patterns)
            else:
                # If more patterns requested than available axes, allow repeats
                orientation_list = random.choices(available_axes, k=num_patterns)

        elif sampling_mode in ['uniform', 'powder']:
            # For powder diffraction, we want truly random orientations with no low-index bias
            if sampling_mode == 'powder':
                no_low_index_approximation = True

            # Generate uniform random orientations
            orientation_list = []
            for _ in range(num_patterns):
                # For powder mode, use Gaussian method which may provide better uniformity for large samples
                if sampling_mode == 'powder':
                    q = random_quaternion_gaussian()
                else:
                    q = random_quaternion()

                R = quaternion_to_matrix(q)
                za = rotation_to_zone_axis(R)

                if not no_low_index_approximation:
                    # Try to find a nearby low-index direction
                    best_match = find_nearby_low_index(za)

                    if best_match:
                        orientation_list.append(best_match)
                        continue

                # If no approximation or no match found, use the exact orientation
                # For powder diffraction, keep the floating point values
                if no_low_index_approximation:
                    orientation_list.append(za)
                else:
                    # Round to integers if possible, otherwise use float values
                    rounded_za = [round(z) if abs(z - round(z)) < 0.05 else z for z in za]
                    orientation_list.append(rounded_za)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}. Use 'common', 'uniform', 'powder', or 'custom'")

        # Generate patterns for each orientation
        valid_pattern_count = 0
        max_attempts = num_patterns * 5  # Limit the number of attempts to avoid infinite loops
        attempt_count = 0

        while valid_pattern_count < num_patterns and attempt_count < max_attempts:
            attempt_count += 1

            # Get the current orientation to try
            if attempt_count <= len(orientation_list):
                zone_axis = orientation_list[attempt_count - 1]
            else:
                # If we've exhausted the list but need more valid patterns, generate a new orientation
                if sampling_mode == 'powder':
                    q = random_quaternion_gaussian()
                elif sampling_mode == 'uniform':
                    q = random_quaternion()
                else:
                    # For common or custom modes, cycle through the list
                    zone_axis = orientation_list[(attempt_count - 1) % len(orientation_list)]
                    continue

                R = quaternion_to_matrix(q)
                zone_axis = rotation_to_zone_axis(R)

                if not no_low_index_approximation:
                    best_match = find_nearby_low_index(zone_axis)
                    if best_match:
                        zone_axis = best_match

            # Convert 4-index notation to 3-index for hcp if needed
            zone_axis_3index = zone_axis
            zone_axis_str = str(zone_axis)

            if len(zone_axis) == 4 and material in simple_materials and simple_materials[material][
                'structure'] == 'hcp':
                h, k, i_idx, l = zone_axis
                # Convert from Miller-Bravais to Miller indices
                zone_axis_3index = [h, k, l]

            # Create generator based on material type
            if material in simple_materials:
                # Simple material
                a = simple_materials[material]['lattice']
                element = simple_materials[material]['element']

                generator_params = {
                    'size': size,
                    'crystal_structure': simple_materials[material]['structure'],
                    'lattice_params': {'a': a, 'b': a, 'c': a, 'alpha': 90, 'beta': 90, 'gamma': 90},
                    'elements': [element],
                    'atomic_positions': [{'element': element, 'position': [0, 0, 0]}],
                    'voltage': voltage,
                    'camera_length': camera_length,
                    'convergence_angle': convergence_angle,
                    'zone_axis': zone_axis_3index,
                    'sample_thickness': sample_thickness,
                }

                # Special case for diamond structure (Si, Ge)
                if simple_materials[material]['structure'] == 'diamond':
                    generator_params['atomic_positions'] = [
                        {'element': element, 'position': [0, 0, 0]},
                        {'element': element, 'position': [0.25, 0.25, 0.25]},
                    ]

                # Special case for hcp (different c/a ratio and gamma angle)
                if simple_materials[material]['structure'] == 'hcp':
                    c_a_ratio = 1.633 if element not in ['Zn', 'Cd'] else 1.856
                    generator_params['lattice_params'] = {'a': a, 'b': a, 'c': a * c_a_ratio,
                                                          'alpha': 90, 'beta': 90, 'gamma': 120}
                    generator_params['atomic_positions'] = [
                        {'element': element, 'position': [0, 0, 0]},
                        {'element': element, 'position': [1 / 3, 2 / 3, 1 / 2]},
                    ]

                generator = cls(**generator_params)

            else:
                # Complex material - use from_material with specified zone axis
                generator = cls.from_material(
                    material_name=material,
                    size=size,
                    voltage=voltage,
                    camera_length=camera_length,
                    convergence_angle=convergence_angle,
                    zone_axis=zone_axis_3index,
                    sample_thickness=sample_thickness,
                )

            # For powder mode, first generate the reflections and check if we have enough
            if sampling_mode == 'powder':
                # Get the reflections directly from the generator
                reflections = generator._generate_reflections(max_index=6)

                # Filter out reflections with very low intensity (not visible in pattern)
                # Use a very small threshold to only remove essentially zero-intensity spots
                visible_reflections = [ref for ref in reflections if ref['intensity'] > 1e-4]

                # Count the visible reflections (excluding the direct beam)
                num_spots = len(visible_reflections)

                # Require a minimum number of diffraction spots
                min_required_spots = 3

                if num_spots < min_required_spots:
                    # Skip this orientation - not enough diffraction spots
                    continue

            # Generate the actual diffraction pattern based on the requested options
            if generate_mask and return_clean:
                pattern, mask, clean_pattern = generator.generate(
                    generate_mask=True,
                    mask_type=mask_type,
                    intensity_threshold=intensity_threshold,
                    return_clean=True
                )
            elif generate_mask:
                pattern, mask = generator.generate(
                    generate_mask=True,
                    mask_type=mask_type,
                    intensity_threshold=intensity_threshold
                )
            elif return_clean:
                pattern, clean_pattern = generator.generate(
                    generate_mask=False,
                    return_clean=True
                )
            else:
                pattern = generator.generate()

            # If we've reached here, the pattern is valid
            valid_pattern_count += 1

            # Save pattern if requested
            if save_dir:
                # Format zone axis string for filename
                if isinstance(zone_axis[0], float):
                    # For floating point zone axes (like in powder mode), round to 3 decimal places for filename
                    za_str = '_'.join(f"{idx:.3f}".replace('.', 'p') for idx in zone_axis)
                else:
                    za_str = '_'.join(str(idx) for idx in zone_axis)

                filename = f"{material}_{za_str}.png"
                filepath = os.path.join(save_dir, filename)
                plt.imsave(filepath, pattern, cmap='viridis')

                # Save mask if generated
                if generate_mask:
                    mask_filename = f"{material}_{za_str}_mask.png"
                    mask_filepath = os.path.join(save_dir, mask_filename)
                    plt.imsave(mask_filepath, mask, cmap='gray')

                # Save clean pattern if generated
                if return_clean:
                    clean_filename = f"{material}_{za_str}_clean.png"
                    clean_filepath = os.path.join(save_dir, clean_filename)
                    plt.imsave(clean_filepath, clean_pattern, cmap='viridis')

                # Save metadata
                metadata = {
                    'material': material,
                    'zone_axis': zone_axis,
                    'voltage': voltage,
                    'camera_length': camera_length,
                    'convergence_angle': convergence_angle,
                    'sample_thickness': sample_thickness,
                    'size': size,
                    'sampling_mode': sampling_mode
                }

                # Add mask and clean pattern info to metadata if applicable
                if generate_mask:
                    metadata['mask_type'] = mask_type
                    metadata['intensity_threshold'] = intensity_threshold

                if return_clean:
                    metadata['return_clean'] = True

                # Add spot count and reflection data for powder mode
                if sampling_mode == 'powder':
                    metadata['diffraction_spot_count'] = num_spots
                    metadata['reflections'] = [
                        {'h': ref['h'], 'k': ref['k'], 'l': ref['l'],
                         'intensity': float(ref['intensity'])}
                        for ref in visible_reflections[:10]  # Store the 10 strongest reflections
                    ]

                metadata_file = os.path.join(save_dir, f"{material}_{za_str}_metadata.json")
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

            # Store results
            patterns.append(pattern)
            orientations.append(zone_axis_str)
            if return_generators:
                generators.append(generator)
            if generate_mask:
                masks.append(mask)
            if return_clean:
                clean_patterns.append(clean_pattern)

        # Warn if we couldn't find enough valid patterns
        if valid_pattern_count < num_patterns:
            import warnings
            warnings.warn(
                f"Only found {valid_pattern_count} valid patterns out of {num_patterns} requested after {attempt_count} "
                f"attempts.")

        # Prepare return dictionary
        results = {
            'patterns': patterns,
            'orientations': orientations,
            'material': material,
            'parameters': {
                'voltage': voltage,
                'camera_length': camera_length,
                'convergence_angle': convergence_angle,
                'sample_thickness': sample_thickness,
                'size': size,
                'sampling_mode': sampling_mode
            }
        }

        # Add additional data to results dictionary based on options
        if return_generators:
            results['generators'] = generators

        if generate_mask:
            results['masks'] = masks
            results['parameters']['mask_type'] = mask_type
            results['parameters']['intensity_threshold'] = intensity_threshold

        if return_clean:
            results['clean_patterns'] = clean_patterns
            results['parameters']['return_clean'] = True

        return results

    def demo1(self):
        """
        Generate and display 6 random diffraction patterns with different materials and parameters.
        Each pattern is displayed with information about the material, voltage, and sample thickness.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import random

        # Select 6 random materials from the available options
        materials = ['Al', 'Cu', 'Fe', 'Si', 'Au', 'Fe3O4', 'GaAs', 'SrTiO3', 'TiO2_rutile', 'ZnO']
        selected_materials = random.sample(materials, 6)

        # Create a figure with 2x3 subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()

        for i, material in enumerate(selected_materials):
            # Generate random parameters
            voltage = random.uniform(80, 300)
            thickness = random.uniform(10, 80)

            # Generate the pattern
            if material in ['Al', 'Cu', 'Fe', 'Si', 'Au']:
                # Simple material
                pattern, _, params_dict = self.generate_random(material=material)
            else:
                # Complex material
                zone_axis = random.choice([[0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1]])
                generator = self.from_material(material_name=material,
                                               zone_axis=zone_axis)
                pattern = generator.generate()
                params_dict = {
                    'material': material,
                    'voltage': voltage,
                    'sample_thickness': thickness,
                    'zone_axis_str': f"[{','.join(map(str, zone_axis))}]"
                }

            # Display the pattern
            axs[i].imshow(pattern, cmap='viridis')
            title = f"{params_dict['material']} {params_dict.get('zone_axis_str', '')}\n{params_dict['voltage']:.1f} kV, {params_dict['sample_thickness']:.1f} nm"
            axs[i].set_title(title)
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

    def demo2(self):
        """
        Generate and display 6 different orientations of Fe3O4 at 200kV.
        Shows how diffraction patterns change with crystal orientation.
        """
        import matplotlib.pyplot as plt

        # Generate 6 patterns for Fe3O4 with different orientations
        results = self.sample_orientations(
            material='Fe3O4',
            num_patterns=6,
            voltage=500,
            sample_thickness=40,
            sampling_mode='common'
        )

        patterns = results['patterns']
        orientations = results['orientations']

        # Create a figure with 2x3 subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()

        for i, (pattern, orientation) in enumerate(zip(patterns, orientations)):
            # Display the pattern
            axs[i].imshow(pattern, cmap='viridis')
            axs[i].set_title(f"Fe3O4 {orientation}\n200 kV, 40 nm")
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

    def generate_and_plot_with_mask(self):
        """
        Generate a random diffraction pattern with its binary mask and plot them side by side.
        Simple visualization without titles or additional styling.
        """
        import matplotlib.pyplot as plt

        # Generate random diffraction pattern with its binary mask
        diffraction_pattern, mask, _, _ = self.generate_random(
            generate_mask=True,
            mask_type="binary",
            intensity_threshold=0.01
        )

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot diffraction pattern
        ax1.imshow(diffraction_pattern, cmap='viridis')
        ax1.axis('off')

        # Plot binary mask
        ax2.imshow(mask, cmap='gray')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_pattern_comparison(self, pattern=None, clean_pattern=None, mask=None, title=None, save_path=None):
        """
        Visualize a comparison between a noisy and clean diffraction pattern, optionally with mask.

        This method allows for easy visualization of the effects of noise and background on diffraction patterns.

        Parameters:
        -----------
        pattern : 2D numpy array, optional
            Diffraction pattern with noise and background. If None, a new pattern will be generated.
        clean_pattern : 2D numpy array, optional
            Clean diffraction pattern without noise and background. If None but pattern is provided,
            this will be generated from the same parameters.
        mask : 2D numpy array, optional
            Mask identifying diffraction spots. If None, no mask will be shown.
        title : str, optional
            Title for the plot. If None, a default title will be used.
        save_path : str, optional
            Path to save the figure. If None, the figure will be displayed but not saved.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Generate patterns if not provided
        if pattern is None:
            if mask is not None and clean_pattern is not None:
                # If mask and clean_pattern are provided but pattern is not, generate all three
                pattern, mask, clean_pattern = self.generate(generate_mask=True, return_clean=True)
            elif clean_pattern is not None:
                # If only clean_pattern is provided, generate pattern and clean_pattern
                pattern, clean_pattern = self.generate(return_clean=True)
            elif mask is not None:
                # If only mask is provided, generate pattern and mask
                pattern, mask = self.generate(generate_mask=True)
            else:
                # If nothing is provided, generate pattern, mask, and clean_pattern
                pattern, mask, clean_pattern = self.generate(generate_mask=True, return_clean=True)
        elif clean_pattern is None:
            # If pattern is provided but clean_pattern is not, generate clean_pattern
            # We need to create a new generator with the same parameters but no noise
            generator_params = self.__dict__.copy()
            generator_params['noise_level'] = 0
            generator_params['background_variation'] = 0
            generator_params['detector_defects'] = False

            # Create a clean generator and generate a clean pattern
            clean_generator = type(self)(**generator_params)
            clean_pattern = clean_generator.generate()

        # Determine how many subplots we need
        if mask is not None and clean_pattern is not None:
            n_plots = 3
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        elif mask is not None or clean_pattern is not None:
            n_plots = 2
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        else:
            n_plots = 1
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            axs = [axs]  # Make axs a list so we can index into it

        # Plot the patterns
        im1 = axs[0].imshow(pattern, cmap='viridis')
        axs[0].set_title("Diffraction Pattern with Noise and Background")
        axs[0].axis('off')
        fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

        # Plot index for the next subplot
        plot_idx = 1

        # Plot clean pattern if available
        if clean_pattern is not None:
            im2 = axs[plot_idx].imshow(clean_pattern, cmap='viridis')
            axs[plot_idx].set_title("Clean Diffraction Pattern")
            axs[plot_idx].axis('off')
            fig.colorbar(im2, ax=axs[plot_idx], fraction=0.046, pad=0.04)
            plot_idx += 1

        # Plot mask if available
        if mask is not None:
            im3 = axs[plot_idx].imshow(mask, cmap='gray')
            axs[plot_idx].set_title("Diffraction Spot Mask")
            axs[plot_idx].axis('off')
            fig.colorbar(im3, ax=axs[plot_idx], fraction=0.046, pad=0.04)

        # Set the main title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            # Create a descriptive title based on the pattern parameters
            title_parts = []
            if hasattr(self, 'material'):
                title_parts.append(f"Material: {self.material}")
            elif hasattr(self, 'crystal_structure'):
                title_parts.append(f"Structure: {self.crystal_structure}")

            if hasattr(self, 'zone_axis'):
                za_str = '[' + ','.join(map(str, self.zone_axis)) + ']'
                title_parts.append(f"Zone Axis: {za_str}")

            if hasattr(self, 'voltage'):
                title_parts.append(f"{self.voltage} kV")

            if hasattr(self, 'sample_thickness'):
                title_parts.append(f"{self.sample_thickness} nm")

            if title_parts:
                fig.suptitle(', '.join(title_parts), fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig

    def demo_clean_vs_noisy(self, material=None, zone_axis=None):
        """
        Demonstrate the difference between clean and noisy diffraction patterns.

        This method generates and visualizes a set of patterns showing the effect of
        various noise and background settings on the diffraction pattern quality.

        Parameters:
        -----------
        material : str, optional
            Material to simulate. If None, a random material will be selected.
        zone_axis : list, optional
            Zone axis to use. If None, a random zone axis will be selected.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import random

        # Select a random material if not specified
        materials_list = [
            'Al', 'Cu', 'Fe', 'Ni', 'Au', 'Si', 'GaAs', 'Fe3O4', 'SrTiO3'
        ]

        if material is None:
            material = random.choice(materials_list)

        # Select a random zone axis if not specified
        common_zone_axes = [
            [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 0, 1]
        ]

        if zone_axis is None:
            zone_axis = random.choice(common_zone_axes)

        # Create a generator for the specified material
        if material in ['Al', 'Cu', 'Fe', 'Ni', 'Au', 'Si']:
            # Simple material
            simple_materials = {
                'Al': {'structure': 'fcc', 'lattice': 4.046, 'element': 'Al'},
                'Cu': {'structure': 'fcc', 'lattice': 3.615, 'element': 'Cu'},
                'Fe': {'structure': 'bcc', 'lattice': 2.866, 'element': 'Fe'},
                'Ni': {'structure': 'fcc', 'lattice': 3.524, 'element': 'Ni'},
                'Au': {'structure': 'fcc', 'lattice': 4.078, 'element': 'Au'},
                'Si': {'structure': 'diamond', 'lattice': 5.431, 'element': 'Si'},
            }

            structure = simple_materials[material]['structure']
            a = simple_materials[material]['lattice']
            element = simple_materials[material]['element']

            params = {
                'size': 256,
                'crystal_structure': structure,
                'lattice_params': {'a': a, 'b': a, 'c': a, 'alpha': 90, 'beta': 90, 'gamma': 90},
                'elements': [element],
                'atomic_positions': [{'element': element, 'position': [0, 0, 0]}],
                'voltage': 200,
                'camera_length': 300,
                'convergence_angle': 1.5,
                'zone_axis': zone_axis,
                'sample_thickness': 40,
            }

            # Special case for diamond structure (Si)
            if structure == 'diamond':
                params['atomic_positions'] = [
                    {'element': element, 'position': [0, 0, 0]},
                    {'element': element, 'position': [0.25, 0.25, 0.25]},
                ]

            generator = type(self)(**params)

        else:
            # Complex material
            generator = type(self).from_material(
                material_name=material,
                zone_axis=zone_axis,
                voltage=200,
                camera_length=300,
                convergence_angle=1.5,
                sample_thickness=40
            )

        # Generate the clean pattern with no noise or background
        clean_params = generator.__dict__.copy()
        clean_params['noise_level'] = 0
        clean_params['background_variation'] = 0
        clean_params['detector_defects'] = False
        clean_generator = type(self)(**clean_params)
        clean_pattern = clean_generator.generate()

        # Generate patterns with increasing levels of noise and background
        noise_levels = [0.005, 0.01, 0.02, 0.03]
        bg_levels = [0.005, 0.01, 0.02, 0.03]

        # Create a figure with subplots
        fig, axs = plt.subplots(len(noise_levels), len(bg_levels) + 1, figsize=(15, 12))

        # Add the clean pattern to the first column
        for i in range(len(noise_levels)):
            axs[i, 0].imshow(clean_pattern, cmap='viridis')
            axs[i, 0].set_title("Clean Pattern" if i == 0 else "")
            axs[i, 0].axis('off')

        # Generate and display patterns with different noise and background levels
        for i, noise in enumerate(noise_levels):
            for j, bg in enumerate(bg_levels):
                # Create a generator with the specified noise and background
                noisy_params = generator.__dict__.copy()
                noisy_params['noise_level'] = noise
                noisy_params['background_variation'] = bg
                noisy_generator = type(self)(**noisy_params)

                # Generate the pattern
                noisy_pattern = noisy_generator.generate()

                # Display the pattern
                axs[i, j + 1].imshow(noisy_pattern, cmap='viridis')
                if i == 0:
                    axs[i, j + 1].set_title(f"BG: {bg}")
                if j == 0:
                    axs[i, j + 1].set_ylabel(f"Noise: {noise}")
                axs[i, j + 1].axis('off')

        # Set the main title
        za_str = '[' + ','.join(map(str, zone_axis)) + ']'
        fig.suptitle(f"{material} {za_str} - Effect of Noise and Background on Diffraction Pattern", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return fig

    def generate_clean_pattern_dataset(cls, materials=None, num_patterns_per_material=10, size=256,
                                       save_dir=None, sampling_mode='common', generate_mask=True,
                                       return_results=False):
        """
        Generate a dataset of clean and noisy diffraction patterns for multiple materials.

        This method creates a set of diffraction patterns for each specified material,
        including both the standard noisy patterns and clean patterns without noise/background.

        Parameters:
        -----------
        materials : list, optional
            List of materials to include. If None, a default set will be used.
        num_patterns_per_material : int
            Number of different orientations to generate for each material
        size : int
            Size of the output images (size x size pixels)
        save_dir : str, optional
            Directory to save generated patterns. If None, patterns are not saved.
        sampling_mode : str
            Mode for orientation sampling ('common', 'uniform', 'powder', or 'custom')
        generate_mask : bool
            If True, also generates masks identifying diffraction peaks
        return_results : bool
            If True, returns the generated patterns and metadata

        Returns:
        --------
        dict
            Dictionary containing the generated patterns and metadata (if return_results=True)
        """
        import os
        import time

        # Default materials list if none provided
        if materials is None:
            materials = [
                'Al', 'Cu', 'Fe', 'Ni', 'Au', 'Si', 'Ge',
                'Fe3O4', 'TiO2_rutile', 'SrTiO3', 'GaAs', 'ZnO'
            ]

        # Create save directory if specified
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Dictionary to store results if requested
        if return_results:
            results = {
                'materials': materials,
                'patterns': [],
                'clean_patterns': [],
                'masks': [] if generate_mask else None,
                'orientations': [],
                'metadata': []
            }

        # Generate patterns for each material
        for material in materials:
            print(f"Generating patterns for {material}...")

            # Record start time for this material
            start_time = time.time()

            # Create a subfolder for this material if saving
            material_dir = None
            if save_dir:
                material_dir = os.path.join(save_dir, material)
                if not os.path.exists(material_dir):
                    os.makedirs(material_dir)

            # Generate the patterns for this material
            material_results = cls.sample_orientations(
                material=material,
                num_patterns=num_patterns_per_material,
                size=size,
                sampling_mode=sampling_mode,
                generate_mask=generate_mask,
                return_clean=True,
                save_dir=material_dir
            )

            # Store results if requested
            if return_results:
                results['patterns'].extend(material_results['patterns'])
                results['clean_patterns'].extend(material_results['clean_patterns'])
                if generate_mask:
                    results['masks'].extend(material_results['masks'])

                # Add material and orientation info to each pattern
                for i, orientation in enumerate(material_results['orientations']):
                    results['orientations'].append(orientation)
                    results['metadata'].append({
                        'material': material,
                        'orientation': orientation,
                        'parameters': material_results['parameters']
                    })

            # Report completion and timing
            elapsed_time = time.time() - start_time
            print(f"  Completed {num_patterns_per_material} patterns for {material} in {elapsed_time:.2f} seconds")

        if return_results:
            return results
        else:
            print("Dataset generation complete!")

    def generate_and_plot_with_clean(self):
        """
        Generate a random diffraction pattern with its binary mask and clean version, then plot them side by side.
        This method demonstrates the visual difference between the noisy pattern, clean pattern, and mask.
        """
        import matplotlib.pyplot as plt

        # Generate random diffraction pattern with its binary mask and clean version
        diffraction_pattern, mask, clean_pattern, _, params_dict = self.generate_random(
            generate_mask=True,
            mask_type="binary",
            intensity_threshold=0.01,
            return_clean=True
        )

        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot diffraction pattern with noise and background
        im1 = ax1.imshow(diffraction_pattern, cmap='viridis')
        ax1.set_title('Diffraction Pattern\n(with noise & background)')
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Plot clean diffraction pattern
        im2 = ax2.imshow(clean_pattern, cmap='viridis')
        ax2.set_title('Clean Diffraction Pattern\n(no noise or background)')
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Plot binary mask
        im3 = ax3.imshow(mask, cmap='gray')
        ax3.set_title('Binary Mask\n(diffraction spot positions)')
        ax3.axis('off')
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Create a descriptive title based on the parameters
        material = params_dict.get('material', 'Unknown')
        zone_axis = params_dict.get('zone_axis_str', '[Unknown]')
        voltage = params_dict.get('voltage', 'Unknown')
        thickness = params_dict.get('sample_thickness', 'Unknown')

        fig.suptitle(f'Material: {material}, Zone Axis: {zone_axis}, {voltage} kV, {thickness} nm', fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def create_diffraction_animation(self, material=None, zone_axis=None, duration=10, fps=30,
                                     num_electrons=50, save_path=None, electron_speed=1.0,
                                     crystal_size=1.0, fig_size=(10, 8), dpi=100):
        """
        Create a 3D animation showing electrons traveling through a crystal and forming a diffraction pattern.

        The animation starts from a side view of electrons entering the crystal, then gradually rotates
        to a top-down view showing the resulting diffraction pattern.

        Parameters:
        -----------
        material : str, optional
            Name of the material to simulate. If None, a random material will be used.
        zone_axis : list, optional
            Zone axis direction [h,k,l]. If None, a default will be used.
        duration : float
            Duration of the animation in seconds
        fps : int
            Frames per second
        num_electrons : int
            Number of electron traces to show in the animation
        save_path : str, optional
            Path to save the animation (e.g., 'animation.mp4'). If None, will display interactively.
        electron_speed : float
            Relative speed of electrons (higher values mean faster electrons)
        crystal_size : float
            Size of the crystal for visualization
        fig_size : tuple
            Figure size in inches
        dpi : int
            DPI for the animation

        Returns:
        --------
        matplotlib.animation.Animation
            The created animation object
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D
        import random
        from scipy.spatial.transform import Rotation
        import matplotlib.colors as mcolors
        from matplotlib.patches import Circle
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

        # Set random seed for reproducibility
        np.random.seed(42)

        # Select a random material if not specified
        materials_list = [
            'Al', 'Cu', 'Fe', 'Ni', 'Au', 'Si', 'GaAs', 'Fe3O4', 'SrTiO3'
        ]
        if material is None:
            material = random.choice(materials_list)

        # Define common zone axes
        common_zone_axes = [
            [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 2]
        ]
        if zone_axis is None:
            zone_axis = random.choice(common_zone_axes)

        # Define crystal dimensions
        crystal_length = crystal_size
        crystal_width = crystal_size * 0.8
        crystal_height = crystal_size * 0.8

        # For displaying the diffraction pattern
        detector_distance = crystal_length * 2  # Distance from crystal center to detector
        detector_size = crystal_width * 3  # Size of the detector plane

        # Generate the diffraction pattern for the selected material and zone axis
        if material in ['Al', 'Cu', 'Fe', 'Ni', 'Au', 'Si']:
            pattern, generator, params = self.generate_random(material=material)
        else:
            generator = self.from_material(
                material_name=material,
                zone_axis=zone_axis
            )
            pattern = generator.generate()

        # Generate allowed reflections to get their positions in k-space
        reflections = generator._generate_reflections(max_index=4)

        # Filter to include only significant reflections
        visible_reflections = []
        for reflection in reflections:
            if reflection['intensity'] > 0.01:
                visible_reflections.append(reflection)

        # Add direct beam (000) if not already present
        direct_beam = True
        for ref in visible_reflections:
            if ref['h'] == 0 and ref['k'] == 0 and ref['l'] == 0:
                direct_beam = False
                break

        if direct_beam:
            visible_reflections.append({
                'h': 0, 'k': 0, 'l': 0,
                'intensity': 1.0,
                'g_magnitude': 0.0
            })

        # Limit reflections for clarity
        max_reflections = 30
        visible_reflections = sorted(visible_reflections, key=lambda x: x['intensity'], reverse=True)[:max_reflections]

        # Calculate scale factor for pixel coordinates (similar to what's used in generate method)
        scale_factor = detector_size / 6  # Adjust this to match scaling in your class

        # Calculate zone axis unit vectors for projection (copied from your generate method)
        if hasattr(generator, 'zone_axis'):
            zone_axis = generator.zone_axis
        # Find unit vectors perpendicular to zone axis
        if abs(zone_axis[2]) < 0.9:
            u1 = np.cross(zone_axis, [0, 0, 1])
        else:
            u1 = np.cross(zone_axis, [1, 0, 0])
        u1 = u1 / np.linalg.norm(u1)

        u2 = np.cross(zone_axis, u1)
        u2 = u2 / np.linalg.norm(u2)

        # Map reflection positions to detector coordinates using same math as in generator.generate()
        reflection_positions = []
        reflection_indices = []
        reflection_intensities = []
        reflection_colors = []

        # Direct beam at center
        reflection_positions.append((0, 0, detector_distance))
        reflection_indices.append((0, 0, 0))
        reflection_intensities.append(1.0)
        reflection_colors.append(plt.cm.viridis(1.0))

        # Generate diffraction spot positions on detector using the same calculation as in your class
        for reflection in visible_reflections:
            h, k, l = reflection['h'], reflection['k'], reflection['l']
            intensity = reflection['intensity']

            # Skip direct beam (already added)
            if h == 0 and k == 0 and l == 0:
                continue

            # Project g-vector onto diffraction plane (copied from your generate method)
            if hasattr(generator, 'a_recip') and hasattr(generator, 'b_recip') and hasattr(generator, 'c_recip'):
                g_vector = [h * generator.a_recip, k * generator.b_recip, l * generator.c_recip]
            else:
                # Fallback if attributes aren't available
                g_vector = [h, k, l]

            proj_u1 = np.dot(g_vector, u1)
            proj_u2 = np.dot(g_vector, u2)

            # Convert to pixel coordinates (using method similar to your generate)
            spot_x = proj_u1 * scale_factor
            spot_y = proj_u2 * scale_factor

            # Store position and properties
            reflection_positions.append((spot_x, spot_y, detector_distance))
            reflection_indices.append((h, k, l))
            reflection_intensities.append(intensity)

            # Determine color based on intensity
            norm_intensity = min(1.0, intensity * 2)
            reflection_colors.append(plt.cm.viridis(norm_intensity))

        # Setup the figure and 3D axis
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Define crystal dimensions
        crystal_length = crystal_size
        crystal_width = crystal_size * 0.8
        crystal_height = crystal_size * 0.8

        # Define the crystal as a rectangular prism
        crystal_x = np.array([-crystal_width / 2, crystal_width / 2, crystal_width / 2, -crystal_width / 2,
                              -crystal_width / 2, crystal_width / 2, crystal_width / 2, -crystal_width / 2])
        crystal_y = np.array([-crystal_height / 2, -crystal_height / 2, crystal_height / 2, crystal_height / 2,
                              -crystal_height / 2, -crystal_height / 2, crystal_height / 2, crystal_height / 2])
        crystal_z = np.array([-crystal_length / 2, -crystal_length / 2, -crystal_length / 2, -crystal_length / 2,
                              crystal_length / 2, crystal_length / 2, crystal_length / 2, crystal_length / 2])

        # Define crystal faces (list of indices for vertices)
        faces = [
            [0, 1, 2, 3],  # Front face
            [4, 5, 6, 7],  # Back face
            [0, 3, 7, 4],  # Left face
            [1, 2, 6, 5],  # Right face
            [0, 1, 5, 4],  # Bottom face
            [3, 2, 6, 7]  # Top face
        ]

        # For displaying the diffraction pattern
        detector_distance = crystal_length * 2  # Distance from crystal center to detector
        detector_size = crystal_width * 3  # Size of the detector plane

        # Create a grid for the detector plane (diffraction pattern)
        detector_x, detector_y = np.meshgrid(
            np.linspace(-detector_size / 2, detector_size / 2, 50),
            np.linspace(-detector_size / 2, detector_size / 2, 50)
        )
        detector_z = np.ones_like(detector_x) * detector_distance

        # Scale the diffraction pattern to fit nicely on the detector
        scaled_pattern = np.zeros((50, 50))
        pattern_small = np.array(pattern)

        # Resize pattern to 50x50 for the detector
        from scipy.ndimage import zoom
        try:
            zoom_factor = 50 / pattern_small.shape[0]
            scaled_pattern = zoom(pattern_small, zoom_factor)
        except Exception as e:
            print(f"Warning: Error resizing pattern: {e}")
            scaled_pattern = np.zeros((50, 50))

        # Extract actual diffraction peak locations from the pattern
        reflection_positions = []
        reflection_indices = []
        reflection_intensities = []
        reflection_colors = []

        # Detect peaks in the diffraction pattern
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max

        # Find peaks in the pattern
        try:
            # Normalize pattern for peak detection
            pattern_norm = pattern / np.max(pattern)

            # Find local maxima (peaks)
            coordinates = peak_local_max(pattern_norm, min_distance=10, threshold_abs=0.1)

            print(f"Found {len(coordinates)} diffraction peaks in the pattern")

            # If no peaks found or too few, use calculated positions as fallback
            if len(coordinates) < 3:
                raise ValueError("Too few peaks detected")

            # Convert peak coordinates to detector positions
            center_x = pattern.shape[1] // 2
            center_y = pattern.shape[0] // 2

            for coord in coordinates:
                y, x = coord
                # Calculate position relative to center
                rel_x = (x - center_x) / center_x  # -1 to 1
                rel_y = (y - center_y) / center_y  # -1 to 1

                # Scale to detector size
                det_x = rel_x * detector_size / 2
                det_y = rel_y * detector_size / 2

                # Get intensity at this peak
                intensity = pattern_norm[y, x]

                # Store data about this peak
                reflection_positions.append((det_x, det_y, detector_distance))
                reflection_intensities.append(intensity)

                # Assign closest Miller indices (h,k,l) as estimation
                closest_idx = 0
                min_dist = float('inf')
                for idx, ref in enumerate(visible_reflections):
                    h, k, l = ref['h'], ref['k'], ref['l']
                    # Skip direct beam for mapping
                    if h == 0 and k == 0 and l == 0 and not (abs(rel_x) < 0.1 and abs(rel_y) < 0.1):
                        continue
                    # If this is near center, it's likely the direct beam
                    if abs(rel_x) < 0.1 and abs(rel_y) < 0.1:
                        closest_idx = [idx for idx, ref in enumerate(visible_reflections)
                                       if ref['h'] == 0 and ref['k'] == 0 and ref['l'] == 0][0]
                        break

                    # Calculate distance between detector position and calculated position
                    calc_x = h * detector_size / 8
                    calc_y = k * detector_size / 8
                    dist = np.sqrt((det_x - calc_x) ** 2 + (det_y - calc_y) ** 2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx

                # Store the closest Miller indices
                h, k, l = visible_reflections[closest_idx]['h'], visible_reflections[closest_idx]['k'], \
                visible_reflections[closest_idx]['l']
                reflection_indices.append((h, k, l))

                # Determine color based on intensity
                norm_intensity = min(1.0, intensity * 2)  # Scale intensity for better visualization
                reflection_colors.append(plt.cm.viridis(norm_intensity))

        except Exception as e:
            print(f"Warning: Using calculated positions due to error: {e}")
            # Fallback to calculated positions based on Miller indices
            scale_factor = detector_size / 8

            for idx, reflection in enumerate(visible_reflections):
                h, k, l = reflection['h'], reflection['k'], reflection['l']
                intensity = reflection['intensity']

                # Direct beam at center
                if h == 0 and k == 0 and l == 0:
                    det_x, det_y = 0, 0
                else:
                    # Map hkl to detector position with scaling
                    det_x = h * scale_factor * 0.5
                    det_y = k * scale_factor * 0.5

                # Store position and properties
                reflection_positions.append((det_x, det_y, detector_distance))
                reflection_indices.append((h, k, l))
                reflection_intensities.append(intensity)

                # Determine color based on intensity
                norm_intensity = min(1.0, intensity * 2)
                reflection_colors.append(plt.cm.viridis(norm_intensity))

        # Make sure we have direct beam (000) as one of the spots
        has_direct_beam = False
        for indices in reflection_indices:
            if indices == (0, 0, 0):
                has_direct_beam = True
                break

        if not has_direct_beam:
            reflection_positions.append((0, 0, detector_distance))
            reflection_indices.append((0, 0, 0))
            reflection_intensities.append(1.0)
            reflection_colors.append(plt.cm.viridis(1.0))

        # Initialize electron positions: starting before the crystal in a grid
        def initialize_electrons(num_electrons):
            # Grid size for the array of electrons
            grid_size = int(np.ceil(np.sqrt(num_electrons)))
            positions = []

            # Initial position is a grid before the crystal
            source_distance = crystal_length * 1.5

            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) < num_electrons:
                        # Small random offset for visual interest
                        offset_x = (np.random.random() - 0.5) * 0.1 * crystal_width
                        offset_y = (np.random.random() - 0.5) * 0.1 * crystal_height

                        # Position in a grid pattern
                        x = (i / (grid_size - 1 or 1) - 0.5) * crystal_width * 0.8 + offset_x
                        y = (j / (grid_size - 1 or 1) - 0.5) * crystal_height * 0.8 + offset_y
                        z = -source_distance

                        positions.append((x, y, z))

            return np.array(positions)

        # Initialize electron positions
        electron_positions = initialize_electrons(num_electrons)

        # Assign each electron to a specific diffraction spot
        electron_destinations = []
        electron_paths = [[] for _ in range(num_electrons)]
        electron_colors = []

        # Record initial positions
        for i, pos in enumerate(electron_positions):
            electron_paths[i].append(pos)

        # Assign electrons to diffraction spots with probability weighted by intensity
        total_intensity = sum(reflection_intensities)
        probabilities = [intensity / total_intensity for intensity in reflection_intensities]

        # Assign destinations with direct beam getting more electrons
        for i in range(num_electrons):
            # Higher probability of direct beam
            if np.random.random() < 0.4:
                # Find the direct beam (000)
                for idx, (h, k, l) in enumerate(reflection_indices):
                    if h == 0 and k == 0 and l == 0:
                        destination_idx = idx
                        break
            else:
                # Weighted random choice for other spots
                destination_idx = np.random.choice(len(reflection_positions), p=probabilities)

            electron_destinations.append(destination_idx)
            electron_colors.append(reflection_colors[destination_idx])

        # Animation state
        anim_state = {
            'frame': 0,
            'total_frames': int(duration * fps)
        }

        # Storage for paths
        full_paths = [[] for _ in range(num_electrons)]

        # Function to update the animation
        def update(frame):
            # Clear the axis
            ax.clear()

            # Calculate progress as a value from 0 to 1
            progress = frame / anim_state['total_frames']

            # Update electron positions
            for i in range(num_electrons):
                # Current position
                x, y, z = electron_positions[i]

                # Get destination spot
                dest_idx = electron_destinations[i]
                dest_x, dest_y, dest_z = reflection_positions[dest_idx]
                h, k, l = reflection_indices[dest_idx]

                # Electrons move in stages
                if z < -crystal_length / 2:
                    # Before crystal: straight path
                    new_z = z + crystal_length * 0.15 * electron_speed
                    new_x, new_y = x, y
                elif z < crystal_length / 2:
                    # Inside crystal: gradual shift toward destination
                    # Calculate fraction through crystal
                    crystal_progress = (z - (-crystal_length / 2)) / crystal_length

                    # Gradually shift toward final trajectory
                    if h == 0 and k == 0 and l == 0:
                        # Direct beam continues straight
                        shift_x, shift_y = 0, 0
                    else:
                        # Shift toward diffraction spot direction
                        angle_x = np.arctan2(dest_x, detector_distance) * crystal_progress * 2
                        angle_y = np.arctan2(dest_y, detector_distance) * crystal_progress * 2
                        shift_x = angle_x * 0.1
                        shift_y = angle_y * 0.1

                    # Add random scattering inside crystal with decreasing magnitude
                    scatter_x = (np.random.random() - 0.5) * 0.02 * (1 - crystal_progress)
                    scatter_y = (np.random.random() - 0.5) * 0.02 * (1 - crystal_progress)

                    # New position
                    new_z = z + crystal_length * 0.1 * electron_speed
                    new_x = x + shift_x + scatter_x
                    new_y = y + shift_y + scatter_y
                else:
                    # After crystal: linear path to destination
                    # Calculate remaining travel distance to detector
                    remaining_z = detector_distance - z
                    if remaining_z <= 0:
                        # At or past detector, stop
                        new_x, new_y, new_z = dest_x, dest_y, dest_z
                    else:
                        # Linear interpolation to destination
                        frac_remain = (detector_distance - z) / (detector_distance - crystal_length / 2)
                        new_z = z + remaining_z * 0.1 * electron_speed

                        # Shift gradually toward the exact spot
                        new_x = x + (dest_x - x) * 0.1
                        new_y = y + (dest_y - y) * 0.1

                # Update position
                electron_positions[i] = (new_x, new_y, new_z)
                electron_paths[i].append((new_x, new_y, new_z))

                # Limit path history for visual clarity
                tail_length = min(30, int(frame * 1.5) + 5)  # Growing tail length with time
                if len(electron_paths[i]) > tail_length:
                    electron_paths[i] = electron_paths[i][-tail_length:]

                # Save full path for reference
                full_paths[i].append((new_x, new_y, new_z))

            # Camera angles
            # Start with side view (10°) and rotate to top-down view (80°)
            if progress < 0.7:
                # First 70%: rotate from side (10°) toward top-down (80°)
                elev = 10 + progress * 70  # Start low (side view) and rotate upward
                azim = -90  # Side view (looking along X axis)
            else:
                # Last 30%: keep top-down view but rotate slightly for better perspective
                elev = 80  # Top-down view
                azim = -90 + (progress - 0.7) * 20  # Slight rotation

            # Set camera angle
            ax.view_init(elev=elev, azim=azim)

            # Draw crystal
            # Create 3D polygons for each face
            polygons = []
            for face in faces:
                vertices = []
                for i in face:
                    vertices.append([crystal_x[i], crystal_y[i], crystal_z[i]])
                polygons.append(vertices)

            # Add crystal faces as translucent polygons
            crystal_faces = Poly3DCollection(polygons, alpha=0.2, facecolor='lightblue', edgecolor='darkblue',
                                             linewidth=0.5)
            ax.add_collection3d(crystal_faces)

            # Draw electron paths
            for i in range(num_electrons):
                if len(electron_paths[i]) > 1:
                    path = np.array(electron_paths[i])
                    # Draw with color matching destination spot
                    ax.plot(path[:, 0], path[:, 1], path[:, 2], '-', color=electron_colors[i], alpha=0.8, lw=1.0)

            # Draw the detector plane if we're far enough into the animation
            if progress > 0.4:
                # Fade in the detector
                fade_in = min(1.0, (progress - 0.4) / 0.2)

                # Draw detector frame
                detector_frame_x = [-detector_size / 2, detector_size / 2, detector_size / 2, -detector_size / 2,
                                    -detector_size / 2]
                detector_frame_y = [-detector_size / 2, -detector_size / 2, detector_size / 2, detector_size / 2,
                                    -detector_size / 2]
                detector_frame_z = [detector_distance] * 5

                ax.plot(detector_frame_x, detector_frame_y, detector_frame_z, 'k-', lw=1, alpha=fade_in)

                # Draw detector as a semi-transparent surface
                try:
                    detector_surface = ax.plot_surface(
                        detector_x, detector_y, detector_z,
                        color='gray', alpha=0.1 * fade_in,
                        rstride=4, cstride=4
                    )
                except Exception:
                    # Fallback if plot_surface fails
                    pass

            # Draw diffraction spots on detector
            if progress > 0.5:
                spot_fade_in = min(1.0, (progress - 0.5) / 0.3)

                # Draw each diffraction spot
                for idx, (pos, intensity, color) in enumerate(
                        zip(reflection_positions, reflection_intensities, reflection_colors)):
                    x, y, z = pos

                    # Spot size based on intensity and animation progress
                    spot_size = 0.05 + intensity * 0.2  # Base size
                    spot_size *= spot_fade_in  # Fade in size

                    # Plot spot as a scatter point
                    ax.scatter(x, y, z, s=500 * spot_size,
                               c=[color], alpha=spot_fade_in,
                               edgecolor='white', linewidth=0.5)

            # Overlay diffraction pattern as image texture if we're in top-down view
            if progress > 0.7:
                pattern_fade_in = min(1.0, (progress - 0.7) / 0.3)

                # Only attempt to show pattern if we're using a compatible view angle
                if elev > 60:  # Only show when camera is near top-down
                    try:
                        # Create a meshgrid specifically for the diffraction pattern
                        # Use higher resolution for better quality
                        pattern_x, pattern_y = np.meshgrid(
                            np.linspace(-detector_size / 2, detector_size / 2, 100),
                            np.linspace(-detector_size / 2, detector_size / 2, 100)
                        )
                        pattern_z = np.ones_like(pattern_x) * (detector_distance + 0.01)  # Slightly above detector

                        # Use the original pattern for better quality
                        # Resize to fit the meshgrid
                        from scipy.ndimage import zoom
                        pattern_resized = zoom(pattern, (100 / pattern.shape[0], 100 / pattern.shape[1]))

                        # Normalize the pattern for visualization
                        pattern_normalized = pattern_resized / np.max(pattern_resized)

                        # Use alpha channel to make background transparent
                        pattern_rgba = plt.cm.viridis(pattern_normalized)

                        # Make low-intensity parts more transparent
                        for i in range(pattern_rgba.shape[0]):
                            for j in range(pattern_rgba.shape[1]):
                                pattern_rgba[i, j, 3] = pattern_normalized[i, j] ** 0.5 * pattern_fade_in

                        pattern_surface = ax.plot_surface(
                            pattern_x, pattern_y, pattern_z,
                            facecolors=pattern_rgba,
                            alpha=1.0,  # Use per-pixel alpha from pattern_rgba
                            rstride=1, cstride=1
                        )
                    except Exception as e:
                        print(f"Error displaying diffraction pattern: {e}")
                        # Fallback to simpler representation
                        pass

            # Set axis limits
            ax.set_xlim(-detector_size / 2 * 1.2, detector_size / 2 * 1.2)
            ax.set_ylim(-detector_size / 2 * 1.2, detector_size / 2 * 1.2)
            ax.set_zlim(-crystal_length * 1.5, detector_distance * 1.2)

            # Add axis labels and title
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)

            # Format zone axis for title
            zone_axis_str = f"[{zone_axis[0]},{zone_axis[1]},{zone_axis[2]}]"
            ax.set_title(f"{material} Crystal - Zone Axis {zone_axis_str}\nElectron Diffraction Animation", fontsize=10)

            # Increment the frame counter
            anim_state['frame'] += 1

            # Return nothing for blit=False
            return

        # Create the animation
        frames = int(duration * fps)
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

        # Save or display
        if save_path:
            # Ensure correct extension
            if not (save_path.endswith('.mp4') or save_path.endswith('.gif')):
                save_path += '.mp4'

            if save_path.endswith('.mp4'):
                # For MP4, need to specify codec
                try:
                    animation.save(save_path, writer='ffmpeg', dpi=dpi)
                except:
                    # Fallback to pillow if ffmpeg isn't available
                    if save_path.endswith('.mp4'):
                        save_path = save_path[:-4] + '.gif'
                    animation.save(save_path, writer='pillow', dpi=dpi)
            else:
                # For GIF
                animation.save(save_path, writer='pillow', dpi=dpi)

            print(f"Animation saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()

        return animation


