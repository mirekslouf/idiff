'''
Package: IDIFF
--------------
IDIFF = functions to Improve DIFFraction patterns.

* idiff.bkg2d  = background subtraction for 2D-arrays/images
* idiff.deconv = advanced deconvolution methods (beyond RL)
* idiff.psf    = functions to estimate 2D-PSF = 2D point spread function       

The functions are employed in our diffraction-related packages:

* STEMDIFF = convert 4D-STEM datasets to 2D-powder diffractograms
* EDIFF    = convert 2D-diffratograms to 1D-profiles and compare with theory. 
'''

__version__ = "0.1.1"

import idiff.bkg2d
import idiff.deconv
import idiff.psf
