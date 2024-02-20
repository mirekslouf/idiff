'''
Package: IDIFF
--------------
Improve DIFFractograms = improve quality of 2D diffraction patterns.

* idiff.bcorr  = background correction/subtraction
* idiff.ncorr  = noise correction/reduction 
* idiff.deconv = advanced deconvolution methods (beyond RL)
* idiff.psf    = functions to estimate 2D-PSF = 2D point spread function       
'''

__version__ = "0.1"

import idiff.bcorr
import idiff.deconv
import idiff.ncorr
import idiff.psf
