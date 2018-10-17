# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

"""
utils to support SubpixelrefinementGenerator
"""

import numpy as np
from skimage.feature import register_translation
from skimage import draw
from skimage.transform import rescale

def get_experimental_square(z,vector,square_size,upsample_factor):
    """
    'Cuts out' a region around a given diffraction vector and returns an upsampled copy.

    Args
    ----

    z: np.array() - Single diffraction pattern
    vector: np.array() - Single vector to be cut out, in pixels (int) [x,y] with top left as [0,0]
    square_size: The length of one side of the bounding square (must be even)
    upsample_factor: The factor by which to up-sample (must be even)

    Returns
    -------

    square: np.array of size (L,L) where L = square_size*upsample_factor

    """


    cx,cy,half_ss = vector[0], vector[1], int(square_size/2)
    _z = z[cx-half_ss:cx+half_ss,cy-half_ss:cy+half_ss]
    _z = rescale(_z,upsample_factor)

    return _z

def get_simulated_disc(square_size,disc_radius,upsample_factor):
    """
    Create a uniform disc for correlating with the experimental square

    Args
    ----

    square size: int (even) - size of the bounding box
    disc_radius: int - radius of the disc
    upsample_factor: int - The factor by which to upsample (must be even)
    """

    ss = int(square_size)#*upsample_factor) #upsample square size
    arr = np.zeros((ss,ss))
    rr, cc = draw.circle(int(ss/2), int(ss/2), radius=disc_radius, shape=arr.shape) #is the thin disc a good idea
    arr[rr, cc] = 1
    arr = rescale(arr,upsample_factor)
    return arr

def _conventional_xc(exp_disc,sim_disc):
    """
    Takes two images of disc and finds the shift between them using conventional cross correlation

    Args
    ----
    exp_disc - np.array() - A numpy array of the "experimental" disc
    sim_disc - np.array() - A numpy array of the disc used a template

    Return
    -------
    shifts - Pixel shifts required to register the two images
    """

    shifts,error,_ = register_translation(exp_disc,sim_disc)
    return shifts
