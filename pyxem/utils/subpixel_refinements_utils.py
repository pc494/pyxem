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
from skimage.filters import sobel

def get_experimental_square(z,vector,square_size,upsample_factor):
    """
    'Cuts out' a a region around a given diffraction vector and returns and upsample_factor
    copy.

    Args
    ----

    z: np.array() - Single diffraction pattern
    vector: np.array() - Single vector to be cut out
    square_size: The length of one side of the bounding square_size (must be even)
    upsample_factor: The factor by which to up-sample (must be even)

    """
    cx,cy,half_ss = vector[0], vector[1], int(square_size/2)
    _z = z[cx-half_ss:cx+half_ss,cy-half_ss:cy+half_ss]
    _z = rescale(_z,upsample_factor)
    return _z

def get_simulated_disc(square_size,disc_radius,up_sample_factor):
    upsss = int(square_size*upsample_factor) #upsample square size
    arr = np.zeros((upsss,upsss))
    rr, cc = draw.circle(int(upsss/2), int(upsss/2), radius=disc_radius*upsample_factor, shape=arr.shape)
    arr[rr, cc] = 1
    return arr

def _sobel_filtered_xc(exp_disc,sim_disc):
    # sobel filter exp_disc
    # register translation
    # return answer
    pass

def _conventional_xc():
    pass
