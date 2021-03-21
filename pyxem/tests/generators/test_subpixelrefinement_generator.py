# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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

import pytest
import numpy as np

from skimage import draw

from pyxem.generators import SubpixelrefinementGenerator
from pyxem.generators.subpixelrefinement_generator import (
    get_simulated_disc,
    get_experimental_square,
)
from pyxem.signals import DiffractionVectors, ElectronDiffraction2D


@pytest.fixture()
def exp_disc():
    ss, disc_radius, upsample_factor = int(60), 6, 10

    arr = np.zeros((ss, ss))
    rr, cc = draw.circle(
        int(ss / 2) + 20, int(ss / 2) - 10, radius=disc_radius, shape=arr.shape
    )
    arr[rr, cc] = 1
    return arr


@pytest.mark.filterwarnings("ignore::UserWarning")  # various skimage warnings
def test_experimental_square_size(exp_disc):
    square = get_experimental_square(exp_disc, [17, 19], 6)
    assert square.shape[0] == int(6)
    assert square.shape[1] == int(6)


def test_failure_for_non_even_entry_to_get_simulated_disc():
    with pytest.raises(ValueError, match="'square_size' must be an even number"):
        _ = get_simulated_disc(61, 5)


def test_failure_for_non_even_errors_get_experimental_square(exp_disc):
    with pytest.raises(ValueError, match="'square_size' must be an even number"):
        _ = get_experimental_square(exp_disc, [17, 19], 7)


class Test_init_xfails:
    def test_out_of_range_vectors_numpy(self):
        """Test that putting vectors that lie outside of the
        diffraction patterns raises a ValueError"""
        vector = np.array([[1, -100]])
        dp = ElectronDiffraction2D(np.ones((20, 20)))

        with pytest.raises(
            ValueError,
            match="Some of your vectors do not lie within your diffraction pattern",
        ):
            _ = SubpixelrefinementGenerator(dp, vector)

    def test_out_of_range_vectors_DiffractionVectors(self):
        """Test that putting vectors that lie outside of the
        diffraction patterns raises a ValueError"""
        vectors = DiffractionVectors(np.array([[1, -100]]))
        dp = ElectronDiffraction2D(np.ones((20, 20)))

        with pytest.raises(
            ValueError,
            match="Some of your vectors do not lie within your diffraction pattern",
        ):
            _ = SubpixelrefinementGenerator(dp, vectors)

    def test_wrong_navigation_dimensions(self):
        """Tests that navigation dimensions must be appropriate too."""
        dp = ElectronDiffraction2D(np.zeros((2, 2, 8, 8)))
        vectors = DiffractionVectors(np.zeros((1, 2)))
        dp.axes_manager.set_signal_dimension(2)
        vectors.axes_manager.set_signal_dimension(0)

        # Note - uses regex via re.search()
        with pytest.raises(
            ValueError,
            match=r"Vectors with shape .* must have the same navigation shape as .*",
        ):
            _ = SubpixelrefinementGenerator(dp, vectors)

class Test_subpixelpeakfinders():
    """Tests the various peak finders have the correct x,y conventions for
    both the vectors and the shifts, in both the numpy and the DiffractionVectors
    cases as well as confirming we have avoided 'off by one' errors"""

    @pytest.fixture()
    def diffraction_vectors(self):
        v1 = np.array([[90 - 64, 30 - 64]])
        v2 = np.array([[90 - 64, 30 - 64], [100 - 64, 60 - 64]])
        z = np.array([[v1, v1], [v2, v2]])
        vectors = DiffractionVectors(z)
        vectors.axes_manager.set_signal_dimension(2)
        return vectors

    @pytest.fixture()
    def dp(self):
        dp =  np.zeros((2,2,128, 128))

        rr, cc = draw.circle(30, 90, radius=4, shape=(128,128))  # 30 is y!
        dp[:,:,rr, cc] = 1
        rr2, cc2 = draw.circle(100, 60, radius=4, shape=(128,128))
        dp[1,:,rr2, cc2] = 1

        # marks centers for com and gaussian methods
        dp[:,:,30, 90] = 2
        dp[1,:,100,60] = 2

        dp = ElectronDiffraction2D(dp)
        return dp

    def test_square_access(self,dp,diffraction_vectors):
        spg = SubpixelrefinementGenerator(dp,diffraction_vectors)
        def map_over_nav(z,vectors):
            for i, vector in enumerate(vectors):
                expt_disc = get_experimental_square(z, vector, 6)
                assert np.sum(expt_disc) > 0
        # does the test at every pixel
        dp.map(map_over_nav,vectors=spg.vector_pixels)

    @pytest.mark.skip(reason="Dont care yet")
    @pytest.mark.parametrize("strained",(True,False))
    def test_com(self,dp,diffraction_vectors,strained):
        small_strain = 1.05
        if strained:
            dp.apply_affine_transformation(D=np.eye(3)*small_strain,inplace=True)
        spg = SubpixelrefinementGenerator(dp, diffraction_vectors)
        vectors = spg.center_of_mass_method(20)
        if strained:
            ground_truth = vectors.data * small_strain
        else:
            ground_truth = vectors.data
        # all navigation positions match up
        for idx in [0,1]:
            for idy in [0,1]:
                assert np.allclose(vectors.data[idx,idy],ground_truth[idx,idy],atol=0.1)


    @pytest.mark.skip(reason="Dont care yet")
    @pytest.mark.parametrize("strained",(True,False))
    def test_xc_method(self,dp,diffraction_vectors,strained):
        small_strain = 1.05
        if strained:
            dp.apply_affine_transformation(D=np.eye(3)*small_strain,inplace=True)
        spg = SubpixelrefinementGenerator(dp, diffraction_vectors)
        vectors = spg.conventional_xc(12, 4, 8)
        if strained:
            ground_truth = vectors.data * small_strain
        else:
            ground_truth = vectors.data
        # all navigation positions match up
        for idx in [0,1]:
            for idy in [0,1]:
                assert np.allclose(vectors.data[idx,idy],ground_truth[idx,idy],atol=0.1)


    def test_gaussian_method(self):
        pass


def test_xy_errors_in_conventional_xc_method_as_per_issue_490():
    """ This was the MWE example code for the issue """
    dp = get_simulated_disc(100, 20)
    # translate y by +4
    shifted = np.pad(dp, ((0, 4), (0, 0)), "constant")[4:].reshape(1, 1, *dp.shape)
    signal = ElectronDiffraction2D(shifted)
    spg = SubpixelrefinementGenerator(signal, np.array([[0, 0]]))
    peaks = spg.conventional_xc(100, 20, 1).data[0, 0, 0]  # as quoted in the issue
    np.testing.assert_allclose([0, -4], peaks)
    """ we also test com method for clarity """
    peaks = spg.center_of_mass_method(60).data[0, 0, 0]
    np.testing.assert_allclose([0, -4], peaks, atol=1.5)
    """ we also test reference_xc """
    peaks = spg.reference_xc(100, dp, 1).data[0, 0, 0]  # as quoted in the issue
    np.testing.assert_allclose([0, -4], peaks)
