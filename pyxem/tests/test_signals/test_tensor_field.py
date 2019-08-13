# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

from pyxem.signals.tensor_field import _polar_decomposition, _get_rotation_angle
from pyxem.signals.tensor_field import DisplacementGradientMap


@pytest.mark.parametrize('D, side, R, U', [
    (np.array([[0.98860899, -0.2661997, 0.],
               [0.2514384, 0.94324267, 0.],
               [0., 0., 1.]]),
     'right',
     np.array([[0.96592583, -0.25881905, 0.],
               [0.25881905, 0.96592583, 0.],
               [0., 0., 1.]]),
     np.array([[1.02, -0.013, 0.],
               [-0.013, 0.98, 0.],
               [0., 0., 1.]])),
])
def test_polar_decomposition(D, side, R, U):
    Rc, Uc = _polar_decomposition(D, side=side)
    np.testing.assert_almost_equal(Rc, R)
    np.testing.assert_almost_equal(Uc, U)


@pytest.mark.parametrize('R, theta', [
    (np.array([[0.76604444, 0.64278761, 0.],
               [-0.64278761, 0.76604444, 0.],
               [0., 0., 1.]]),
     0.6981317007977318),
    (np.array([[0.96592583, -0.25881905, 0.],
               [0.25881905, 0.96592583, 0.],
               [0., 0., 1.]]),
     -0.2617993877991494),
])
def test_get_rotation_angle(R, theta):
    tc = _get_rotation_angle(R)
    np.testing.assert_almost_equal(tc, theta)


@pytest.mark.parametrize('dgm, rotation_map, distortion_map', [
    (DisplacementGradientMap(np.array([[[[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]],
                                        [[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]]],
                                       [[[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]],
                                        [[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]]]])),
     np.array([[[[0.96592583, -0.25881905, 0.],
                 [0.25881905, 0.96592583, 0.],
                 [0., 0., 1.]],
                [[0.96592583, -0.25881905, 0.],
                 [0.25881905, 0.96592583, 0.],
                 [0., 0., 1.]]],
               [[[0.96592583, -0.25881905, 0.],
                 [0.25881905, 0.96592583, 0.],
                 [0., 0., 1.]],
                [[0.96592583, -0.25881905, 0.],
                 [0.25881905, 0.96592583, 0.],
                 [0., 0., 1.]]]]),
     np.array([[[[1.02, -0.013, 0.],
                 [-0.013, 0.98, 0.],
                 [0., 0., 1.]],
                [[1.02, -0.013, 0.],
                 [-0.013, 0.98, 0.],
                 [0., 0., 1.]]],
               [[[1.02, -0.013, 0.],
                 [-0.013, 0.98, 0.],
                 [0., 0., 1.]],
                [[1.02, -0.013, 0.],
                 [-0.013, 0.98, 0.],
                 [0., 0., 1.]]]])),
])
def test_map_polar_decomposition(dgm,
                                 rotation_map,
                                 distortion_map):
    Rc, Uc = dgm.polar_decomposition()
    np.testing.assert_almost_equal(Rc.data, rotation_map)
    np.testing.assert_almost_equal(Uc.data, distortion_map)


@pytest.mark.parametrize('dgm, strain_answers', [
    (DisplacementGradientMap(np.array([[[[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]],
                                        [[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]]],
                                       [[[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]],
                                        [[0.98860899, -0.2661997, 0.],
                                         [0.2514384, 0.94324267, 0.],
                                         [0., 0., 1.]]]])),
     np.array([[[-0.02, -0.02],
                [-0.02, -0.02]],
               [[0.02, 0.02],
                [0.02, 0.02]],
               [[-0.013, -0.013],
                [-0.013, -0.013]],
               [[-0.26179939, -0.26179939],
                [-0.26179939, -0.26179939]]])),
])
def test_get_strain_maps(dgm,
                         strain_answers):
    strain_results = dgm.get_strain_maps()
    np.testing.assert_almost_equal(strain_results.data, strain_answers)

""" These test will be operational once a basis change functionality is introduced """

from pyxem.tests.test_generators.test_displacement_gradient_tensor_generator import generate_test_vectors
import hyperspy.api as hs
from pyxem.generators.displacement_gradient_tensor_generator import get_DisplacementGradientMap
from pyxem.signals.tensor_field import _get_xnew_xold_Rmatrix

def test__get_xnew_xold_Rmatrix():
    x_new = [1,1] # our rotation should be 45
    R_to_test = _get_xnew_xold_Rmatrix(x_new)
    x_new.append(0)
    x_new = np.asarray(x_new)
    output = np.matmul(R_to_test,x_new)
    assert np.allclose(x_new,output)

#@pytest.fixture() #fixtures stay the same for too long
def Displacement_Grad_Map():
    xy = np.asarray([[1, 0], [0, 1]])
    deformed = hs.signals.Signal2D(generate_test_vectors(xy))
    D = get_DisplacementGradientMap(deformed,xy)
    return D

@pytest.mark.skip(reason="Failures above")
def test_something_changes():
    oneone_strain_original = Displacement_Grad_Map().get_strain_maps().inav[0].data
    local_D  = Displacement_Grad_Map()
    local_D.rotate_strain_basis([1.3,+1.9]) #works in place
    oneone_strain_alpha =  local_D.get_strain_maps().inav[0].data
    assert not np.allclose(oneone_strain_original,oneone_strain_alpha, atol=2)


@pytest.mark.skip(reason="Failing test above")
def test_rotation(Displacement_Grad_Map):  # pragma: no cover
    """
    We should always measure the same rotations, regardless of basis
    """
    local_D  = Displacement_Grad_Map
    original = local_D.get_strain_maps().inav[3].data
    local_D.rotate_strain_basis([1.3,+1.9]) #works in place
    rotation_alpha =  local_D.get_strain_maps().inav[3].data
    local_D = Displacement_Grad_Map
    local_D.rotate_strain_basis([1.7,-0.3])
    rotation_beta = local_D.get_strain_maps().inav[3].data

    # check the functionality has left invarient quantities invarient
    np.testing.assert_almost_equal(original, rotation_alpha, decimal=2)  # rotations
    np.testing.assert_almost_equal(original, rotation_beta, decimal=2)  # rotations


@pytest.mark.skip(reason="basis change functionality not yet implemented")
def test_trace(xy_vectors, right_handed, multi_vector):  # pragma: no cover
    """
    Basis does effect strain measurement, but we can simply calculate suitable invarients.
    See https://en.wikipedia.org/wiki/Infinitesimal_strain_theory for details.
    """
    np.testing.assert_almost_equal(
        np.add(
            xy_vectors.inav[0].data, xy_vectors.inav[1].data), np.add(
            right_handed.inav[0].data, right_handed.inav[1].data), decimal=2)
    np.testing.assert_almost_equal(
        np.add(
            xy_vectors.inav[0].data, xy_vectors.inav[1].data), np.add(
            multi_vector.inav[0].data, multi_vector.inav[1].data), decimal=2)
