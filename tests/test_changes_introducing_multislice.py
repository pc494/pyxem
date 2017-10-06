import numpy as np
import hyperspy.api as hs
import pycrystem as pc
import pymatgen as pmg
from pycrystem.indexation_generator import IndexationGenerator
from scipy.constants import pi

si = pmg.Element("Si")
lattice = pmg.Lattice.cubic(5.431)
silicon = pmg.Structure.from_spacegroup("Fd-3m",lattice, [si], [[0, 0, 0]])

from pymatgen.transformations.standard_transformations import RotationTransformation

size = 256

radius=1.5
ediff = pc.ElectronDiffractionCalculator(300., 0.025)

rotaxis = [0, 0, 1]
thetas = np.arange(0, 45, 1)

data_silicon = []
for theta in thetas:
    rot = RotationTransformation(rotaxis, theta)
    sieg = rot.apply_transformation(silicon)
    diff_dat = ediff.calculate_ed_data(sieg,reciprocal_radius=radius)
    dpi = diff_dat.as_signal(256, 0.03, 1.2)
    data_silicon.append(dpi.data)
    
data = [data_silicon] * 3 
test_data = pc.ElectronDiffraction(data)
test_data.set_calibration(1.2/128)

#test_data.plot()
help(ediff)
