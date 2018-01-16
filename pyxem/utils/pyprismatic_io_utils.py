import numpy as np
import pymatgen as pmg
import os
import pyprismatic as pr

def generate_pyprismatic_input(structure):
        """     Generates a .XYZ file on which Pyprismatic can run
        Args:
            structure: pymatgen.Structure object 
                The entire structure
        Returns:
            None + file 'PP_input.XYZ'
        
        """
        
        if os.path.exists('PP_input.XYZ'):
            raise IOError('The file the program is attempting to write to already exists, please remove it.')
            #It would be addition (ie not overwrite) here
            pass
        
        atomic_numbers = np.asarray([S.Z for S in structure.species]).reshape(len(structure.species),1)
        cart_coords = structure.cart_coords
        percent_occupied  = np.ones_like(atomic_numbers)
        debeye_waller = np.zeros_like(atomic_numbers)
        # TODO Get dw from Element's name
        
        printing_array = np.hstack([atomic_numbers,cart_coords,percent_occupied,debeye_waller])
        
        # TODO introduce logic to allow actual unit cells to be introduced
        
        line_2 = [np.max(cart_coords[:,0]),np.max(cart_coords[:,1]),np.max(cart_coords[:,2])]
        
        # TODO consider if these violations are prevented by the use of pymatgen.Structure
        if [0,0,0] not in structure.cart_coords:
            raise ValueError('To use the whole sample as a unit cell we require [0,0,0] as a co-ordinate')
        if np.max(line_2) <= 1:
            raise ValueError('Inputs should not be in fractional form')
        
        with open('PP_input.XYZ', 'a') as f:
            print("Default Comment",file=f)
            print('    {0:.3g}   {1:.3g}   {2:.3g}'.format(line_2[0],line_2[1],line_2[2]),file=f)
            for row in printing_array:
                    print('{0:.3g} {1:.4f} {2:.4f} {3:.4f} {4:.3f} {5:.3f}'.format(
                            row[0],row[1],row[2],row[3],row[4],row[5]),
                        file=f)
            print("-1",file=f)
        return None
        
def run_pyprismatic_simulation(prismatic_kwargs=None):
    if prismatic_kwargs == None:
        prismatic_kwargs = {}
    print(prismatic_kwargs)
    if 'filenameAtoms' not in prismatic_kwargs.keys():
        prismatic_kwargs.update({'filenameAtoms':"PP_input.XYZ"})
    if 'filenameOutput' not in prismatic_kwargs.keys():
        prismatic_kwargs.update({'filenameOutput':"PP_output.mrc"})
    print(prismatic_kwargs)
    meta = pr.Metadata(**prismatic_kwargs) ##Sticks to defaults apart from the unpacked dict
    ## Delete input
    ## Print the meta to a file
    meta.go()
    return meta

def import_pyprismatic_data(meta):
    read_file = meta.filenameOutput
    output = pr.fileio.readMRC(read_file)
    ## Delete output
    return output