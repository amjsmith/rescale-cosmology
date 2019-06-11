import numpy as np
from outerrim import read_outerrim
from displacement_field import DisplacementField
import sys


def make_grid(file_number, snapshot, redshift, nbins):

    pos, vel, mass = read_outerrim(snapshot, redshift)

    try:
        # code will calculate density field and save to file,
        # then will crash since Plin is set to None
        d = DisplacementField(pos, z=redshift, Plin=None, L=3000.,
                              nbins=nbins, b_eff=1., Rnl=1.,
                              grid_file="delta%i.hdf5"%file_number,
                              save_grid=True, read_grid=False)
    except AttributeError:
        print("Done")


if __name__ == "__main__":

    file_number = int(sys.argv[1])

    snapshot=198
    redshift=1.494
    nbins = 750
    
    make_grid(file_number, snapshot, redshift, nbins)



