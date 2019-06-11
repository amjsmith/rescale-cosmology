import numpy as np
import h5py


def merge_grid(Nfiles=110):
    deltaH = None
    N=0

    for i in range(Nfiles):
        print(i)
        f = h5py.File("delta%i.hdf5"%i)
        d = f["deltaH"][...]
        N += f["N"][...][0]
        f.close()
    
        if i==0:
            deltaH=d
        else:
            deltaH += d

        print(N)

    f = h5py.File("delta.hdf5")
    f.create_dataset("deltaH", data=deltaH, compression="gzip")
    f.create_dataset("N", data=[N,])
    f.close()


if __name__ == "__main__":

    merge_grid()
