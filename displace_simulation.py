import numpy as np
import h5py
from displacement_field import DisplacementField
from power_spectrum import PowerSpec
from rescale import Rescale
from outerrim import *


def rescale_outerrim(file_number, snapshot, zorig, ztarg, Pk_orig, Pk_targ):

    pos, vel, mass = read_outerrim(snapshot, zorig, file_number)
    
    Mmin, Mmax = np.min(mass), np.max(mass)
    M1, M2 = 10**10.56772, 10**14.508154

    # Rescale to get s and z    
    r = Rescale(Pk_orig, Pk_targ)
    s, z = r.rescale(ztarg, M1, M2)

    # factor for scaling masses
    s_m = s**3 * OmegaM_targ/OmegaM_orig
    
    # factor for scaling velocities
    Ftarg = P_targ.cosmo.E(ztarg) * P_targ.cosmo.growth_rate(ztarg) / (1+ztarg)
    Forig = P_orig.cosmo.E(z) * P_orig.cosmo.growth_rate(z) / (1+z)
    s_v = s * Ftarg / Forig

    # scale pos, vel, mass
    print("z_orig = %.5f"%zorig)
    print("z_scaled = %.5f"%z) #this should match zorig
    print("difference = %.5f"%(z-zorig))
    print("z_targ = %.5f"%ztarg)
    print("s = %.5f"%s)
    print("s_m = %.5f"%s_m)
    print("s_v = %.5f"%s_v)
    
    pos = pos*s
    mass = mass * s_m
    L = 3000. * s
    vel_orig = vel.copy()
    vel = vel * s_v

    b_eff = P_targ.b_eff(Mmin*s_m, Mmax*s_m, ztarg)
    print("b_eff = %.5f"%b_eff)

    Rnl = P_targ.R_nl(ztarg)[0]
    print("R_nl = %.5f"%Rnl)

    d = DisplacementField(pos, ztarg, P_targ, L, nbins=750,
                          b_eff=b_eff, Rnl=Rnl, grid_file="delta.hdf5",
                          save_grid=False, read_grid=True)
    pos_new, vel_new = \
        d.apply_shift(pos, vel, mass, P_orig, P_targ, s, zorig, ztarg)


    
    f = h5py.File("outerrim_new.hdf5")
    f.create_dataset("Mass", data=mass, compression="gzip")
    f.create_dataset("Position", data=pos_new, compression="gzip")
    f.create_dataset("Velocity", data=vel_new, compression="gzip")
    f.close()



    

if __name__ == "__main__":

    import sys
    file_number = int(sys.argv[1])
    
    # original WMAP7 OuterRim cosmology
    h0_orig, OmegaM_orig, OmegaL_orig = 0.71, 0.2648, 0.7352
    Pkfile_orig = "Pk_orig.dat"
    P_orig = PowerSpec(Pkfile_orig, h0_orig, OmegaM_orig)

    # new target cosmology
    h0_targ, OmegaM_targ, OmegaL_targ = 0.70, 0.3, 0.7
    Pkfile_targ = "Pk_targ.dat"
    P_targ = PowerSpec(Pkfile_targ, h0_targ, OmegaM_targ)

    # redshift of original snapshot which is being rescaled
    zorig = 1.494
    snapshot = 198

    # target redshift in new cosmology
    ztarg = 1.433
    
    rescale_outerrim(file_number, snapshot, zorig, ztarg, P_orig, P_targ)
