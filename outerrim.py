import numpy as np
import h5py
import genericio

def read_outerrim(snapshot, redshift, file_number, path=""):
    """
    Reads halo position, velocity and mass from OuterRim file
    
    Args:
        snapshot: OuterRim snapshot number
        redshift: redshift of OuterRim snapshot
        file_number: file number of OuterRim file to read
        path:     path of directory where OuterRim files are stored
    Returns:
        pos:  array of comoving positions in units [Mpc/h]
        vel:  array of proper velocities in units [km/s]
        mass: array of masses in units [Msun/h]
    """

    f = path + "02_17_2016.OuterRim.%i.fofproperties#%i"%(snap,file_number)

    # read halo mass
    mass = genericio.gio_read(halo_cat, 'fof_halo_mass')[0]

    # read comoving positions
    x = genericio.gio_read(halo_cat, 'fof_halo_center_x')[0]
    y = genericio.gio_read(halo_cat, 'fof_halo_center_y')[0]
    z = genericio.gio_read(halo_cat, 'fof_halo_center_z')[0]
    pos = np.array([x,y,z]).transpose()
    del x, y, z

    # read comoving velocities
    vx = genericio.gio_read(halo_cat, 'fof_halo_mean_vx')[0] 
    vy = genericio.gio_read(halo_cat, 'fof_halo_mean_vy')[0] 
    vz = genericio.gio_read(halo_cat, 'fof_halo_mean_vz')[0]
    vel = np.array([vx, vy, vz]).transpose()
    del vx, vy, vz

    # convert velocities from comoving to proper
    vel = vel / (1.+redshift)
    
    return pos, vel, mass

