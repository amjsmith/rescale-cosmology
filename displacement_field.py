import numpy as np
from scipy.fftpack import fftfreq
import pyfftw
import h5py


class DisplacementField(object):
    """
    Class which creates the displacement field 

    Args:
        pos:   array of halo comoving positions in units [Mpc/h]
        z:     target redshift
        Plin:  PowerSpec object in the target cosmology
        L:     simulation comoving box size, after scaling, in units [Mpc/h]
        nbins: number of bins in grid
        b_eff: effective bias of haloes
        Rnl:   non-linear smoothing scale in comoving [Mpc/h]
        grid_file: filename where density field is stored
        read_grid: if True, reads density field from delta_file
        save_grid: if True, saves density field to delta_file
    """
    def __init__(self, pos, z, Plin, L, nbins, b_eff, Rnl,
                 grid_file=None, read_grid=False, save_grid=False):

        self.Plin=Plin
        self.z=z

        self.L = L
        self.nbins = nbins
        self.binsize = self.L / self.nbins

        idx = np.where(pos==self.L)
        pos[idx] -= 1e-10
        
        # effective bias
        self.b_eff = b_eff

        # assign haloes to a grid
        delta = self.get_delta(pos, grid_file=grid_file, read_grid=read_grid,
                               save_grid=save_grid)
        
        # fourier modes
        self.k = fftfreq(self.nbins, d=self.binsize) * 2*np.pi
        self.kx = self.k[:,None,None]
        self.ky = self.k[None,:,None]
        self.kz = self.k[None,None,:]
        self.modk = (self.kx**2 + self.ky**2 + self.kz**2)**0.5

        # get x,y,z components of displacement field in fourier space
        # f_k,x = -i * delta_k / |k|^2 * k_x
        deltak = self.fft(delta)
        del delta
        
        fk = deltak / self.modk**2
        fk[0, 0, 0] = 0
        del deltak
        
        # smooth on scale Rnl
        fk = fk * np.exp(-self.modk**2 * Rnl**2 / 2.)
        
        self.fkx = -fk * -1j * self.kx 
        self.fky = -fk * -1j * self.ky
        self.fkz = -fk * -1j * self.kz
        del fk

        fx = self.ifft(self.fkx)
        fy = self.ifft(self.fky)
        fz = self.ifft(self.fkz)

        # scale displacement field so it has the expected variance
        fvar = np.var(fx) + np.var(fy) + np.var(fz)
        print("Measured variance = %.5f"%fvar)

        fvar_th = self.Plin.var_f(Rnl, self.L, self.z) 
        print("Theoretical variance = %.5f"%fvar_th)
        
        scale_factor = np.sqrt(fvar_th/fvar)

        print("Scaling factor = %.5f"%scale_factor)
        
        fx = fx * scale_factor
        fy = fy * scale_factor
        fz = fz * scale_factor
        
        self.fkx = self.fft(fx)
        self.fky = self.fft(fy)
        self.fkz = self.fft(fz)
        

    def get_delta(self, pos, grid_file=None, read_grid=False, save_grid=False):
        """
        Returns the matter density field on the grid

        Args:
            pos: array of halo positions in units [Mpc/h]
            grid_file: filename to read or save grid
            read_grid: if True, reads grid from grid_file
            save_grid: if True, saves grid to grid_file
        Returns:
            3d array of grid of matter density field
        """
        if not read_grid:
            N = pos.shape[0]
            idx = np.where(pos==self.L)
            pos[idx] -= 1e-1
            deltaH = self.assign_to_grid(pos)

        else:
            print("Reading density field from %s"%grid_file)
            f = h5py.File(grid_file)
            deltaH = f["deltaH"][...]
            N = f["N"][...][0]
            f.close()
            
        if save_grid:
            # save file of density field
            f = h5py.File(grid_file)
            f.create_dataset("deltaH", data=deltaH, compression="gzip")
            f.create_dataset("N", data=[N,], compression="gzip")
            f.close()
            
        n_av = float(N) / (self.nbins**3) # av number of haloes per grid cell
        deltaH = deltaH/n_av - 1.         # overdensity of haloes
        delta = deltaH / self.b_eff       # mass overdensity
            
        return delta

    
    def assign_to_grid(self, pos):
        """
        Puts haloes on a grid, with linear interpolation between grid cells

        Args:
            pos: array of halo comoving positions in units [Mpc/h]
        Returns:
            grid of haloes
        """
        delta  = pyfftw.empty_aligned((self.nbins, self.nbins, self.nbins),
                                      dtype='complex128')
        x, y, z = pos[:,0]/self.binsize, pos[:,1]/self.binsize,\
                  pos[:,2]/self.binsize
        i, j, k = x.astype(int), y.astype(int), z.astype(int)

        ddx, ddy, ddz = x-i, y-j, z-k

        deltag = np.zeros((self.nbins, self.nbins, self.nbins))
        edges = [np.linspace(0, self.nbins, self.nbins+1), \
                 np.linspace(0, self.nbins, self.nbins+1), \
                 np.linspace(0, self.nbins, self.nbins+1)]

        # loop through 8 cubic cells closest to each halo
        # assign linear interpolated weight to each grid cell
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    pos = np.array([i+ii, j+jj, k+kk]).transpose()

                    # periodic boundary conditions
                    idx = pos==self.nbins
                    pos[idx]=0
            
                    weight = ( ((1-ddx)+ii*(-1+2*ddx))*\
                               ((1-ddy)+jj*(-1+2*ddy))*\
                               ((1-ddz)+kk*(-1+2*ddz)) )

                    delta_t, edges = np.histogramdd(pos, bins=edges,
                                                    weights=weight)

                    deltag += delta_t
                    
        delta[...] = deltag
        return delta


    def get_delta_pos(self, P_orig, P_targ, s, z_orig, z_targ):
        """
        Returns differential displacement field

        Args:
            P_orig: PowerSpec object in the original cosmology 
            P_targ: PowerSpec object in the target cosmology 
            s:      factor used for scaling comoving positions
            z_orig: original simulation snapshot redshift
            z_targ: target redshift
        Returns:
            delta_fx: x-component of differential displacement field in units [Mpc/h]
            delta_fy: y-component of differential displacement field in units [Mpc/h]
            delta_fz: z-component of differential displacement field in units [Mpc/h]
        """
        Deltap = (self.modk)**3   * P_targ.P_lin(self.modk,   z_targ) 
        Delta  = (self.modk*s)**3 * P_orig.P_lin(self.modk*s, z_orig)

        F = np.sqrt(Deltap/Delta) - 1
        F[0,0,0]=0

        delta_fkx = self.fkx * F
        delta_fx = self.ifft(delta_fkx)
        del delta_fkx
        
        delta_fky = self.fky * F
        delta_fy = self.ifft(delta_fky)
        del delta_fky
        
        delta_fkz = self.fkz * F
        delta_fz = self.ifft(delta_fkz)
        del delta_fkz

        return delta_fx, delta_fy, delta_fz


    def get_delta_vel(self, P_orig, P_targ, s, z_orig, z_targ):
        """
        Returns differential velocity field

        Args:
            P_orig: PowerSpec object in the original cosmology 
            P_targ: PowerSpec object in the target cosmology 
            s:      factor used for scaling comoving positions
            z_orig: original simulation snapshot redshift
            z_targ: target redshift
        Returns:
            delta_vx: x-component of differential velocity field in units [km/s]
            delta_vy: y-component of differential velocity field in units [km/s]
            delta_vz: z-component of differential velocity field in units [km/s]
        """
        Deltap = (self.modk)**3   * P_targ.P_lin(self.modk,   z_targ) 
        Delta  = (self.modk*s)**3 * P_orig.P_lin(self.modk*s, z_orig)

        F = np.sqrt(Deltap/Delta) - 1
        F[0,0,0]=0

        # Get delta_v needed to dispace velocities

        factor = 100. * P_targ.cosmo.E(z_targ) * \
            P_targ.cosmo.growth_rate(z_targ) / (1+z_targ)

        
        delta_vkx = factor * F * self.fkx
        delta_vx = self.ifft(delta_vkx)
        del delta_vkx
        
        delta_vky = factor * F * self.fky
        delta_vy = self.ifft(delta_vky)
        del delta_vky
        
        delta_vkz = factor * F * self.fkz
        delta_vz = self.ifft(delta_vkz)
        del delta_vkz

        return delta_vx, delta_vy, delta_vz

    
    def apply_shift(self, pos, vel, mass, P_orig, P_targ, s, z_orig, z_targ):
        """
        Applies displacements to the halo positions and velocities

        Args:
            pos:    array of halo positions in units [Mpc/h]
            vel:    array of halo velocities in units [km/s]
            mass:   array of halo masses in units [Msun/h]
            P_orig: PowerSpec object in the original cosmology 
            P_targ: PowerSpec object in the target cosmology 
            s:      factor used for scaling comoving positions
            z_orig: original simulation snapshot redshift
            z_targ: target redshift
        Returns:
            pos_new: array of displaced halo positions in units [Mpc/h]
            vel_new: array of displaced halo velocities in units [km/s]
        """
        # shift positions of haloes
        # power spectrum and redshift needed for mass-dependent displacements
        shift_x, shift_y, shift_z, shift_vx, shift_vy, shift_vz = \
                  self.get_shift(pos, mass, P_orig, P_targ, s, z_orig, z_targ)

        pos_new = pos.copy()
        pos_new[:,0] += shift_x
        pos_new[:,1] += shift_y
        pos_new[:,2] += shift_z

        vel_new = vel.copy()
        vel_new[:,0] += shift_vx
        vel_new[:,1] += shift_vy
        vel_new[:,2] += shift_vz

        # peridic boundary
        pos_new[pos_new>=self.L] -= self.L
        pos_new[pos_new<0] += self.L

        return pos_new, vel_new

        

    def get_shift(self, pos, mass, P_orig, P_targ, s, z_orig, z_targ):
        """
        Returns the displacements to be applied to the position and velocity
        of each halo

        Args:
            pos:    array of halo positions in units [Mpc/h]
            mass:   array of halo masses in units [Msun/h]
            P_orig: PowerSpec object in the original cosmology 
            P_targ: PowerSpec object in the target cosmology 
            s:      factor used for scaling comoving positions
            z_orig: original simulation snapshot redshift
            z_targ: target redshift
        Returns:
            shift_x:  array of x-component of position displacement in units [Mpc/h]
            shift_y:  array of y-component of position displacement in units [Mpc/h]
            shift_z:  array of z-component of position displacement in units [Mpc/h]
            shift_vx: array of x-component of velocity displacement in units [km/s]
            shift_vy: array of y-component of velocity displacement in units [km/s]
            shift_vz: array of z-component of velocity displacement in units [km/s]
        """
        delta_fx, delta_fy, delta_fz = \
                    self.get_delta_pos(P_orig, P_targ, s, z_orig, z_targ)

        delta_vx, delta_vy, delta_vz = \
                    self.get_delta_vel(P_orig, P_targ, s, z_orig, z_targ)
        
        x, y, z = pos[:,0]/self.binsize, pos[:,1]/self.binsize,\
                  pos[:,2]/self.binsize
        i, j, k = x.astype(int), y.astype(int), z.astype(int)

        ddx, ddy, ddz = x-i, y-j, z-k
        
        shift_x = np.zeros(len(x))
        shift_y = np.zeros(len(x))
        shift_z = np.zeros(len(x))
        
        shift_vx = np.zeros(len(x))
        shift_vy = np.zeros(len(x))
        shift_vz = np.zeros(len(x))
        
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    
                    weight = ( ((1-ddx)+ii*(-1+2*ddx))*\
                               ((1-ddy)+jj*(-1+2*ddy))*\
                               ((1-ddz)+kk*(-1+2*ddz)) )

                    iii = i+ii
                    jjj = j+jj
                    kkk = k+kk
                    
                    iii[iii==self.nbins]=0
                    jjj[jjj==self.nbins]=0
                    kkk[kkk==self.nbins]=0
                    
                    pos2 = (iii,jjj,kkk)
                    
                    shift_x += np.real(delta_fx[pos2])*weight
                    shift_y += np.real(delta_fy[pos2])*weight
                    shift_z += np.real(delta_fz[pos2])*weight

                    shift_vx += np.real(delta_vx[pos2])*weight
                    shift_vy += np.real(delta_vy[pos2])*weight
                    shift_vz += np.real(delta_vz[pos2])*weight

        # mass dependent shifts to get right b(M)
        bias = self.Plin.bM(mass, z_targ)
            
        logM = np.log10(mass)
            
        shift_x *= bias
        shift_y *= bias
        shift_z *= bias
                    
        return shift_x, shift_y, shift_z, shift_vx, shift_vy, shift_vz

    
    def fft(self, array, inv=False):
        """
        Fourier transforms an array. This is normalized so that ifft(fft(array)) = array

        Args:
            array: array to transform
            inv:   if True does a forward transform
                   if False does the inverse transform
        Returns:
            transformed array
        """
        array2 = pyfftw.empty_aligned((self.nbins, self.nbins, self.nbins),
                                      dtype='complex128')

        if not inv:
            fft_obj = pyfftw.FFTW(array, array2, axes=[0, 1, 2])
        else:
            fft_obj = pyfftw.FFTW(array, array2, axes=[0, 1, 2],
                                  direction='FFTW_BACKWARD')

        return fft_obj()

    
    def ifft(self, array):
        """
        Inverse Fourier transforms an array. 

        Args:
            array: array to inverse transform
        Returns:
            inverse transformed array
        """
        return self.fft(array, inv=True)



