import numpy as np
from power_spectrum import PowerSpec
from scipy.integrate import quad
from scipy.optimize import minimize

class Rescale(object):
    """
    Class which rescales the cosmology of a simulation by matching
    sigma(R) between the original and target cosmologies

    Args:
        Pk:  PowerSpec object in the original simulation cosmology   
        Pkp: PowerSpec object in the target cosmology
    """  
    def __init__(self, Pk, Pkp):

        self.Pk  = Pk
        self.Pkp = Pkp


    def __delta2(self, x, zp, R1, R2):
        # function to minimize (Eq. 2 from Mead & Peacock)
        s, z = x
        R1p, R2p = R1*s, R2*s
        return quad(self.__f_int, R1p, R2p, args=(s,z,zp))[0] / np.log(R2p/R1p)
            
    
    def __f_int(self, R, s, z, zp):
        # integral from Eq. 2 of Mead & Peacock
        M  = self.Pk.R_to_M(R/s)
        Mp = self.Pkp.R_to_M(R)
        return (1. - self.Pk.sigma(M, z)/self.Pkp.sigma(Mp, zp))**2 / R
            

    def rescale(self, zp, Mmin, Mmax):
        """
        Finds the scaling factor s and redshift z needed to reproduce
        the sigma(R) for the new cosmology

        Args:
            zp:   target redshift in new cosmology
            Mmin: minimum halo mass of interest in units [Msun/h]
            Mmax: maximum halo mass of interest in units [Msun/h]
        Returns:
            s: factor for scaling comoving positions
            z: redshift in original cosmology
        """
        R1 = self.Pk.M_to_R(Mmin)
        R2 = self.Pk.M_to_R(Mmax)

        s, z = minimize(self.__delta2, x0=[1, zp], args=(zp, R1, R2)).x

        return s, z
    
