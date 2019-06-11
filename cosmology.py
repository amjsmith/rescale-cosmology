#! /usr/bin/env python
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad
from scipy.interpolate import splev, splrep

# constants
Msun_g = 1.989e33 # solar mass in g
Mpc_cm = 3.086e24 # Mpc in cm


class Cosmology(object):
    """
    Class containing useful cosmology methods. Assumes flat LCDM Universe.

    Args:
        h0:     Hubble parameter at z=0, in units [100 km/s/Mpc]
        OmegaM: Omega matter at z=0
    """
    def __init__(self, h0, OmegaM):

        self.h0     = h0
        self.OmegaM = OmegaM
        self.OmegaL = 1. - self.OmegaM

        # assumes cosmology is flat LCDM
        self.__cosmo = FlatLambdaCDM(H0=h0*100, Om0=OmegaM)

        # RegualarGridInterpolator used to convert comoving distance to z
        self.__interpolator = self.__initialize_interpolator()

        # Cubic spline used when calculating the growth rate
        self.__spl_growth_rate = self.__initialize_growth_rate_spline()


        
    def __initialize_interpolator(self):
        # create RegularGridInterpolator for converting comoving
        # distance to redshift
        z = np.arange(0, 3, 0.0001)
        rcom = self.comoving_distance(z)
        return RegularGridInterpolator((rcom,), z,
                                       bounds_error=False, fill_value=None)


    def comoving_distance(self, redshift):
        """
        Converts redshift to comoving distance

        Args:
            redshift: array of redshift
        Returns:
            array of comoving distance in units [Mpc/h]
        """
        return self.__cosmo.comoving_distance(redshift).value*self.h0


    def redshift(self, distance):
        """
        Converts comoving distance to redshift

        Args:
            distance: comoving distance in units [Mpc/h]
        Returns:
            array of redshift
        """
        return self.__interpolator(distance)
    

    
    def E(self, z):
        """
        Returns function E(z)
        """
        return np.sqrt(self.OmegaM*(1.+z)**3 + self.OmegaL)
    
    
    def H(self, z):
        """
        Returns the hubble parameter H(z) at redshift z
        Args:
            z: redshift
        Returns:
            H(z)
        """
        return 100*self.h0 * self.E(z)

    
    def OmegaM_z(self, z):
        """
        Returns OmegaM at redshift z
        
        Args:
            z: redshift
        Returns:
            OmegaM
        """
        return self.mean_density(z)/self.critical_density(z)
    
    def critical_density(self, redshift):
        """
        Critical density of the Universe as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of critical density in units [Msun Mpc^-3 h^2]
        """
        rho_crit = self.__cosmo.critical_density(redshift).value # in g cm^-3

        # convert to Msun Mpc^-3 h^2
        rho_crit *= Mpc_cm**3 / Msun_g / self.h0**2

        return rho_crit


    def mean_density(self, redshift):
        """
        Mean matter density of the Universe as a function of redshift

        Args:
            redshift: array of redshift
        Returns:
            array of critical density in units [Msun Mpc^-3 h^2]
        """
        # mean density at z=0
        rho_mean0 = self.critical_density(0) * self.OmegaM

        # evolve to redshift z
        return  rho_mean0 * (1+redshift)**3

    

    def __func(self, z):
        # Function that is integrated when calculating the growth factor
        return (1+z)/self.H(z)**3

    
    def growth_factor(self, z):
        """
        Linear growth factor D(a), as a function of redshift

        Args:
            z: array of redshift
        Returns:
            Linear growth factor
        """

        if isinstance(z, np.ndarray):
            # do loop if numpy array
            a = np.zeros(len(z))
            for i in range(len(z)):
                a[i] = quad(self.__func, z[i], np.inf)[0]
        else:
            a = quad(self.__func, z, np.inf)[0]
        b = quad(self.__func, 0, np.inf)[0]
        return self.H(z)/self.H(0) * a / b


    def __initialize_growth_rate_spline(self):
        # Cubic spline of growth rate
        a = np.arange(1, 0.1, -0.001)
        zs = 1./a - 1
        
        g = self.growth_factor(zs)
        
        f = (np.log(g[1:]) - np.log(g[:-1])) / (np.log(a[1:])-np.log(a[:-1]))
        
        zs = (zs[:-1]+zs[1:])/2.
        return splrep(zs, f)
        
    
    def growth_rate(self, z):
        """
        Returns the growth rate, f = dln(D)/dln(a)

        Args:
            z: array of redshift
        Returns:
            Growth rate
        """
        return splev(z, self.__spl_growth_rate)

    
    
if __name__ == "__main__":

    h0=0.71
    OmegaM=(0.1109 + 0.02258) / h0**2
    
    c = Cosmology(h0, OmegaM)

    print(c.mean_density(0))
    print(c.critical_density(0))
    
    print(c.growth_factor(1.433))
    print(c.growth_rate(1.433))
    print(c.OmegaM_z(1.433)**0.55)
