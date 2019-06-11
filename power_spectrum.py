import numpy as np
import matplotlib.pyplot as plt
from cosmology import Cosmology
from scipy.integrate import simps, quad
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize

class PowerSpec(object):
    """
    Class containing the linear power spectrum and useful methods

    Args:
        filename: Tabulated file of linear P(k) at z=0
        h0:       Hubble parameter at z=0, in units [100 km/s/Mpc]
        OmegaM:   Omega matter at z=0
    """
    def __init__(self, filename, h0, OmegaM):
        self.k, self.P = np.loadtxt(filename, unpack=True)
        self.cosmo = Cosmology(h0, OmegaM)
        self.tck = self.__get_sigma_spline() #spline fit to sigma(M,z=0)

        
    def P_lin(self, k, z):
        """
        Returns the linear power spectrum at redshift z

        Args:
            k: array of k in units [h/Mpc]
            z: array of z
        Returns:
            array of linear power spectrum in units [Mpc/h]^-3
        """
        tck = splrep(np.log10(self.k), np.log10(self.P))
        P0 = 10**splev(np.log10(k), tck)
        return P0 * self.cosmo.growth_factor(z)**2

    
    def Delta2_lin(self, k, z):
        """
        Returns the dimensionless linear power spectrum at redshift z,
        defined as Delta^2(k) = 4pi * (k/2pi)^3 * P(k)

        Args:
            k: array of k in units [h/Mpc]
            z: array of z
        Returns:
            array of dimensionless linear power spectrum
        """
        return self.P_lin(k, z) * k**3 / (2*np.pi**2)

    
    def W(self, k, R):
        """
        Window function in k-space (Fourier transform of top hat window)

        Args:
            k: array of k in units [h/Mpc]
            z: array of R in units [Mpc/h]
        Returns:
            window function
        """
        return 3 * (np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3

    
    def R_to_M(self, R):
        """
        Average mass enclosed by a sphere of comoving radius R

        Args:
            R: array of comoving radius in units [Mpc/h]
        Returns:
            array of mass in units [Msun/h]
        """
        return 4./3 * np.pi * R**3 * self.cosmo.mean_density(0)
    
    
    def M_to_R(self, M):
        """
        Comoving radius of a sphere which encloses on average mass M

        Args:
            M: array of mass in units [Msun/h]
        Returns:
            array of comoving radius in units [Mpc/h]
        """
        return (3*M / (4 * np.pi * self.cosmo.mean_density(0)))**(1./3)

    
    def __func(self, k, R):
        # function to integrate to get sigma(M)
        return self.k**2 * self.P * self.W(k,R)**2

    
    def __get_sigma_spline(self):
        # spline fit to sigma(R) at z=0
        logR = np.arange(-2,2,0.01)
        sigma = np.zeros(len(logR))
        R = 10**logR
        for i in range(len(R)):
            sigma[i] = simps(self.__func(self.k, R[i]), self.k)

        sigma = sigma / (2 * np.pi**2)
        sigma = np.sqrt(sigma)
        
        return splrep(logR, np.log10(sigma))

    
    def sigmaR_z0(self, R):
        """
        Returns sigma(R), the rms mass fluctuation in spheres of radius R,
        at redshift 0

        Args:
            R: array of comoving distance in units [Mpc/h]
        Returns:
            array of sigma
        """
        return 10**splev(np.log10(R), self.tck)

    
    def sigmaR(self, R, z):
        """
        Returns sigma(R,z), the rms mass fluctuation in spheres of radius R,
        at redshift z

        Args:
            R: array of comoving distance in units [Mpc/h]
            z: array of redshift
        Returns:
            array of sigma
        """
        return self.sigmaR_z0(R) * self.delta_c(0) / self.delta_c(z)

    
    def sigma_z0(self, M):
        """
        Returns sigma(M), the rms mass fluctuation in spheres of mass M,
        at redshift 0

        Args:
            M: array of mass in units [Msun/h]
        Returns:
            array of sigma
        """
        R = self.M_to_R(M)
        return self.sigmaR_z0(R)

    
    def sigma(self, M, z):
        """
        Returns sigma(M), the rms mass fluctuation in spheres of mass M,
        at redshift z

        Args:
            M: array of mass in units [Msun/h]
            z: array of redshift
        Returns:
            array of sigma
        """
        return self.sigma_z0(M) * self.delta_c(0) / self.delta_c(z)
    
    
    def nu(self, M, z):
        """
        Returns nu = delta_c(z=0) / (sigma(M,z=0) * D(z))

        Args:
            M: array of mass in units [Msun/h]
            z: array of redshift
        Returns:
            array of nu
        """
        return self.delta_c(z) / self.sigma_z0(M)

    
    def f_ST(self, nu):
        """
        Returns Sheth-Tormen mass function

        Args:
            nu: array of nu
        Returns:
            array of mass function
        """
        A = 0.216
        q = 0.707
        p = 0.3

        x = q*nu**2

        return A * (1. + 1./x**p) * np.exp(-x/2.)

    
    def mass_function(self, M, z):
        """
        Returns number density of haloes predicted by the Sheth-Tormen 
        mass function

        Args:
            M: array of mass in units [Msun/h]
            z: array of redshift
        Returns:
            array of number density in units [Mpc/h]^-3
        """
        nu = self.nu(M, z)
        f = self.f_ST(nu)
        return f * nu * -self.alpha(M,z) * np.log(10) * \
            self.cosmo.mean_density(0) / M

    
    def b(self, nu, z):
        """
        Returns Sheth-Tormen halo bias, where the values of the parameters
        have been modified to reproduce the halo bias measured from the
        OuterRim simulation, in which haloes are defined as friends-of-friends
        groups with linking length b=0.168

        Args:
            nu: array of nu
            z:  array of redshift
        Returns:
            array of halo bias
        """
        # bias (peak background split)
        dc = self.delta_c(0)
        a = 0.707 * 1.15
        p = 0.15
        x = a*nu**2

        A = (x-1.)/dc
        B = 2*p / (dc*(1.+x**p))

        return 1. + A + B

    
    def bM(self, M, z):
        """
        Returns Sheth-Tormen halo bias, as a function of mass,
        where the values of the parameters
        have been modified to reproduce the halo bias measured from the
        OuterRim simulation, in which haloes are defined as friends-of-friends
        groups with linking length b=0.168

        Args:
            M: array of mass in units [Msun/h]
            z: array of redshift
        Returns:
            array of halo bias
        """
        nu = self.nu(M, z)
        return self.b(nu, z)


    
    def __f_int1(self, M, z):
        #function to integrate when calculating b_eff
        nu = self.nu(M, z)
        x=self.b(nu, z) * self.f_ST(nu) / M
        return x
    
    def __f_int2(self, M, z):
        #function to integrate when calculating b_eff
        nu = self.nu(M, z)
        return self.f_ST(nu) / M

    
    def b_eff(self, Mmin, Mmax, z):
        """
        Returns the effective bias of haloes in the mass range
        Mmin < M < Mmax at redshift z

        Args:
            Mmin: minimum halo mass in units [Msun/h]
            Mmax: maximum halo mass in units [Msun/h]
            z:    redshift
        Returns:
            effective halo bias
        """
        A = quad(self.__f_int1, Mmin, Mmax, args=z)[0]
        B = quad(self.__f_int2, Mmin, Mmax, args=z)[0]
        
        return A/B


    def R_nl(self, z):
        """
        Returns the non-linear scale, defined as the value of R where
        sigma(R,z) = 1

        Args:
            z: redshift
        Returns:
            non-linear scale in units [Mpc/h]
        """
        def func(logM, z):
            return np.log10(self.sigma(10**logM,z))**2
        
        M = 10**minimize(func, x0=12, args=(z,))['x']

        R = self.M_to_R(M)
        
        return R


    
    def var_f(self, Rnl, Lbox, z):
        """
        Returns the expected variance of the the smoothed displacement
        field

        Args:
            Rnl: non-linear smoothing scale in units [Mpc/h]
            Lbox: simulation box size in units [Mpc/h]
            z: redshift
        Returns:
            variance of the displacement field
        """
        def func(k, Rnl, z):
            return np.exp(-(k**2*Rnl**2)) * \
                self.Delta2_lin(k,z) / k**2 

        kbox = 2*np.pi/Lbox

        lnk = np.arange(np.log(kbox), 10, 0.001)
        k = np.exp(lnk)

        f = func(k, Rnl, z)

        return np.sum(f)*0.001 




    def delta_c(self, z):
        """
        Returns delta_c, the linear density threshold for collapse, 
        at redshift z

        Args:
            z: redshift
        Returns:
            delta_c
        """
        return 1.686 / self.cosmo.growth_factor(z)

    
    
if __name__ == "__main__":
    
    filename0 = "pk_Mill.dat"
    P0 = PowerSpec(filename0, 0.73, 0.25) #WMAP1
