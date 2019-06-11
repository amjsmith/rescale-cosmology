import numpy as np
from nbodykit.lab import cosmology as nbodykit_cos
from power_spectrum import PowerSpec
from rescale import Rescale
from scipy.optimize import minimize


class Cosmo(object):
    """
    Cosmology class, used for finding a set of rescaled cosmological parameters

    Args:
        z:         snapshot redshift
        Omega_b:   Omega baryon at z=0
        Omega_cdm: Omega cold dark matter at z=0
        h:         hubble parameter at z=0, in units [100 km/s/Mpc]
        sigma8:    sigma8
        n_s:       n_s
        Pk_file:   file where P(k) at z=0 will be stored
    """
    def __init__(self, z, Omega_b, Omega_cdm, h, sigma8, n_s, Pk_file):

        self.z = z
        self.params = {'Omega_b':Omega_b,
                       'Omega_cdm':Omega_cdm,
                       'h':h,
                       'sigma8':sigma8,
                       'n_s': n_s}

        self.c = self.__nbodykit_cosmo()

        self.Pk_file = Pk_file
        self.save_file()

        
    def __nbodykit_cosmo(self):
        #Creates an nbodykit cosmology object for this set of cosmological parameters
        c = nbodykit_cos.WMAP7
        c = c.clone(Omega0_b=self.params['Omega_b'],
                    Omega0_cdm=self.params['Omega_cdm'],
                    h=self.params['h'], n_s=self.params['n_s'])
        c = c.match(sigma8=self.params['sigma8'])
        return c


    def save_file(self):
        """
        Creates a tabulated ascii file of the linear power spectrum P(k) at z=0.
        The first column contains k in units [h/Mpc], and the second column contains
        P(k), in units [Mpc/h]^-3
        """
        Plin = nbodykit_cos.LinearPower(self.c, redshift=0, transfer='CLASS')
        k = np.logspace(-6, 6, 1000)
        P = Plin(k)
        data = np.array([k,P]).transpose()
        np.savetxt(self.Pk_file, data)

    
    def set(self, param, value):
        """
        Set one of the cosmological parameters to a new value

        Args:
            param: name of cosmological parameter. can be "Omega_b", "Omega_cdm",
                   "h", "sigma8" or "n_s"
            value: new value
        """
        self.params[param] = value
        self.c = self.__nbodykit_cosmo()
        self.save_file()


    def get(self, param):
        """
        Returns the value of one of the cosmological parameters

        Args:
            param: name of cosmological parameter. can be "Omega_b", "Omega_cdm",
                   "h", "sigma8", "n_s", "OmegaM" or "OmegaL"
        """
        if param=="OmegaM":
            # total Omega matter
            return self.params["Omega_b"] + self.params["Omega_cdm"]
        elif param=="OmegaL":
            # Omega lambda, assumes flat LCDM
            return 1.-self.get("OmegaM")
        else:
            return self.params[param]

        
        
    def match_redshift(self, cosmo_orig, param):
        """
        Adjust one of the cosmological parameters so the the scaled redshift
        agrees with the original snapshot redshift

        Args:
            cosmo_orig: object of class Cosmo, in the original simulation cosmology
            param:      name of cosmological parameter to adjust. can be "Omega_b", 
                        "Omega_cdm", "h", "sigma8" or "n_s"
        """

        print("Finding value of %s that is needed to match the original redshift z=%.3f"%(param, cosmo_orig.z))
        
        minimize(self.__difference2, x0=self.get(param), args=(cosmo_orig, param))

        print("Done")
        print("P(k) in new cosmology with %s=%.5f saved to %s"%(param, self.get(param), self.Pk_file))


            
    def __difference2(self, value, cosmo_orig, param):
        # do rescaling, and return square of difference between z after scaling
        # and the original redshift.
        # this function is minimized
        self.set(param, value)

        P_orig = PowerSpec(cosmo_orig.Pk_file, cosmo_orig.get('h'),
                           cosmo_orig.get('OmegaM'))
        P_targ = PowerSpec(self.Pk_file, self.get('h'), self.get('OmegaM'))

        r = Rescale(P_orig, P_targ)

        M1 = 10**10.56772
        M2 = 10**14.508154

        s, z = r.rescale(self.z, M1, M2)

        print("%s=%.5f, z=%.5f"%(param,value[0],z))
        
        return (cosmo_orig.z - z)**2
    


if __name__ == '__main__':

    # In this example, the OuterRim snapshot at z=1.494 is rescaled to a
    # new cosmology at z=1.433

    # Original OuterRim WMAP7 cosmology
    zorig = 1.494 # original redshift of simulation snapshot
    Omega0_b_orig=0.0448
    Omega0_cdm_orig = 0.2200
    h_orig = 0.71
    sigma8_orig = 0.8
    n_s_orig = 0.963
    f_orig = "Pk_orig.dat" # file to save P(k) at z=0

    C_orig = Cosmo(zorig, Omega0_b_orig, Omega0_cdm_orig, h_orig, sigma8_orig,
                   n_s_orig, f_orig)

    
    # New target cosmology
    ztarg = 1.433 # target redshift we want to make mock at
    Omega0_b_targ=0.05
    Omega0_cdm_targ = 0.25
    h_targ = 0.70
    sigma8_targ = 0.80
    n_s_targ = 1.
    f_targ = "Pk_targ.dat" # file to save P(k) at z=0

    C_targ = Cosmo(ztarg, Omega0_b_targ, Omega0_cdm_targ, h_targ, sigma8_targ,
                   n_s_targ, f_targ)


    # adjust n_s in the target cosmology so that the scaled redshift is equal
    # to zorig of the input snapshot
    C_targ.match_redshift(C_orig, "n_s")
