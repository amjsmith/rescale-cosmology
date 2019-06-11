# rescale-cosmology

This repository contains a Python implementation of the method of Mead & Peacock for rescaling the cosmology of a halo
catalogue from an N-body simulation.

The method is described in the papers https://arxiv.org/abs/1308.5183 and https://arxiv.org/abs/1408.1047, and is an
extension of the method of Angulo & White (https://arxiv.org/abs/0912.4277).

See https://github.com/alexander-mead/particle-rescaling for a Fortran implementation of the method.

The code for putting haloes on a grid and calculating the displacement field was based on the reconstruction code 
from https://github.com/julianbautista/eboss_clustering

# Rescaling the OuterRim simulation

## Downloading an OuterRim snapshot

Snapshots from the OuterRim simulation can be accessed from Portsmouth using a globus end point. For more details, see
https://trac.sdss.org/wiki/eBOSS/CosmoSim/DMOsims

Each simulation snapshot is split into 110 files, which are in the genericio format. A Python library for reading
these files can be downloaded from https://trac.alcf.anl.gov/projects/genericio.

The mapping between snapshot number and redshift can be found in the file
https://github.com/viogp/outerrim_mocks/blob/master/step_redshift.txt

## Choosing a new cosmology (get_cosmo.py)

In the first part of the method, an OuterRim snapshot at redshift z is rescaled to a new cosmology at a target redshift z' by
matching sigma(R). However, the new target cosmology must be chosen so that, when rescaling, the redshift z matches
the redshift of the simulation snapshot. The file `get_cosmo.py` can be used to find a suitable cosmology. The cosmological
parameters in the new cosmology can be set to any value, and then one parameter can be chosen to be modified so that the
original snapshot redshift is produced by the rescaling procedure.

This will also create tabulated files of the linear P(k) at z=0 in the original and target cosmologies, which is needed
later.

`get_cosmo.py` requires nbodykit (https://nbodykit.readthedocs.io/en/latest/)

## Calculating the density field (make_grid.py and merge_grid.py)

The OuterRim box has a box length of 3 Gpc/h, and it would require a large amound of memory to read in the whole snapshot
at once. In order to calculate the matter density field from the halo catalogue, it is therefore easier to calculate
the density field separately for each of the 110 OuterRim snapshot files, and then

`make_grid.py` can be run separately on each of the 110 OuterRim files to put the haloes on a grid

`merge_grid.py` can be run to combine the 110 grid files into a single file, which is then read in when applying the 
rescaling procedure.

If running this code on a smaller simulation, this step is not necessary, and `displace_simulation.py` can be modified
to calculate the grid, instead of reading it from a file.

## Rescaling the simulation snapshot (displace_simulation.py)

Once a file containing the haloes on a grid has been created, the main program `displace_simulation.py` can be run
to perform the rescaling procedure. Since the entire simulation is so large, this should be run separately for
each of the 110 OuterRim files.


