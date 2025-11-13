This code allows one to locate atomic columns positions on HAADF STEM images and to refine it against the idealized lattice with different degrees of freedom.

2D Bravais lattice is in use here, with allowance of arbitrary cell and motif

The algorithm of usage is the following:
 - Using the 'detect_columns' notebook - which is basically a wrapper for atomap - one can detect atomic columns positions, intensities and ellipticity for a given HAADF STEM image (TIFF files are default). This data exported as a numpy array
 - With the external calibrations provided (ratio between number of pixels and frame size is expected) one may call routines from fit_lattice, specifying starting lattice parameters
 - the theoretical lattice is then created and paired with observed one
 - the lattice parameters which are allowed to be flexible are refined to minimize the average distance between theoretical and observed lattices
 - outcome is a few vector plots of differences, some statistical data, and a number of csv files with refined parameters, ratios, and minimized value of distance

Known issues and TODO:

 - [major] revisit the handling of parameters to allow equations
 - code cleanup is required
 - more comments needed
 - re-fit by 2D gaussian call from 'fit_lattice' is currently not working properly, to fix
 - support for simultaneous imaging (BF/DF) to be considered
 - starting lattice parameters can be estimated by FFT
 - residuals of 2D gauss fit to be visualized
 - average unit cell picture to be rendered
 - add auto-assessment for max/min i,j values (number of unit cells in use)
 - (?) use diffpy to handle fraq coordinates
 - use a dedicated min for negative distances
 - check extra shift
 - double-check rotations for images and vmaps
 - (done; keep track) double-check kernel4 function, which computes difference on the basis of 4 nearest neighbours
 - (?) dask for pandas
 
Acknowledgements:
 - Lewys Jones for ideas and supervision
 - Project SFI/21/US/3785 for financial support

Part of the code has been created with AI assistance (OpenAI GPT-5) and manually reviewed
