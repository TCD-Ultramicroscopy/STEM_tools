This code allows one to locate atomic columns positions on HAADF STEM images and to refine it against the idealized lattice with different degrees of freedom.

The distorted centered rectangular 2D Bravais lattice is in use here, with allowance of non-orthogonality ('nonortho'), modulations da,db describing a shift from center ('modulated'), and splitted positions ('dumbbells').

The algorithm of usage is the following:
 - Using the 'detect_columns' notebook - which is basically a wrapper for atomap - one can detect atomic columns positions, intensities and ellipticity for a given HAADF STEM image (TIFF files are default). This data exported as a numpy array
 - With the external calibrations provided (ratio between number of pixels and frame size is expected) one may call routines from fit_lattice, specifying a type of centering, need to handle some atomic columns separately, and starting lattice parameters
 - the theoretical lattice is then created and paired with observed one
 - the lattice parameters which are allowed to be flexible are refined to minimize the average distance between theoretical and observed lattices
 - outcome is a few vector plots of differences, some statistical data, and a number of csv files with refined parameters, ratios, and minimized value of distance

Known issues and TODO:

 - preview is needed
 - code cleanup is required
 - re-fit by 2D gaussian call from 'fit_lattice' is currently not working properly, to fix
 - image processing functions to be extracted to one lib, refinement-related to another
 - support for [110]PC lattice to be implemented
 - support for simultaneous imaging (BF/DF) to be considered
 - starting lattice parameters can be estimated by FFT

Acknowledgements:
 - Lewys Jones for ideas and supervision
 - Project SFI/21/US/3785 for financial support
