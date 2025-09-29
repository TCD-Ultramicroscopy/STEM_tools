This code meant to:

 - import crystallographic data from a given cif file
 - expand the lattice to create a large set (superblock) of atoms
 - rotate a superblock to direct a given uvw (or normal to hkl, depends on settings) along Z
 - crop a superblock to the given 'lamella' sizes
 - perform a simulation using abTEM routines

The main benefit of this code is the possibility to deal with non-orthogonal space groups
Trigonal symmentry has not been assessed yet; looking forward to hear any feedback on that! 

Known issues and TODO:
 - (done) move most variables to JSON
 - add a support of variables as lists to iterate over all combinations
 - fix the issue with random_seed for vacancies generator
 - add a possibility of different types of vacancies simultaneously
 - add a way to import an ase set
 - output file names to reflect was it uvw or hkl
 - first frame is simulated separately from frozen phonons, and this simulation is just repeated later on.
	maybe add a flag?
 - (done) dry\_run should be implemented as a flag in a full_run
 - (done) separated lib file to be created
 - gaussian blur is not handling borders correctly
 - BF images to be confirmed
 - for the in-plane rotation to add 'this hkl up' functionality
 - check imported libraries

Acknowledgements:
 - Julie M. Bekkevold for the invalueable help with ab-initio simulations and guidance with abTEM code
 - Project SFI/21/US/3785 for financial support
