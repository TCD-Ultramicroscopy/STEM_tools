#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

from routines import *
from refinement_routines import *
from plot_routines import *

#Size of a virtual set of points; has to be large for large imgs, but more demanding
#ij_max = 300
#ij_max = 100

###############################################
# Here we are starting the refinement
##############################################

#Meant to be 0.3867, 0.5469
lat_params = { 'abg':[0.3805, 0.5369, 89.75],
		'fit_abg':[True,True,True],
		'base':[-0.0005,.20,1.88],
		'fit_base':[True,True,True]
}

#Atom at (0,0); first sublattice. Since all other atoms are functionally connected to this one,
#it is reasonable to fix it due to a full correlation with shx/shy (lat_params['base'][0] and lat_params['base'][1])

motif = {'A_1':{'atom':'Si_1',
			'coord':(0.,0.),
			'I':1,
			'use':True,
			'fit':[False,False]},
}

#Centered atom of the first sublattice. Since 'eq' are present and not None, 'coords' and 'fit' are disabled
motif['A_1c'] = {	'atom':'Si_2',
			'coord':(0.,0.),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':  ["= motif['A_1'][0] + extra_pars['centering_a']", "= motif['A_1'][1] + extra_pars['centering_b']"]
			}
			
#Second sublattice; 'A_1' + dumbbell vector (in polar coordinates)
motif['B_1'] =  {'atom':'Si_3',
			'coord':(0.,0.2),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':["= motif['A_1'][0] + extra_pars['db_dist']*np.sin(extra_pars['db_angle']/180*np.pi)/lat_params['abg'][0]",
					"= motif['A_1'][1] + extra_pars['db_dist']*np.cos(extra_pars['db_angle']/180*np.pi)/lat_params['abg'][1]"]}

#Second sublattice; centered
motif['B_1c'] = {'atom':'Si_4',
			'coord':(0.5,0.7),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':["= motif['B_1'][0] + extra_pars['centering_a']", "= motif['B_1'][1] + extra_pars['centering_b']"]}

#Extra variables - dumbbell vector in absolute polar coordinated relative to b; expected to be (L,0) but can be refined
#Centering vector in fractional coordinates
#True/False enables/disables refinement

extra_pars = {'db_dist':(0.1,True),
		'db_angle':(0,True),
		'centering_a':(0.5,True),
		'centering_b':(0.5,True)}
		
		
print(motif)


folder = '/home/vasily/test_vmap/'
fname = 'test_Si'

#Calibrations. Only ratio is important
#this 1024px frame was acquired from 90% area of 16nm scan
calib = 16/1024*.9


#Manual presets for shifts/lattice rotations
sub_area = [2,4,2,4]#in nm

save_folder_name = None
_,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params,motif,extra_pars=extra_pars,
					show_initial_spots=True,vec_scale=0.1,sub_area=sub_area,max_dist=0.1)
lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)
print(lat_params_prefit,motif_prefit,extra_pars_prefit)


#Automated refinements with gradual expansion of the ROI
st_p = 2
r = 2
k = st_p+r
while k<16:
	sub_area = [st_p,k,st_p,k]
	print('Auto sub-area',sub_area)
	save_folder_name = None
	
	lat_params_prefit['fit_abg'] = [False,False,False]
	
	_,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit,motif_prefit,extra_pars=extra_pars_prefit,
						show_initial_spots=False,vec_scale=0.01,sub_area=sub_area,max_dist=0.1)
	lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)
	
	lat_params_prefit['fit_abg'] = [True,True,True]
	
	_,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit,motif_prefit,extra_pars=extra_pars_prefit,
						show_initial_spots=False,vec_scale=0.01,sub_area=sub_area,max_dist=0.1)
	lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)
	
	k+=2

#Full image refinement with a preview and outputs
sub_area = None
save_folder_name = fname+'_fix_motif'
_,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params,motif,extra_pars=extra_pars_prefit,
					show_initial_spots=True,vec_scale=0.1,sub_area=sub_area,max_dist=0.1)
lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)

#central area refinement
sub_area = [2,12,2,12]
#sub_area = [1,6,1,6]

save_folder_name = fname+'_fix_motif_center'
_,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params,motif,extra_pars=extra_pars_prefit,
					show_initial_spots=True,vec_scale=0.1,sub_area=sub_area,max_dist=0.1)
lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)


