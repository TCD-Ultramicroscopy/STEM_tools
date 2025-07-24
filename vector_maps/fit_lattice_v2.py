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

lat_params = { 'abg':[0.4108,0.187,90.1],
		'fit_abg':[True,True,True],
		'base':[0,0,-96.96],
		'fit_base':[True,True,True]
}

motif = {'A_1':{'atom':'Zr',
			'coord':(0.,0.),
			'I':1,
			'use':True,
			'fit':[True,True]},
	'B_1':{'atom':'Pb',
			'coord':(1/2.,1/2.),
			'I':1,
			'use':True,
			'fit':[True,True]}
}
print(motif)

folder = '/home/vasily/test_fit_atomap/'
fname = 'test_frame'#the presence of .npy file is essential; tiff file is expected for previews


#Calibrations. Only ratio is important
calib_px = 1024
calib_size = 16*.9 #nm!
calib = calib_size/calib_px

save_folder_name = None
_,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params,motif,
					show_initial_spots=False,vec_scale=0.1)
lat_params_prefit,motif_prefit = unpack_vector(lat_params_vec,lat_params,motif)

save_folder_name = 'free_all'
metadata,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit,motif_prefit,
						show_initial_spots=True,vec_scale=0.1,extra_shift_ab=(0.25,0.25))
lat_params_prefit,motif_prefit = unpack_vector(lat_params_vec,lat_params,motif)

save_folder_name = 'Zr_fixed'
lat_params_prefit_tmp,motif_prefit_tmp = unpack_vector(lat_params_vec,lat_params,motif)
motif_prefit_tmp = copy.deepcopy(motif)
motif_prefit_tmp['A_1']['fit'] = [False,False]
refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit_tmp,motif_prefit_tmp,
						show_initial_spots=False,vec_scale=0.1)
						
save_folder_name = 'Ortho'
lat_params_prefit_tmp,motif_prefit_tmp = unpack_vector(lat_params_vec,lat_params,motif)
#motif_prefit_tmp['A_1']['fit'] = [False,False]
lat_params_prefit_tmp['abg'] = [lat_params_prefit_tmp['abg'][0],lat_params_prefit_tmp['abg'][1],90]
lat_params_prefit_tmp['fit_abg'] = [True,True,False]
refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit_tmp,motif_prefit_tmp,
						show_initial_spots=False,vec_scale=0.1)

save_folder_name = 'fixed_motif'
lat_params_prefit_tmp,motif_prefit_tmp = unpack_vector(lat_params_vec,lat_params,motif)
motif_prefit_tmp = copy.deepcopy(motif)
motif_prefit_tmp['A_1']['fit'] = [False,False]
motif_prefit_tmp['B_1']['fit'] = [False,False]
refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit_tmp,motif_prefit_tmp,
						show_initial_spots=False,vec_scale=0.1)

save_folder_name = 'fixed_motif_ortho'
lat_params_prefit_tmp,motif_prefit_tmp = unpack_vector(lat_params_vec,lat_params,motif)
motif_prefit_tmp = copy.deepcopy(motif)
motif_prefit_tmp['A_1']['fit'] = [False,False]
motif_prefit_tmp['B_1']['fit'] = [False,False]
lat_params_prefit_tmp['abg'] = [lat_params_prefit_tmp['abg'][0],lat_params_prefit_tmp['abg'][1],90]
lat_params_prefit_tmp['fit_abg'] = [True,True,False]
print(motif_prefit_tmp)
print(motif)

refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit_tmp,motif_prefit_tmp,
						show_initial_spots=False,vec_scale=0.1)
						
refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit_tmp,motif_prefit_tmp,
						do_fit=False,recall_zero=False,show_initial_spots=False,vec_scale=0.1)
