#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import copy
import numpy as np
import hyperspy.api as hs
import atomap.api as am
import atomap.initial_position_finding as ipf
from scipy.spatial.transform import Rotation as R
import scipy
import pandas as pd

from routines import *
from plot_routines import *

#from scipy.stats import rayleigh
#from matplotlib.colors import DivergingNorm


test_vector_map=False

max_lim=(100,100)

#Size of a virtual set of points; has to be large for large imgs, but more demanding
#ij_max = 300
ij_max = 100


def gen_ij(ij_range):
	
	i_range = np.arange(ij_range[0],ij_range[1])
	j_range = np.arange(ij_range[0],ij_range[1])
	
	I, J = np.meshgrid(i_range, j_range, indexing='ij')
	
	ij_set = np.stack((I.ravel(), J.ravel()), axis=-1)
	
	return ij_set

def vectorize_params(lat_params,motif):
	lat_vec = np.concatenate((lat_params['abg'],lat_params['base']))
	lat_fit = np.concatenate((lat_params['fit_abg'],lat_params['fit_base']))
	
	motif_keys = list(motif.keys())
	#print(motif_keys)
	#motif_use = [motif[i]['use'] for i in motif_keys]
	motif_vec = np.concatenate([motif[i]['coord'] for i in motif_keys if motif[i]['use'] ])
	motif_fit = np.concatenate([motif[i]['fit'] for i in motif_keys if motif[i]['use'] ])
	#print(motif_vec)
	
	#we also can add here arbitrary intensities
	
	param_vec = np.concatenate((lat_vec,motif_vec))
	fit_vec = np.concatenate((lat_fit,motif_fit))
	
	return param_vec,fit_vec

def unpack_vector(param_vec,lat_params,motif):	
	abg = param_vec[:3]
	sh = param_vec[3:6]
	lat_params['abg'] = abg
	lat_params['base'] = sh

	coords_vec = param_vec[6:]
	coord_fraq = coords_vec.reshape(-1,2)
	
	k = [i for i in list(motif.keys()) if motif[i]['use']]
	
	motif2 = copy.deepcopy(motif)
	c = 0
	while c <len(coord_fraq):
		motif2[k[c]]['coord'] = coord_fraq[c]
		c+=1

	return lat_params,motif2
	
def get_coords_from_ij(ij,param_vec,max_lim,crop=False):
	#de-vectorize
	a,b,gamma = param_vec[:3]
	shx,shy,phi = param_vec[3:6]

	phi = phi/180.*np.pi
	gamma = gamma/180.*np.pi
	
	coords_vec = param_vec[6:]
	coord_fraq = coords_vec.reshape(-1,2)
	
	#fractional to cartesian, for all atoms within u.c.
	fraq_a = coord_fraq[:,0]
	fraq_b = coord_fraq[:,1]
	cart_x = fraq_a*a + fraq_b*b*np.cos(gamma)
	cart_y = fraq_b*b*np.sin(gamma)
	coord_cart = np.stack((cart_x,cart_y), axis = -1)
	#print(coord_cart)
	
	#lattice reference points from ij to cartesian
	i = ij[:,0]
	j = ij[:,1]
	x = i*a + j*b*np.cos(gamma)
	y = b*j*np.sin(gamma)
	base_lat = np.stack((x,y), axis = -1)
	#print(str(base_lat.shape)+' base_lat')
	
	#add all atoms to the set	
	full_set = []
	for at in coord_cart:
		full_set.append(base_lat + at)
	full_set = np.array(full_set)
	full_set = full_set.reshape(-1,2)
	
	#print(str(full_set.shape)+' full_lat')
	
	#rotate the whole set; vectorized now
	#lat = np.array([ (x*np.cos(phi) + y*np.sin(phi), y*np.cos(phi) - x*np.sin(phi)) for x,y in lat])
	lat_x = full_set[:,0]
	lat_y = full_set[:,1]
	
	fin_lat_x = lat_x*np.cos(phi) + lat_y*np.sin(phi)
	fin_lat_y = lat_y*np.cos(phi) - lat_x*np.sin(phi)

	lat = np.stack((fin_lat_x,fin_lat_y), axis = -1)
	
	#shift the set
	lat = lat + (shx,shy)	
			
	if crop:
		###!TODO minlim

		mask = np.all(lat>=(-5,-5),axis=1)*np.all(lat<=max_lim,axis=1)
	
		mask_r = mask.reshape(len(mask) // len(ij),len(ij))
		mask_ij = np.any(mask_r, axis=0)	

		cr_lat = lat[mask] #More strict then the reiterate non-crop over cropped ij!
		##!TODO consider a recurrent version
		cr_ij = np.array(ij)[mask_ij]
	else:
		cr_lat = lat
		cr_ij = ij
		
	return cr_lat,cr_ij


def filter_lat(obs,theor,param,max_d=0):
	if max_d == 0:
		max_d = np.sqrt((param[0]/4)**2+(param[1]/4)**2)
		
	#lookup table shows for each exp atom the corresponding theor one (both by indexes)
	lookup_t = []
	
	i = 0
	while i < len(obs):
		at = obs[i]
		#for at in obs:
		dist = np.array(theor)-np.array(at)
		dist = np.sqrt(dist**2)
		dist = np.sum(dist,axis=1)
		
		if min(dist)<=max_d:
			t = theor[dist<=min(dist)+.000001]
			if len(t)>1:
				print('Err! len>1', t)
			lookup_t.append( np.where( np.isclose(dist, min(dist)) )[0][0] )
		else:
			lookup_t.append(np.nan)
		i += 1
		
	#lookup_t = np.array(lookup_t,dtype='int')
	
	#at the moment, all observed are returned, even if the nearest one is far away... tbf
	#print('Sanity check,lens shall be equal',len(f_obs),len(f_theor))
	return lookup_t

def cost_function(f_obs,f_theor):
	diff = np.array(f_obs)-np.array(f_theor)
	diff = np.sum(diff**2,axis=1)
	
	#sqrt - linear dist
	#diff = np.sqrt(diff)
	#square dist now
	tot_dist = sum(diff)/len(diff)
	return tot_dist


def get_diff(par,raw_par,fit_flags,ij_cr,obs_cr,lookup_t,max_lim):
	#we need to variate only those params selected by fit booleans
	corr_par = np.array([ par[i] if fit_flags[i] else raw_par[i] for i in np.arange(len(fit_flags)) ])
	
	theor_tmp,_ = get_coords_from_ij(ij_cr,corr_par,max_lim)
	theor = np.array([ theor_tmp[i] for i in lookup_t if not np.isnan(i)])

	diff = cost_function(obs_cr,theor)
	return diff
		


def refinement_run(folder,sf,fname,calib,lat_params,motif,show_initial_spots=False,vec_scale=0.05):

	s = load_frame(folder,fname,calib)
	x0,y0,ell0,rot0,i_0 = np.load(folder+fname.split('.')[0]+'.npy')

	observed_xy = np.array([ (i*calib,j*calib) for i,j in zip(x0,y0)])

	ij = gen_ij((-100,100))

	bring_ith_atom_to_0 = 0
	lat_params['base'] = [observed_xy[bring_ith_atom_to_0,0],observed_xy[bring_ith_atom_to_0,1],lat_params['base'][2]]

	param_vec,fit_param_vec = vectorize_params(lat_params,motif)
	theor,ij_cr = get_coords_from_ij(ij,param_vec,max_lim,crop=True)
	theor,_ = get_coords_from_ij(ij_cr,param_vec,max_lim,crop=False)

	lookup_t = filter_lat(observed_xy,theor,param_vec,max_d=0)

	#print(lookup_t)
	th_relevant = np.array([ theor[i] for i in lookup_t if not np.isnan(i)])
	obs_cr = observed_xy[~np.isnan(lookup_t)]
	res = scipy.optimize.minimize(get_diff,param_vec, args=(param_vec,fit_param_vec,ij_cr,obs_cr,lookup_t,max_lim))
	#print(res)
	print('Residual',res.fun)
	
	metadata = {}
	metadata['atoms_used'] = len(obs_cr)
	metadata['residual_x1000'] = np.sqrt(res.fun)*1000
	metadata['param'] = res.x
	metadata['a/b'] = res.x[0]/res.x[1]
	#err = get_errors(res)
	
	#std,fin_lat,vdiff_xy,vdiff_ref,vdiff_xy_corr,ang,ang_corr,str_mean = vector_map_calc(fin_par,fname_save,f_ij_g,f_obs_g,no_modulation,only_ortho,max_lim)
	#metadata['std'] = std

	if show_initial_spots:
		plt.scatter(observed_xy[:,0],observed_xy[:,1])
		plt.scatter(th_relevant[:,0],th_relevant[:,1])
		plt.show()
	
	obs_lat = am.Sublattice(observed_xy/calib, image=s, color='b')
	theor_lat = am.Sublattice(theor/calib, image=s, color='r') #before refinement, full
	theor_rel_lat = am.Sublattice(th_relevant/calib, image=s, color='r') #before refinement, filtered to paired ones
	
	
	theor_res,_ = get_coords_from_ij(ij_cr,res.x.copy(),max_lim,crop=False)
	theor_res_rel = np.array([ theor_res[i] for i in lookup_t if not np.isnan(i)])
	
	refined_lat = am.Sublattice(np.array(theor_res)/calib, image=s, color='r') #refined full
	refined_rel_lat = am.Sublattice(np.array(theor_res_rel)/calib, image=s, color='r') #refined filtered to paired ones


	lat_params_fin,_ = unpack_vector(res.x,lat_params,motif)
	phi = lat_params_fin['base'][2]
	std,vdist,vdiff_xy,vdiff_ref,vdiff_xy_corr,ang,ang_corr = vector_map_calc(phi,obs_cr,theor_res_rel)
	metadata['std'] = std
	
	if not sf is None:
		export_data(folder,sf,fname,res.x,lat_params,motif,metadata)
			
		plot_lattice(s,[obs_lat,theor_lat],fname,folder,sf,'initial_guess_full_'+sf)
		plot_lattice(s,[obs_lat,refined_rel_lat],fname,folder,sf,'fit_'+sf)
		
		file_s = folder + sf + '/' +fname +'_'+sf
		plot_stats_rep(vdist,file_s+'_diff',ang=False,ang_weights=None)
		plot_stats_rep(ang,file_s+'_angles',ang=True,ang_weights=None)
		

		plot_quiver(file_s + '_vmap_abs',th_relevant,vdiff_xy,ang,vec_scale,hd_w=2,units_v='$1 \AA$',ell=False)
		plot_quiver(file_s + '_vmap_rotated',th_relevant,vdiff_xy_corr,ang_corr,vec_scale,hd_w=2,units_v='$1 \AA$',ell=False)
	
	return metadata, res.x

def export_data(folder,sf,fname,lat_params_vec,raw_lat_params,raw_motif,metadata):

	#check the existance of the output folder
	l = os.listdir(folder)
	if not sf in l:
		os.mkdir(folder+sf)
		print('Folder %s created' % sf)

	lat_params_fin,motif_fin = unpack_vector(lat_params_vec,raw_lat_params,raw_motif)
	
	export_name = folder+sf+'/'+fname.split('.')[0]+'_'+sf
	
	mdata = pd.DataFrame.from_dict(metadata,orient='index')
	mdata.to_csv(export_name +'_metadata.csv',sep='\t')
	
	par = pd.DataFrame.from_dict(lat_params_fin,orient='index')
	par.to_csv(export_name + '_lattice.csv',sep='\t')
	
	motif = pd.DataFrame.from_dict(motif_fin,orient='index')
	motif.to_csv(export_name + '_motif.csv',sep='\t')

def vector_map_calc(phi,obs,calc):
	#phi = param[2]/180*np.pi
	#fin_lat = get_coords_from_ij(f_ij,param,no_modulation,only_ortho,max_lim)[0]
	
	vdiff_xy = obs - calc
	vdiff_ref = np.mean(vdiff_xy,axis=0)
	
	vproj = [(x*np.cos(phi) + y*np.sin(phi), y*np.cos(phi) - x*np.sin(phi)) for x,y in vdiff_xy]
	print(np.std(vproj,axis=0))
	print(np.std(vdiff_xy,axis=0))
	print('ref',vdiff_ref)
	print('Std rot',np.sqrt(np.sum(np.std(vproj,axis=0)**2)),'len',len(vproj))
	print('Std raw',np.sqrt(np.sum(np.std(vdiff_xy,axis=0)**2)),'len',len(vdiff_xy))
	
	#std_to_report = np.std(vproj,axis=0)
	
	
	vdist = np.sqrt(np.sum(vdiff_xy**2,axis=1))
	vdiff_xy_corr = vdiff_xy - vdiff_ref
	print('Test dist',sum(vdist)/len(vdist))
	#print(np.mean(vdiff_xy_corr,axis=0))
	
	#print(np.std(abs(vdiff_xy_corr),axis=0))
	ang = [np.arctan2(j,i) for i,j in vdiff_xy]
	ang_corr = [np.arctan2(j,i) for i,j in vdiff_xy_corr] #np.angle(vdiff_xy, deg=True)
	
	std_to_report = np.std(abs(vdiff_xy),axis=0)
	#vdist = np.sqrt(np.sum(vdiff_xy**2,axis=1))
	#str_mean = plot_stats_rep(vdist,fname_save)
	
	return std_to_report,vdist,vdiff_xy,vdiff_ref,vdiff_xy_corr,ang,ang_corr





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
						show_initial_spots=True,vec_scale=0.1)
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
