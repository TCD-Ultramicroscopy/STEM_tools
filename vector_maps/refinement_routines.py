#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import numpy as np
import hyperspy.api as hs
import atomap.api as am
import atomap.initial_position_finding as ipf
#from scipy.spatial.transform import Rotation as R
import scipy
import matplotlib.pyplot as plt

from routines import *
from plot_routines import *

max_lim=(100,100)

def gen_ij(ij_range):
	
	i_range = np.arange(ij_range[0],ij_range[1])
	j_range = np.arange(ij_range[0],ij_range[1])
	
	I, J = np.meshgrid(i_range, j_range, indexing='ij')
	
	ij_set = np.stack((I.ravel(), J.ravel()), axis=-1)
	
	return ij_set

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


