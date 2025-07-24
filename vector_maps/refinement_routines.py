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
from scipy.spatial.distance import cdist

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
	ij_ref = []
	for at in coord_cart:
		full_set.append(base_lat + at)
		ij_ref.append(ij)
	full_set = np.array(full_set)
	full_set = full_set.reshape(-1,2)
	
	ij_ref = np.array(ij_ref)
	ij_ref = ij_ref.reshape(-1,2)
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
		
		ij_ref_cr = ij_ref[mask]
	else:
		cr_lat = lat
		cr_ij = ij
		ij_ref_cr = ij_ref
		
	return cr_lat,cr_ij,ij_ref

'''
def old_filter_lat(obs,theor,param,max_d=0):
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
'''	
def filter_lat(ij,obs,param,max_d=0):
	#TODO param[i] is not reliable; has to be replaced by a,b ref
	if max_d == 0:
		max_d = np.sqrt((param[0]/4)**2+(param[1]/4)**2)


	theor,_,ij_ref = get_coords_from_ij(ij,param,None,crop=False)

	n = len(theor) // len(ij)
	n_arr = np.arange(n)
	motif = np.repeat(n_arr,len(ij))
	
	print('now, shapes: ij - %s, theor - %s' % (str(np.array(ij).shape), str(theor.shape)) )
	
	dist_matrix = cdist(obs, theor)
	min_dists = np.min(dist_matrix, axis=1)
	min_idxs = np.argmin(dist_matrix, axis=1)
	
	matched_theor = theor[min_idxs]
	matched_ij = ij_ref[min_idxs]
	matched_motif = motif[min_idxs]
	
	'''
	#lookup table shows for each exp atom the corresponding theor one (both by indexes)
	lookup_t = np.where(
		min_dists[:] <= max_d,
		min_idxs,
		np.nan
		)
	'''
	df = pd.DataFrame({
		'x_obs': obs[:, 0],
		'y_obs': obs[:, 1],
		'x_theor': matched_theor[:, 0],
		'y_theor': matched_theor[:, 1],
		'i': matched_ij[:, 0],
		'j': matched_ij[:, 1],
		'motif': matched_motif,
		'distance': min_dists,
		'lookup_t':min_idxs
		})
		
	#return lookup_t,ij_inuse
	df.loc[df['distance'] >= max_d, 'distance'] = np.nan
	df.loc[df['distance'] >= max_d, 'lookup_t'] = np.nan
	
	#legacy_output = np.array([df['x_theor'].values,df['y_theor'].values])
	
	return df

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
	
	theor_tmp,_,_ = get_coords_from_ij(ij_cr,corr_par,max_lim)
	theor = np.array([ theor_tmp[int(i)] for i in lookup_t if not np.isnan(i)])

	diff = cost_function(obs_cr,theor)
	return diff


def calculate_rel_diff(df,labels_raw,relative_to,kernel=1):
	labels_i = [i for i,_ in enumerate(labels_raw) ]
	relative_to_i = labels_raw.index(relative_to)	
	
	q = ['vdiff_xy', 'vproj', 'vdiff_xy_corr']
	q_ref = ['vdiff_xy_ref', 'vproj_ref', 'vdiff_xy_corr_ref']
	q_rel = ['vdiff_xy_rel', 'vproj_rel', 'vdiff_xy_corr_rel']

	for X,X_ref,X_rel in zip(q,q_ref,q_rel):
		#df[X_ref] = df.set_index(['i', 'j']).index.map(X)
		ref_q = df[df['motif'] == relative_to_i].set_index(['i', 'j'])[X]
		if kernel == 1:
			df[X_ref] = df.apply(lambda row: ref_q.get((row['i'], row['j']), np.nan), axis=1)
		elif kernel == 4:
			df[X_ref] = df.apply(lambda row: kernel4(ref_q, row['i'], row['j']), axis=1)
		df[X_rel] = df.apply(lambda row: np.array(row[X]) - np.array(row[X_ref]) if not np.any(np.isnan(row[X_ref])) else np.nan, axis=1)

	vdiff_xy_rel = np.array([i if not np.any(np.isnan(i)) else [np.nan,np.nan] for i in df['vdiff_xy_rel'].values])
	df['ang_rel'] = [np.arctan2(j,i) for i,j in np.array(vdiff_xy_rel)]
	vdiff_xy_corr_rel = [i if not np.any(np.isnan(i)) else [np.nan,np.nan] for i in df['vdiff_xy_rel'].values]
	df['ang_corr_rel'] = [np.arctan2(j,i) for i,j in np.array(vdiff_xy_corr_rel)]

	vdist_rel = np.sqrt(np.sum(vdiff_xy_rel**2,axis=1))
	df['vdist_rel'] = vdist_rel
	#print(df)

	return df
	
def kernel4(ref_q,i,j):
	l = [(i,j),(i+1,j),(i,j+1),(i+1,j+1)]
	#print([ref_q.get(d, np.nan) for d in l])

	s = [ref_q.get(d, [np.nan, np.nan]) for d in l]
	s = np.array(s,dtype=object)
	#print(s)
	s = s.astype(float)

	return np.sum(s, axis=0)/4.

def refinement_run(folder,sf,fname,calib,lat_params,motif,recall_zero=True,show_initial_spots=False,vec_scale=0.05,do_fit=True,relative_to=None,kernel=1,extra_shift_ab=None):
	if not do_fit:
		recall_zero=False
		
	s = load_frame(folder,fname,calib)
	
	dataset = np.load(folder+fname.split('.')[0]+'.npy').T
	#print(dataset.shape)
	if min(dataset.shape) == 2:
		print('No ellipticity found!')
		df_raw = pd.DataFrame(dataset, columns=['x_obs0', 'y_obs0'])
	else:
		df_raw = pd.DataFrame(dataset, columns=['x_obs0', 'y_obs0','ell0','rot0','I0'])
		#x0,y0,ell0,rot0,i_0 = np.load(folder+fname.split('.')[0]+'.npy')
		#except:
		#print('Historical data with no ellipticity provided')
		#x0,y0 = np.load(folder+fname.split('.')[0]+'.npy')
		#temporary workaround for comparison with some historical data

	observed_xy = np.array([ (i*calib,j*calib) for i,j in df_raw[['x_obs0', 'y_obs0']].values])
	df_raw[['x_obs', 'y_obs']] = observed_xy
	ij = gen_ij((-100,100))

	if recall_zero:
		bring_ith_atom_to_0 = int(len(observed_xy)/2)
		lat_params['base'] = [observed_xy[bring_ith_atom_to_0,0],observed_xy[bring_ith_atom_to_0,1],lat_params['base'][2]]
	
	#If extra shift is provided in fraq coordinates
	#we can convert it to (x,y) with the standard functionality as a r-vector to the (shx,shy) for u.c. with ij [0,0]
	if not extra_shift_ab is None:
		print(np.array(list(lat_params['abg'])+list(lat_params['base'])+list(extra_shift_ab)))
		tmp_val,_,_ = get_coords_from_ij(np.array([(0,0)]),np.array(list(lat_params['abg'])+list(lat_params['base'])+list(extra_shift_ab)),max_lim,crop=False)
		tmp_val = tmp_val[0]
		print(lat_params['base'])
		lat_params['base'] = [lat_params['base'][0]+tmp_val[0],lat_params['base'][1]+tmp_val[1],lat_params['base'][2]]
		print(lat_params['base'])
		
	#vectors to construct theor from ij
	param_vec,fit_param_vec = vectorize_params(lat_params,motif)
	
	#get roughly relevant ij - and theor - from the size estimation
	theor,ij_cr,_ = get_coords_from_ij(ij,param_vec,max_lim,crop=True)
	theor,_,_ = get_coords_from_ij(ij_cr,param_vec,max_lim,crop=False)
	
	
	tmp_df = filter_lat(ij_cr,observed_xy,param_vec,max_d=0)
	l1 = len(df_raw['x_obs'].values)
	l2 = len(tmp_df['x_obs'].values)
	if l1 != l2:
		print(l1,l2)
		raise IOError
	#print(df_raw,tmp_df)
	lookup_df = pd.merge(df_raw, tmp_df, on=['x_obs', 'y_obs'], how='inner')
	l3 = len(lookup_df['x_obs'].values)
	if l1 != l3:
		print(l1,l2,l3)
		raise IOError
	
	lookup_t = lookup_df['lookup_t'].values
	#print(lookup_df)
	#lookup_df_cl = lookup_df.copy()
	#lookup_df_cl = lookup_df_cl.dropna()
	
	#print(ij_inuse)
	
	#th_relevant = np.array([ theor[int(i)] for i in lookup_t if not np.isnan(i)])
	th_relevant = np.array(lookup_df[['x_theor','y_theor']].values)
	#print(th_relevant)
	obs_cr = np.array(lookup_df[['x_obs','y_obs']].values)#observed_xy[~np.isnan(lookup_t)]


	#ij_inuse_cleared = np.array(ij_inuse[~np.isnan(ij_inuse)]).reshape(-1,2).astype(int)
	#print(ij_inuse,ij_inuse_cleared)
	
	metadata = {}
	metadata['refined'] = do_fit
	metadata['relative'] = relative_to
	metadata['atoms_used'] = len(obs_cr)

	if do_fit:
		res = scipy.optimize.minimize(get_diff,param_vec, args=(param_vec,fit_param_vec,ij_cr,obs_cr,lookup_t,max_lim))
		#print(res)
		print('Residual',res.fun)
				
		metadata['residual_x1000'] = np.sqrt(res.fun)*1000
		param_vec = res.x
		#metadata['param'] = res.x
		#metadata['a/b'] = res.x[0]/res.x[1]
		#err = get_errors(res)

	metadata['param'] = param_vec

	#metadata['std'] = std

	if show_initial_spots:
		plt.scatter(observed_xy[:,0],observed_xy[:,1])
		plt.scatter(th_relevant[:,0],th_relevant[:,1])
		plt.show()
	
	
	# (i,j) hashed with its index in the cleared lookup
	#ij_to_index = {tuple(coord): idx for idx, coord in enumerate(ij_inuse_cleared)}
	
	#if show_initial_spots:
	#	plt.scatter(ij_inuse_cleared[:,0],ij_inuse_cleared[:,1])
	#	plt.show()
	#atomap sublattices
	obs_lat = am.Sublattice(observed_xy/calib, image=s, color='b')
	theor_lat = am.Sublattice(theor/calib, image=s, color='r') #before refinement, full
	theor_rel_lat = am.Sublattice(th_relevant/calib, image=s, color='r') #before refinement, filtered to paired ones
	

	theor_res,_,_ = get_coords_from_ij(ij_cr,param_vec.copy(),max_lim,crop=False)
	theor_res_rel = [ theor_res[int(i)] if not np.isnan(i) else (np.nan,np.nan) for i in lookup_t ]
	lookup_df[['x_theor_new','y_theor_new']] = np.array(theor_res_rel, dtype='float')
	#theor_res_rel = theor_res_rel[~np.isnan(theor_res_rel)]
	#print(lookup_df)
	
	refined_lat = am.Sublattice(np.array(theor_res)/calib, image=s, color='r') #refined full
	refined_rel_lat = am.Sublattice(np.array(theor_res_rel)/calib, image=s, color='r') #refined filtered to paired ones


	lat_params_fin,_ = unpack_vector(param_vec,lat_params,motif)
	phi = lat_params_fin['base'][2]
	
		
	#std,vdist,vdiff_xy,vdiff_ref,vdiff_xy_corr,ang,ang_corr = vector_map_calc(phi,obs_cr,theor_res_rel)
	std,diff_df= vector_map_calc(phi,lookup_df)

	#if not relative_to is None:
		
	metadata['std'] = std
	


	
	labels_raw = [i for i in motif.keys() if motif[i]['use'] ]
	types_raw = [ motif[i]['atom'] for i in labels_raw]


	
	if not relative_to is None:
		if not relative_to in labels_raw:
			raise IOError('Suggested reference atom position not found')
		else:
			if len(labels_raw) == 1:
				raise IOError('Only one lattice found, can not compute a relative diff')
			labels_i = [i for i,_ in enumerate(labels_raw) ]
			relative_to_i = labels_raw.index(relative_to)
			diff_df = calculate_rel_diff(diff_df,labels_raw,relative_to,kernel=kernel)
	
	diff_df = diff_df.dropna()			
	
	if relative_to is None:
		vdiff_xy = np.array(diff_df['vdiff_xy'].tolist())
		#print(vdiff_xy)
		th_relevant2 = np.array(diff_df[['x_theor_new','y_theor_new']].values)
		vdist = diff_df['vdist'].values
		ang = diff_df['ang'].values
		ang_corr = diff_df['ang_corr'].values
		vdiff_xy_corr = np.array(diff_df['vdiff_xy_corr'].tolist())
	else:
		vdiff_xy = np.array(diff_df['vdiff_xy_rel'].tolist())
		#print(vdiff_xy)
		th_relevant2 = np.array(diff_df[['x_theor_new','y_theor_new']].values)
		ang = diff_df['ang_rel'].values
		vdist = diff_df['vdist_rel'].values
		ang_corr = diff_df['ang_corr_rel'].values
		vdiff_xy_corr = np.array(diff_df['vdiff_xy_corr_rel'].tolist())
		
	if not sf is None:
		if do_fit:
			export_data(folder,sf,fname,res.x,lat_params,motif,metadata)
		else:
			export_data(folder,sf,fname,param_vec,lat_params,motif,metadata)

		if relative_to is None:	
			plot_lattice(s,[obs_lat,theor_lat],fname,folder,sf,'initial_guess_full_'+sf)
			plot_lattice(s,[obs_lat,refined_rel_lat],fname,folder,sf,'fit_'+sf)
	
	
		
		file_s = folder + sf + '/' +fname +'_'+sf
		plot_stats_rep(vdist,file_s+'_diff',ang=False,ang_weights=None)
		plot_stats_rep(ang,file_s+'_angles',ang=True,ang_weights=None)
		
		if 'ell0' in diff_df.columns and 'rot0' in diff_df.columns:
			f_el = diff_df['ell0'].values
			f_rot = diff_df['rot0'].values
			f_el = [(i,j) for i,j in zip(f_el * np.cos(f_rot),-f_el * np.sin(f_rot))]
			plot_quiver(file_s + '_ellipticity',th_relevant2,f_el,f_rot,1,units_v='rel.u.',ell=True)
		
		
		plot_quiver(file_s + '_vmap_abs',th_relevant2,vdiff_xy,ang,vec_scale,hd_w=2,units_v='$1 \AA$',ell=False)
		plot_quiver(file_s + '_vmap_rotated',th_relevant2,vdiff_xy_corr,ang_corr,vec_scale,hd_w=2,units_v='$1 \AA$',ell=False)
		
		#_corr meant to be aligned with OX already, compensating phi
		phi = phi*np.pi/180
		base_x = np.array([np.cos(phi),np.sin(phi)])
		base_y = np.array([np.cos(phi+np.pi/2),np.sin(phi+np.pi/2)])
		
		proj_a = np.dot(vdiff_xy,base_x)
		proj_a = base_x*proj_a[:,None]
		
		proj_a90 = np.dot(vdiff_xy,base_y)
		proj_a90 = base_y*proj_a90[:,None]
				
		plot_quiver(file_s + '_vmap_proj_a',th_relevant2,proj_a,ang,vec_scale,hd_w=2,units_v='$1 \AA$',ell=False)
		plot_quiver(file_s + '_vmap_proj_a90',th_relevant2,proj_a90,ang,vec_scale,hd_w=2,units_v='$1 \AA$',ell=False)
		
		if 'I0' in diff_df.columns:
			at_labels = [i+'\n'+j for i,j in zip(labels_raw,types_raw)]
			plot_violin(file_s + '_I0_vor',at_labels,diff_df)
	
	if do_fit:
		return metadata, res.x
	
	
