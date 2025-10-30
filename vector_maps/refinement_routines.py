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
from scipy.spatial import cKDTree
import cv2

from routines import *
from plot_routines import *


from matplotlib.widgets import Slider, Button

max_lim=(1000,1000)

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
def remove_close_points(points, threshold=1e-4):
	points = np.asarray(points, dtype=float)
	N = len(points)
	if N == 0:
		print('Empty array of points!')
		raise IOError

	# Pairwise distance matrix
	D = cdist(points, points, metric='euclidean')

	# We'll keep the first occurrence and drop later near-duplicates
	keep = np.ones(N, dtype=bool)

	for i in range(N):
		if not keep[i]:
			continue
		# mark all later points within threshold as duplicates
		close_mask = (D[i, i+1:] < threshold)
		keep[i+1:][close_mask] = False

	return points[keep]
'''

def remove_close_points(points, threshold=1e-4):
	points = np.asarray(points, dtype=float)
	tree = cKDTree(points)
	pairs = tree.query_pairs(threshold)   # all pairs within threshold

	# Collect indices to remove (keep the first occurrence)
	to_remove = set()
	for i, j in pairs:
		# always remove the later index (arbitrary rule)
		to_remove.add(j)

	mask = np.ones(len(points), dtype=bool)
	mask[list(to_remove)] = False
	return points[mask]
	
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
	
	#Test, just to make sure if there is no problem
	if len(min_idxs) != len(np.unique(min_idxs)):
		print('Repeated matches!',len(min_idxs) - len(np.unique(min_idxs)))
		#min_idxs = np.unique(min_idxs)
		#raise IOError
		unique_js, counts = np.unique(min_idxs, return_counts=True)
		colliding_js = unique_js[counts > 1]          # theor indices with collisions
		ambiguous_mask = np.isin(min_idxs, colliding_js)
		matched_theor[ambiguous_mask] = (np.nan,np.nan)		


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
	
	#We also need to mask the reference
	df.loc[df['motif'] == relative_to_i, 'x_obs'] = np.nan
	
	return df
	
def kernel4(df,i,j):##TODO##TODO###TODO###
	l = [(i,j),(i+1,j),(i,j+1),(i+1,j+1)]
	#print([ref_q.get(d, np.nan) for d in l])
	#print(df)
	#s = [ref_q.get(d, [None, None]).values for d in l]
	#look = pd.MultiIndex.from_tuples(l, names=['i', 'j'])
	#picked = df.reindex(look)
	#print(df.columns)
	#s = [np.array(x) for x in df.reindex(l).to_numpy()]
	try:
		s = np.array([np.array(df.loc[d]) if d in df.index else np.array([np.nan, np.nan]) for d in l ])
	except:
		s = np.array([np.nan,np.nan])
	#s = np.stack([np.asarray(df[d].values if d in df.index else [None, None]) for d in l])
	#print(s)
	#s = np.array(s,dtype=float)
	#print(s)
	#s = picked['val'].to_numpy()
	#print(s)
	try:
		return np.sum(s, axis=0)/4.
	except:
		return np.array([np.nan,np.nan]) #sometimes there is a 'list' appearing here ##BUG### 

def preprocess_dataset(lat_params,motif,dataset,calib,recall_zero=False,extra_shift_ab=None,max_dist=0,sub_area=None):
	#TODO proper pandas
	#print(dataset.shape)
	if min(dataset.shape) == 2:
		print('No ellipticity found!')
		df_raw = pd.DataFrame(dataset, columns=['x_obs0', 'y_obs0'])
	elif min(dataset.shape) == 5:
		print('It seems that gaussian Is are not exported')
		df_raw = pd.DataFrame(dataset, columns=['x_obs0', 'y_obs0','ell0','rot0','I0'])
		#x0,y0,ell0,rot0,i_0 = np.load(folder+fname.split('.')[0]+'.npy')
	else:
		df_raw = pd.DataFrame(dataset, columns=['x_obs0', 'y_obs0','ell0','rot0','I_gauss','I0'])
		
	observed_xy = np.array([ (i*calib,j*calib) for i,j in df_raw[['x_obs0', 'y_obs0']].values])
	len1 = len(observed_xy)
	observed_xy = remove_close_points(observed_xy)
	len2 = len(observed_xy)
	if len2 != len1:
		print('A few observed points were omitted due to repeat: ',str(len1-len2))
	
	df_raw[['x_obs', 'y_obs']] = observed_xy
	
	if not sub_area is None:
		df_raw.loc[df_raw['x_obs'] < sub_area[0], 'x_obs'] = np.nan
		df_raw.loc[df_raw['x_obs'] > sub_area[1], 'x_obs'] = np.nan
		df_raw.loc[df_raw['y_obs'] < sub_area[2], 'y_obs'] = np.nan
		df_raw.loc[df_raw['y_obs'] > sub_area[3], 'y_obs'] = np.nan
		df_raw = df_raw.dropna()
		observed_xy = np.array([ (i,j) for i,j in df_raw[['x_obs', 'y_obs']].values])
		
		
	ij = gen_ij((-170,170))

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
	print(param_vec)
	#get roughly relevant ij - and theor - from the size estimation
	theor,ij_cr,_ = get_coords_from_ij(ij,param_vec,max_lim,crop=True)
	theor,_,_ = get_coords_from_ij(ij_cr,param_vec,max_lim,crop=False)
	
	
	tmp_df = filter_lat(ij_cr,observed_xy,param_vec,max_d=max_dist)
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
	
	#lookup_t = lookup_df['lookup_t'].values
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
	
	return ij_cr, th_relevant, observed_xy, obs_cr, lookup_df



def refinement_run(folder,sf,fname,calib,lat_params,motif,recall_zero=False,show_initial_spots=False,vec_scale=0.05,
			do_fit=True,relative_to=None,kernel=1,extra_shift_ab=None,sub_area=None,max_dist=0):
	if not do_fit:
		recall_zero=False
	
		
	s = load_frame(folder,fname,calib).T#!TODO is .T needed?
	
	dataset = np.load(folder+fname.split('.')[0]+'.npy').T

	ij_cr, th_relevant, observed_xy, obs_cr, lookup_df = preprocess_dataset(lat_params,motif,dataset,calib,recall_zero=recall_zero,
									extra_shift_ab=extra_shift_ab,max_dist=max_dist,sub_area=sub_area)
	


	if show_initial_spots:
		
		m_zeros = {'0':{'coord':(0.,0.),
				'use':True,
				'fit':[False,False]}}
		param_vec_zeros,_ = vectorize_params(lat_params,m_zeros)
		zeros,_,_ = get_coords_from_ij(ij_cr,param_vec_zeros,max_lim,crop=False)
		
		_im = cv2.imread(folder+fname+'.tif', cv2.IMREAD_UNCHANGED)
		H, W = _im.shape[:2]
		
		fig, ax = plt.subplots(figsize=(6, 4))
		fig.subplots_adjust(bottom=0.27) 
		
		ax.set_aspect('equal')
		ax.scatter(observed_xy[:,0],observed_xy[:,1], marker='o',s=50, edgecolors="blue", facecolors="none", linewidths=2)
		sc0 = ax.scatter(zeros[:,0],zeros[:,1], marker='o',s=50, edgecolors="k", facecolors="none", linewidths=3)
		
		
		
		
		sc = ax.scatter(th_relevant[:,0],th_relevant[:,1], marker='o',s=50, color='r')
		
		ax.imshow(_im.T,extent=[0, W*calib, 0, H*calib],origin='upper')#,origin='lower')
		
		ax_shx = fig.add_axes([0.05, 0.12, 0.4, 0.03])
		ax_shy = fig.add_axes([0.05, 0.07, 0.4, 0.03])
		ax_ph = fig.add_axes([0.05, 0.02, 0.4, 0.03])
		
		ax_a = fig.add_axes([0.55, 0.12, 0.45, 0.03])
		ax_b = fig.add_axes([0.55, 0.07, 0.45, 0.03])
		ax_g = fig.add_axes([0.55, 0.02, 0.45, 0.03])
		

		s_a = Slider(ax_a, 'a', valmin=lat_params['abg'][0]*.5, valmax=lat_params['abg'][0]*1.5, valinit=lat_params['abg'][0], valstep=0.001)
		s_b = Slider(ax_b, 'b', valmin=lat_params['abg'][1]*.75, valmax=lat_params['abg'][1]*1.25, valinit=lat_params['abg'][1], valstep=0.001)
		s_g = Slider(ax_g, 'g', valmin=lat_params['abg'][2]*.75, valmax=lat_params['abg'][2]*1.25, valinit=lat_params['abg'][2], valstep=0.1)
		s_r = Slider(ax_ph, 'phi', valmin=min(lat_params['base'][2]*.5,-5), valmax=max(lat_params['base'][2]*1.5,5), valinit=lat_params['base'][2], valstep=0.01)
		
		avg_par = lat_params['abg'][0]/2 + lat_params['abg'][1]/2
		s_shx = Slider(ax_shx, 'shx', valmin=-avg_par/2, valmax=avg_par/2, valinit=lat_params['base'][0], valstep=0.001)
		s_shy = Slider(ax_shy, 'shy', valmin=-avg_par/2, valmax=avg_par/2, valinit=lat_params['base'][1], valstep=0.001)
		
		
		def update(_=None):
			lat_params['abg'][0] = s_a.val
			lat_params['abg'][1] = s_b.val
			lat_params['abg'][2] = s_g.val
			lat_params['base'][2] = s_r.val
			lat_params['base'][0] = s_shx.val
			lat_params['base'][1] = s_shy.val
			
			ij_cr, th_relevant, observed_xy, obs_cr, lookup_df = preprocess_dataset(lat_params,motif,dataset,calib,recall_zero=recall_zero,
				extra_shift_ab=extra_shift_ab,max_dist=max_dist,sub_area=sub_area)
			sc.set_offsets(np.c_[th_relevant])	 # update the scatter in-place
			
			#update set of zeros
			param_vec_zeros,_ = vectorize_params(lat_params,m_zeros)
			zeros,_,_ = get_coords_from_ij(ij_cr,param_vec_zeros,max_lim,crop=False)
			sc0.set_offsets(np.c_[zeros])
			fig.canvas.draw_idle()
			
		s_a.on_changed(update)
		s_b.on_changed(update)
		s_g.on_changed(update)
		s_r.on_changed(update)
		s_shx.on_changed(update)
		s_shy.on_changed(update)
		
		ax_reset = fig.add_axes([0.87, 0.25, 0.08, 0.08])
		btn = Button(ax_reset, 'Reset')
		btn.on_clicked(lambda evt: (s_a.reset(), s_b.reset(), s_r.reset(), s_shx.reset(), s_shy.reset(), s_g.reset()))
		
		plt.show()
		print('Params',lat_params)


	ij_cr, th_relevant, observed_xy, obs_cr, lookup_df = preprocess_dataset(lat_params,motif,dataset,calib,recall_zero=recall_zero,
									extra_shift_ab=extra_shift_ab,max_dist=max_dist,sub_area=sub_area)

	lookup_t = lookup_df['lookup_t'].values

	
	metadata = {}
	metadata['refined'] = do_fit
	metadata['relative'] = relative_to
	metadata['atoms_used'] = len(obs_cr)



	#metadata['std'] = std


	param_vec,fit_param_vec = vectorize_params(lat_params,motif) 
	if do_fit:
		res = scipy.optimize.minimize(get_diff,param_vec, args=(param_vec,fit_param_vec,ij_cr,obs_cr,lookup_t,max_lim))
		#print(res)
		print('Residual',res.fun)
				
		metadata['residual_in_pm'] = np.sqrt(res.fun)*1000
		param_vec = res.x
		#metadata['param'] = res.x
		#metadata['a/b'] = res.x[0]/res.x[1]
		#err = get_errors(res)

	metadata['param'] = param_vec
	
	
	# (i,j) hashed with its index in the cleared lookup
	#ij_to_index = {tuple(coord): idx for idx, coord in enumerate(ij_inuse_cleared)}
	
	#if show_initial_spots:
	#	plt.scatter(ij_inuse_cleared[:,0],ij_inuse_cleared[:,1])
	#	plt.show()
	#atomap sublattices
	#obs_lat = am.Sublattice(observed_xy/calib, image=s.T, color='b')
	obs_lat = am.Sublattice(obs_cr/calib, image=s, color='b')
	#theor_lat = am.Sublattice(theor/calib, image=s, color='r') #before refinement, full
	theor_rel_lat = am.Sublattice(th_relevant/calib, image=s, color='r') #before refinement, filtered to paired ones
	

	theor_res,_,_ = get_coords_from_ij(ij_cr,param_vec.copy(),max_lim,crop=False)
	theor_res_rel = [ theor_res[int(i)] if not np.isnan(i) else (np.nan,np.nan) for i in lookup_t ]
	lookup_df[['x_theor_new','y_theor_new']] = np.array(theor_res_rel, dtype='float')
	#theor_res_rel = theor_res_rel[~np.isnan(theor_res_rel)]
	#print(lookup_df)
	
	#refined_lat = am.Sublattice(np.array(theor_res)/calib, image=s, color='r') #refined full
	refined_rel_lat = am.Sublattice(np.array(theor_res_rel)/calib, image=s, color='r') #refined filtered to paired ones


	lat_params_fin,_ = unpack_vector(param_vec,lat_params,motif)
	phi = lat_params_fin['base'][2]
	
		
	#std,vdist,vdiff_xy,vdiff_ref,vdiff_xy_corr,ang,ang_corr = vector_map_calc(phi,obs_cr,theor_res_rel)
	std,diff_df= vector_map_calc(phi,lookup_df)

	#if not relative_to is None:
		
	metadata['std'] = std
	


	
	labels_raw = [i for i in motif.keys() if motif[i]['use'] ]
	types_raw = [ motif[i]['atom'] for i in labels_raw]


	diff_df = diff_df.dropna()
	if not relative_to is None:
		if not relative_to in labels_raw:
			raise IOError('Suggested reference atom position not found')
		else:
			if len(labels_raw) == 1:
				raise IOError('Only one lattice found, can not compute a relative diff')
			#labels_i = [i for i,_ in enumerate(labels_raw) ]
			#relative_to_i = labels_raw.index(relative_to)
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
			#plot_lattice(s,[obs_lat,theor_lat],fname,folder,sf,'initial_guess_full_'+sf)
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
		
		if 'I0' in diff_df.columns and relative_to is None:
			at_labels = [i+'\n'+j for i,j in zip(labels_raw,types_raw)]
			plot_violin(file_s + '_I0_vor',at_labels,diff_df)

		plot_output_page(fname,folder + sf + '/')
		plot_output_page_diff(fname,folder + sf + '/')
	
	if do_fit:
		return metadata, res.x
	else:
		return metadata, param_vec
	
	
