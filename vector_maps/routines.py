#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import numpy as np
import hyperspy.api as hs
import copy
import pandas as pd

def rotate_vec(v,an):
	an = an/180*np.pi
	c = np.cos(an)
	s = np.sin(an)
	vx,vy = v
	#print(vx,vy)
	x = vx*c-vy*s
	y = vy*c+vx*s
	#print(x,y)
	return x,y

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

def load_frame(folder,fname,calib_size_by_px):
	s = hs.load(folder+fname+'.tif')
	metadata = {}
	#'''
	metadata['fname'] = fname

	imsize_px = (s.axes_manager[0].size,s.axes_manager[1].size)
	#xy directions not checked! has to be verified
	#d0,d1 = imsize[0]/imsize_px[0],imsize[1]/imsize_px[1]
	#print(d0,d1)
	
	d0 = calib_size_by_px#calib_size/calib_px
	metadata['nm_per_pix'] = d0

	#Flaw!!! atomap apparently does not support non-sqare pixels!
	s.axes_manager[0].scale = d0
	s.axes_manager[1].scale = d0
	s.axes_manager[0].units = 'nm'
	s.axes_manager[1].units = 'nm'
	
	return s
	
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
