#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import numpy as np
import hyperspy.api as hs
import copy
import pandas as pd
import csv
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def rotate_vec(v,an):
	'''
	Rotate 2-vector in plane
	inputs:
		v - list or nparray, 2-vector
		an - float, rotation angle in degrees
	outputs:
		(x,y) - tuple, 2-vector
	'''

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
	'''
	Reshape the vector of variables to dicts using the structure of dicts provided
	inputs:
		param_vec - list, vector of parameters
		lat_params - dict, dict of the lattice params as provided in presets
		motif - dict, dict of the atomic motif params as provided in presets
	outputs:
		lat_params - dict, renewed dict of the lattice params
		motif2 - dict, renewed dict of the atomic motif params		
	'''	
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
	'''
	Reshape dicts of variables to a vector, if corresponding keys are set in dicts
	inputs:
		lat_params - dict, dict of the lattice params
		motif - dict, dict of the atomic motif params
	outputs:
		param_vec - list, vector of relevant parameters
		fit_vec - list, vector of boolean values to identify parameters as variables for refinement		
	'''	
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

def load_frame(folder,fname,calib_size_by_px): #TODO - we do not really have to have a tiff
	'''
	Loads a tiff file provided as a hyperspy object
	inputs:
		folder - str, path to the workfolder
		fname - str, basename of the tif file
	output:
		s - hyperspy 2Dsignal with pixels enforced to be square 
	'''
	s = hs.load(folder+fname+'.tif')#TODO shall we check if '.tiff' is there?
	metadata = {}
	#'''
	metadata['fname'] = fname#TODO should we return this mdata?

	imsize_px = (s.axes_manager[0].size,s.axes_manager[1].size)
	#xy directions not checked! has to be verified
	#d0,d1 = imsize[0]/imsize_px[0],imsize[1]/imsize_px[1]
	#print(d0,d1)
	
	d0 = calib_size_by_px#calib_size/calib_px
	metadata['nm_per_pix'] = d0

	#TODO Flaw!!! atomap apparently does not support non-sqare pixels!
	s.axes_manager[0].scale = d0
	s.axes_manager[1].scale = d0
	s.axes_manager[0].units = 'nm'
	s.axes_manager[1].units = 'nm'
	
	return s
	
def export_data(folder,sf,fname,lat_params_vec,raw_lat_params,raw_motif,metadata):
	'''
	Save variables as csv
	
	'''

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


def vector_map_calc(phi,df):
	#phi = param[2]/180*np.pi
	#fin_lat = get_coords_from_ij(f_ij,param,no_modulation,only_ortho,max_lim)[0]
	
	obs = np.array(df[['x_obs','y_obs']].values)
	calc = np.array(df[['x_theor_new','y_theor_new']].values)
	
	vdiff_xy = obs - calc
	df['vdiff_xy'] = vdiff_xy.tolist()
	 
	vdiff_ref = np.nanmean(vdiff_xy,axis=0)
	
	vproj = np.array([(x*np.cos(phi) + y*np.sin(phi), y*np.cos(phi) - x*np.sin(phi)) for x,y in vdiff_xy])
	df['vproj'] = vproj.tolist()
	
	#print(np.std(vproj,axis=0))
	#print(np.std(vdiff_xy,axis=0))
	#print('ref',vdiff_ref)
	#print('Std rot',np.sqrt(np.sum(np.std(vproj,axis=0)**2)),'len',len(vproj))
	#print('Std raw',np.sqrt(np.sum(np.std(vdiff_xy,axis=0)**2)),'len',len(vdiff_xy))
	
	#std_to_report = np.std(vproj,axis=0)
	
	
	vdist = np.sqrt(np.sum(vdiff_xy**2,axis=1))
	df['vdist'] = vdist
	
	vdiff_xy_corr = vdiff_xy - vdiff_ref
	df['vdiff_xy_corr'] = vdiff_xy_corr.tolist()
	print('Test dist',sum(vdist)/len(vdist))
	#print(np.mean(vdiff_xy_corr,axis=0))
	
	
	
	#print(np.std(abs(vdiff_xy_corr),axis=0))
	ang = [np.arctan2(j,i) for i,j in vdiff_xy]
	ang_corr = [np.arctan2(j,i) for i,j in vdiff_xy_corr] #np.angle(vdiff_xy, deg=True)
	df['ang'] = ang
	df['ang_corr'] = ang_corr
	
	std_to_report = np.std(abs(vdiff_xy),axis=0)
	#vdist = np.sqrt(np.sum(vdiff_xy**2,axis=1))
	#str_mean = plot_stats_rep(vdist,fname_save)
	
	return std_to_report,df




def load_and_trim_cv2(path, white_threshold=245):
	"""
	Load an image with OpenCV and trim (near-)white borders.
	Keeps all non-white pixels intact.
	Returns an array in RGB/RGBA for matplotlib.
	"""
	path = str(Path(path))
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if img is None:
		raise FileNotFoundError(f"Could not read image: {path}")

	# Grayscale
	if img.ndim == 2:
		nonwhite = img < white_threshold
		if not np.any(nonwhite):
			return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		ys, xs = np.where(nonwhite)
		crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
		return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

	# Color with alpha (BGRA)
	if img.shape[2] == 4:
		b, g, r, a = cv2.split(img)
		# A pixel is considered "content" if it’s visible (a>0) and not pure white
		nonwhite = (a > 0) & ((r < white_threshold) | (g < white_threshold) | (b < white_threshold))
		if not np.any(nonwhite):
			return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
		ys, xs = np.where(nonwhite)
		crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1, :]
		return cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA)

	# Color (BGR)
	b, g, r = cv2.split(img)
	nonwhite = (r < white_threshold) | (g < white_threshold) | (b < white_threshold)
	if not np.any(nonwhite):
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
	ys, xs = np.where(nonwhite)
	crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1, :]
	return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def imshow_no_axes(ax, im):
	ax.imshow(im)
	ax.set_axis_off()

def plot_output_page(fname,folder):
	s = '_'+ folder.split('/')[-2]
	pngs = {
		"a": folder+'../'+fname+".png",
		"b": folder+fname+s + '_vmap_rotated.png',
		"c": folder+fname+s + '_diff_hist.png',
		"d": folder+fname+s + '_angles_hist.png',
		"e": folder+fname+s + '_vmap_rotated_fr0.png',
	}
	
	#print(pngs)
	# Load & trim
	imgs = {k: load_and_trim_cv2(Path(v)) for k, v in pngs.items()}

	# -------- figure layout --------
	# 2 rows, 3 columns; first row col3 is text panel
	# First row slightly taller to make those images "larger"
	fig = plt.figure(figsize=(12, 7))

	gs_parent = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.15)

	# Row 1 (top): 1x3 with [1, 1, 0.3] widths
	gs_top = gs_parent[0].subgridspec(nrows=1, ncols=3, width_ratios=[1, 1, 0.3], wspace=0.05)
	ax_a   = fig.add_subplot(gs_top[0, 0])
	ax_b   = fig.add_subplot(gs_top[0, 1])
	ax_txt = fig.add_subplot(gs_top[0, 2])

	# Row 2 (bottom): 1x3 with equal widths
	gs_bot = gs_parent[1].subgridspec(nrows=1, ncols=3, width_ratios=[1, 1, 1], wspace=0.05)
	ax_c   = fig.add_subplot(gs_bot[0, 0])
	ax_d   = fig.add_subplot(gs_bot[0, 1])
	ax_e   = fig.add_subplot(gs_bot[0, 2])


	# Show images
	imshow_no_axes(ax_a, imgs["a"])
	imshow_no_axes(ax_b, imgs["b"])
	imshow_no_axes(ax_c, imgs["c"])
	imshow_no_axes(ax_d, imgs["d"])
	imshow_no_axes(ax_e, imgs["e"])


	#load text vals
	
	df = pd.read_csv(folder+fname+s +'_metadata.csv', sep="\t", index_col=0)
	print(df)

	try:
		av_dist = df.loc['residual_in_pm','0']
		correct_dist = True
	except:
		av_dist = df.loc['std'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')).to_numpy()[0]
		av_dist = np.sqrt(sum(av_dist**2))*1000
		correct_dist = False
		
	at_num = df.loc['atoms_used','0']
	lat_par = df.loc['param'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')).to_numpy()[0]
	lat_a = np.round(lat_par[0]*10,2)
	lat_b = np.round(lat_par[1]*10,2)
	lat_g = np.round(lat_par[2],1)


	txt_label = "N = " + str(at_num) +"\n"
	txt_label += "a = " + str(lat_a) +"$\AA$ \n"
	txt_label += "b = " + str(lat_b) +"$\AA$ \n"
	txt_label += "$\gamma $ = " + str(lat_g) +"$^{\circ}$ \n"
	# Text area (right side of first row)
	ax_txt.set_axis_off()
	ax_txt.text(
		0.0, 0.80, txt_label,
		transform=ax_txt.transAxes,
		va="top", ha="left",
		fontsize=12
	)

	
	# Overlay text on one image (here, on 'e')
	if correct_dist:
		ttt = "$| \delta | = $"+str(np.round(float(av_dist),1))+'pm'
	else:
		ttt = "$| \Delta d | = $"+str(np.round(float(av_dist),1))+'pm'
	ax_e.text(
		0.6, 0.9, ttt,
		transform=ax_e.transAxes,
		va="top", ha="left",
		fontsize=11#, bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2")
	)

	# Optional: tight layout and save
	plt.tight_layout()
	plt.savefig(folder+"_panel_1.png", dpi=400, bbox_inches="tight")
	#plt.show()

def plot_output_page_diff(fname,folder):
	s = '_'+ folder.split('/')[-2]
	files = [folder+fname+s + '_vmap_rotated.png',
		folder+fname+s + '_vmap_proj_a.png',
		folder+fname+s + '_vmap_proj_a90.png']
	titles = ["Vector map", "Components $\parallel ~ a$", "Components $\perp ~ a$"]

	images = [load_and_trim_cv2(f) for f in files]

	# ---------- Plot in 1×3 grid ----------
	fig, axes = plt.subplots(1, 3, figsize=(12, 4))

	for ax, img, title in zip(axes, images, titles):
	    ax.imshow(img)
	    ax.set_title(title, fontsize=13, pad=10)
	    ax.axis("off")

	plt.tight_layout()
	plt.savefig(folder+"_panel_2.png", dpi=400, bbox_inches="tight")
	#plt.show()
'''
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
'''
