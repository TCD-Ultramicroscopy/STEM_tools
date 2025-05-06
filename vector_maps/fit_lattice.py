#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import numpy as np
import hyperspy.api as hs
import atomap.api as am
import atomap.initial_position_finding as ipf
from scipy.spatial.transform import Rotation as R
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as st

#from scipy.stats import rayleigh
#from matplotlib.colors import DivergingNorm


test_vector_map=False

max_lim=(100,100)

#Size of a virtual set of points; has to be large for large imgs, but more demanding
#ij_max = 300
ij_max = 150

#Under active development, contaings a lot of repeats
#treating all supported lattices (centered, non-centered, dumbbells, ...) in a very similar way but separately 
#has to be rewritten and generalized
def data_processing(folder,fname,calib_px,calib_size,ptonn=0.75,centering=True,initial_guess=[0.41656176,0.41371804,-0.75,0,0,0,0,89.9],
			lat2_is_ref=True,vec_scale=0.05,max_d=0.25,rerun_fit=True,rerun_detect=True,precision_thr=1,basis_hor_angle=0,dumbbells=False,just_centered=False):
	#create metadata here
	#max_lim=(1000,1000)
	metadata={}
	s,size = load_frame(folder,fname,calib_px,calib_size)
	s_sm = s.map(scipy.ndimage.gaussian_filter, sigma=1)
	#load manual x,y
	x0,y0,ell0,rot0,i_0 = np.load(folder+fname.split('.')[0]+'.npy')

	observed_xy = refine_obs_lattice(s, x0,y0, folder,fname, ptonn=ptonn, metadata=metadata,rerun_detect=rerun_detect)

	#Find the observed one, closest to zero
	orig_dist = np.sqrt(np.sum(np.array(observed_xy)**2,axis=1))
	min_orig = np.array(observed_xy)[orig_dist<=min(orig_dist)+.00001][0]
	
	initial_guess[3] = min_orig[0]
	initial_guess[4] = min_orig[1]
	
	ij_range = np.arange(-ij_max,ij_max)
	
	#Basis_vectors
	x_base = (1,0)
	y_base = (0,-1)
	#basis angle counts CW!
	x_base_r = rotate_vec(x_base,basis_hor_angle)
	y_base_r = rotate_vec(y_base,basis_hor_angle)
	
	
	one_lat = not just_centered
	two_lat = not just_centered
	diff_lat = not just_centered
	#print(not just_centered)	
	if diff_lat:
		two_lat = True
	
	if two_lat:
		one_lat = True
		
	if test_vector_map:
		
		orig = [(0,0),(0,1),(1,0),(1,1)]
		v1 = [(1,0) for i in orig]
		v2 = [(0,-1) for i in orig]
		v_diag = [(1,-1) for i in orig]
		
		v1_r = [rotate_vec(i,basis_hor_angle) for i in v1]
		v2_r = [rotate_vec(i,basis_hor_angle) for i in v2]
		plot_quiver(folder+'test_v1',orig,v1,[0 for i in v1],10)
		plot_quiver(folder+'test_v2',orig,v2,[0 for i in v2],10)
		plot_quiver(folder+'test_diag',orig,v_diag,[0 for i in v_diag],10)
		
		plot_quiver(folder+'test_v1r',orig,v1_r,[0 for i in v1],10)
		plot_quiver(folder+'test_v2r',orig,v2_r,[0 for i in v2],10)
	
	if just_centered:
		print('Just centered')
		#lat,ij = gen_centered_ortho_lattice(ij_range,initial_guess,(80,80))
		ij0 = gen_ij(ij_range,centering=True,dumbbells=dumbbells)
		
		no_modulation,only_ortho = True,True
		lat,ij = get_coords_from_ij(ij0,initial_guess,False,False,max_lim,crop=True)
		#print('pass1')
		
		#start from a distorted one
		
		f_obs_g,f_theor,f_ij_g,extra = filter_lat(observed_xy,initial_guess,ij,False,False,max_lim,max_d=max_d,extra_data=[ell0,rot0,i_0]) #,no_modulation=False,only_ortho=False
		plot_quiver(folder+fname+'_ellipticity',f_obs_g,extra[0],extra[1],1,units_v='rel.u.',ell=True)
		#mask = np.all(np.array(f_theor)>=(-1,-1),axis=1)*np.all(np.array(f_theor)<=(1,1),axis=1)
		#print(np.array(f_obs_g)[mask],np.array(f_theor)[mask],np.array(f_ij_g)[mask])

		new_xy = [[i/size,j/size] for i,j in f_theor ]
		metadata['atoms_used'] = len(new_xy)
		
		test_lattice = am.Sublattice(new_xy, image=s, color='b')
		obs_lattice = am.Sublattice(np.array(observed_xy)/size, image=s, color='r')
		
		plot_lattice(s,[obs_lattice,test_lattice],fname,folder,'initial_guess_full')
		plot_lattice(s,[obs_lattice,am.Sublattice([[i/size,j/size] for i,j in lat ], image=s, color='b')],fname,folder,'initial_guess_raw_full')
		compare = {}
		
		if rerun_fit:
			fin_par0,compare['free'],fin_lat,vxy,vref,vxy_c,ang,ang_c,fname_save = refinement_run(folder,fname,metadata,initial_guess,size,s,obs_lattice,
				f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=False,only_ortho=False,extra_str='_cen')
			lat,ij = get_coords_from_ij(ij0,fin_par0,no_modulation,only_ortho,max_lim,crop=True)
			#print('pass2')
			f_obs_g,f_theor,f_ij_g,extra = filter_lat(observed_xy,fin_par0,ij,no_modulation,only_ortho,max_lim,
					max_d=max_d,extra_data=[ell0,rot0,i_0])
		else:
			fin_par0 = initial_guess
		
		fin_par,compare['rigid'],fin_lat,vxy,vref,vxy_c,ang,ang_c,fname_save = refinement_run(folder,fname,metadata,fin_par0,size,s,obs_lattice,
						f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=True,only_ortho=True,extra_str='_cen')
		#plot_quiver(fname_save,fin_lat,vxy_c,ang_c,vec_scale)
		plot_quiver(fname_save,fin_lat,vxy,ang,vec_scale)
		
		fin_par,compare['non-ortho'],fin_lat,vxy,vref,vxy_c,ang,ang_c,fname_save = refinement_run(folder,fname,metadata,fin_par0,size,s,obs_lattice,
						f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=True,only_ortho=False,extra_str='_cen')
		#plot_quiver(fname_save,fin_lat,vxy_c,ang_c,vec_scale)
		plot_quiver(fname_save,fin_lat,vxy,ang,vec_scale)

		fin_par,compare['free'],fin_lat,vxy,vref,vxy_c,ang,ang_c,fname_save = refinement_run(folder,fname,metadata,fin_par0,size,s,obs_lattice,
					f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=False,only_ortho=False,extra_str='_cen')
		#plot_quiver(fname_save,fin_lat,vxy_c,ang_c,vec_scale)
		plot_quiver(fname_save,fin_lat,vxy,ang,vec_scale)

		fin_par,compare['modulated_ortho'],fin_lat,vxy,vref,vxy_c,ang,ang_c,fname_save = refinement_run(folder,fname,metadata,fin_par0,size,s,obs_lattice,
						f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=False,only_ortho=True,extra_str='_cen')
		#plot_quiver(fname_save,fin_lat,vxy_c,ang_c,vec_scale)
		plot_quiver(fname_save,fin_lat,vxy,ang,vec_scale)
		
		data = pd.DataFrame.from_dict(compare,orient='index')
		data.to_csv(folder+fname.split('.')[0]+'_params_centered.csv',sep='\t')
		

		
	if one_lat:
		print('One lat')
		#lat,ij = gen_centered_ortho_lattice(ij_range,initial_guess,(80,80))
		ij0 = gen_ij(ij_range,centering=False)
		
		no_modulation,only_ortho = True,False
		lat,ij = get_coords_from_ij(ij0,initial_guess,no_modulation,only_ortho,max_lim,crop=True)
		#print('pass1')
		f_obs_g,f_theor,f_ij_g,extra = filter_lat(observed_xy,initial_guess,ij,no_modulation,only_ortho,
				max_lim,max_d=max_d,extra_data=[ell0,rot0,i_0])
		plot_quiver(folder+fname+'ellipticity_1',f_obs_g,extra[0],extra[1],1,units_v='rel.u.',ell=True)
		
		I0_lat1 = np.mean(extra[2])
		#mask = np.all(np.array(f_theor)>=(-1,-1),axis=1)*np.all(np.array(f_theor)<=(1,1),axis=1)
		#print(np.array(f_obs_g)[mask],np.array(f_theor)[mask],np.array(f_ij_g)[mask])

		new_xy = [[i/size,j/size] for i,j in f_theor ]
		metadata['atoms_used'] = len(new_xy)
		
		test_lattice = am.Sublattice(new_xy, image=s, color='b')
		obs_lattice = am.Sublattice(np.array(observed_xy)/size, image=s, color='r')
		
		plot_lattice(s,[obs_lattice,test_lattice],fname,folder,'initial_guess1')
		
		compare = {}
		if rerun_fit:
			fin_par0,compare['free'],fin_lat,vxy,vref,vxy_c,ang,ang_c,fname_save = refinement_run(folder,fname,metadata,initial_guess,size,s,obs_lattice,
						f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=False,only_ortho=False,extra_str='_l1')
			lat,ij = get_coords_from_ij(ij0,fin_par0,no_modulation,only_ortho,max_lim,crop=True)
			#print('pass2')
			f_obs_g,f_theor,f_ij_g,extra2 = filter_lat(observed_xy,fin_par0,ij,no_modulation,only_ortho,
					max_lim,max_d=max_d,extra_data=[ell0,rot0,i_0])
		else:
			fin_par0 = initial_guess
					
		fin_par,compare['rigid'],fin_lat,vxy,vref,vxy_c,ang,ang_c,fname_save = refinement_run(folder,fname,metadata,fin_par0,size,s,obs_lattice,
					f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=True,only_ortho=True,extra_str='_l1')
		#plot_quiver(fname_save,fin_lat,vxy_c,ang_c,vec_scale)
		plot_quiver(fname_save+'1',fin_lat,vxy,ang,vec_scale)
		
		fin_par,compare['non-ortho'],fin_lato,vxyo,vrefo,vxy_co,ang,ang_c,fname_save = refinement_run(folder,fname,metadata,fin_par0,size,s,obs_lattice,
					f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=True,only_ortho=False,extra_str='_l1')
		#plot_quiver(fname_save,fin_lato,vxy_co,ang_c,vec_scale)
		plot_quiver(fname_save+'1',fin_lato,vxyo,ang,vec_scale)

		
		data = pd.DataFrame.from_dict(compare,orient='index')
		data.to_csv(folder+fname.split('.')[0]+'_params_lat1.csv',sep='\t')
		
		#Here we are using nonortho one for projections
		d_proj_x = [np.dot(i,x_base) for i in vxyo]
		d_proj_y = [np.dot(i,y_base) for i in vxyo]
		
		proj_x = [np.array(x_base)*i for i in d_proj_x]
		proj_y = [np.array(y_base)*i for i in d_proj_y]
		plot_quiver(fname_save+'1'+'_one_proj_x',fin_lato,proj_x,ang,vec_scale)
		plot_quiver(fname_save+'1'+'_one_proj_y',fin_lato,proj_y,ang,vec_scale)
		
		#and for rotated axes
		d_proj_xr = [np.dot(i,x_base_r) for i in vxyo]
		d_proj_yr = [np.dot(i,y_base_r) for i in vxyo]
		
		proj_xr = [np.array(x_base_r)*i for i in d_proj_xr]
		proj_yr = [np.array(y_base_r)*i for i in d_proj_yr]
		plot_quiver(fname_save+'1'+'_one_proj_xr',fin_lato,proj_xr,ang,vec_scale)
		plot_quiver(fname_save+'1'+'_one_proj_yr',fin_lato,proj_yr,ang,vec_scale)
		
	if two_lat:
		print('Two lat')		
		ij0_2 = ij0 + (.5,.5)
		
		no_modulation,only_ortho = True,False
		lat2,ij2 = get_coords_from_ij(ij0_2,initial_guess,no_modulation,only_ortho,max_lim,crop=True)
		#print('pass1')
		f_obs_g2,f_theor2,f_ij_g2,extra2 = filter_lat(observed_xy,initial_guess,ij2,no_modulation,only_ortho,
				max_lim,max_d=max_d,extra_data=[ell0,rot0,i_0])
		plot_quiver(folder+fname+'ellipticity_2',f_obs_g2,extra2[0],extra2[1],1,units_v='rel.u.',ell=True)
		#mask = np.all(np.array(f_theor)>=(-1,-1),axis=1)*np.all(np.array(f_theor)<=(1,1),axis=1)
		#print(np.array(f_obs_g)[mask],np.array(f_theor)[mask],np.array(f_ij_g)[mask])
		I0_lat2 = np.mean(extra2[2])
		new_xy2 = [[i/size,j/size] for i,j in f_theor2 ]
		metadata['atoms_used'] = len(new_xy2)
		
		test_lattice2 = am.Sublattice(new_xy2, image=s, color='b')
		obs_lattice = am.Sublattice(np.array(observed_xy)/size, image=s, color='r')
		
		plot_lattice(s,[obs_lattice,test_lattice2],fname,folder,'initial_guess2')
		
		compare = {}

		if rerun_fit:
			fin_par0,compare['free2'],fin_lat2,vxy2,vref2,vxy_c2,ang2,ang_c2,fname_save = refinement_run(folder,fname,metadata,initial_guess,size,s,obs_lattice,
			f_ij_g2,f_obs_g2,max_lim=max_lim,no_modulation=False,only_ortho=False,extra_str='l2')
			lat2,ij2 = get_coords_from_ij(ij0_2,fin_par0,no_modulation,only_ortho,max_lim,crop=True)
			#print('pass2')
			f_obs_g2,f_theor2,f_ij_g2,extra2 = filter_lat(observed_xy,fin_par0,ij2,no_modulation,only_ortho,
					max_lim,max_d=max_d,extra_data=[ell0,rot0,i_0])
		else:
			fin_par0 = initial_guess
		fin_par,compare['rigid2'],fin_lat2,vxy2,vref2,vxy_c2,ang2,ang_c2,fname_save = refinement_run(folder,fname,metadata,fin_par0,size,s,obs_lattice,
			f_ij_g2,f_obs_g2,max_lim=max_lim,no_modulation=True,only_ortho=True,extra_str='l2')
		#plot_quiver(fname_save+'2',fin_lat2,vxy_c2,ang_c2,vec_scale)
		plot_quiver(fname_save+'2',fin_lat2,vxy2,ang2,vec_scale)
		
		fin_par,compare['non-ortho2'],fin_lat2o,vxy2o,vref2o,vxy_c2o,ang2,ang_c2,fname_save = refinement_run(folder,fname,metadata,fin_par0,size,s,obs_lattice,
				f_ij_g2,f_obs_g2,max_lim=max_lim,no_modulation=True,only_ortho=False,extra_str='l2')
		#plot_quiver(fname_save+'2',fin_lat2o,vxy_c2o,ang_c2,vec_scale)
		plot_quiver(fname_save+'2',fin_lat2o,vxy2o,ang2,vec_scale)

		data = pd.DataFrame.from_dict(compare,orient='index')
		data.to_csv(folder+fname.split('.')[0]+'_params_lat2.csv',sep='\t')
		
		#Here we are using nonortho one for projections
		d_proj_x = [np.dot(i,x_base) for i in vxy2o]
		d_proj_y = [np.dot(i,y_base) for i in vxy2o]
		
		proj_x = [np.array(x_base)*i for i in d_proj_x]
		proj_y = [np.array(y_base)*i for i in d_proj_y]
		plot_quiver(fname_save+'_two_proj_x',fin_lat2o,proj_x,ang2,vec_scale)
		plot_quiver(fname_save+'_two_proj_y',fin_lat2o,proj_y,ang2,vec_scale)
		
		#and for rotated axes
		d_proj_xr = [np.dot(i,x_base_r) for i in vxy2o]
		d_proj_yr = [np.dot(i,y_base_r) for i in vxy2o]
		
		proj_xr = [np.array(x_base_r)*i for i in d_proj_xr]
		proj_yr = [np.array(y_base_r)*i for i in d_proj_yr]
		plot_quiver(fname_save+'_two_proj_xr',fin_lat2o,proj_xr,ang2,vec_scale)
		plot_quiver(fname_save+'_two_proj_yr',fin_lat2o,proj_yr,ang2,vec_scale)
		
		if I0_lat1/I0_lat2 > 1:
			tmp = 'Lat1 - Pb'
		else:
			tmp = 'Lat2 - Pb'
		print('\n'+ 'Intensity lat1/lat2   '+str(I0_lat1/I0_lat2)+'\n'+tmp+'\n'+'\n')

		
	if diff_lat:
		#non-ortho
		#for 178 - #2 is Zr
		compare = {}		
		if lat2_is_ref:
			pb_ij = np.array(f_ij_g).tolist()
			zr_ij = np.array(f_ij_g2).tolist()
			pb_lat = fin_lato
			zr_lat = fin_lat2o
			pb_vx = vxyo
			zr_vx = vxy2o
		else:
			pb_ij = np.array(f_ij_g2).tolist()
			zr_ij = np.array(f_ij_g).tolist()
			pb_lat = fin_lat2o
			zr_lat = fin_lato
			pb_vx = vxy2o
			zr_vx = vxyo
		
		print(pb_ij[0:10], 'Pb',zr_ij[0:10], 'Zr')
		
		diff_lat,diff_vx = [],[]
		
		
		for a in pb_ij:
			a = np.array(a)
			nb = [a+(-0.5,-0.5),
				a+(0.5,-0.5),
				a+(-0.5,0.5),
				a+(0.5,0.5)]
			#print(nb,'nb')
			#print(zr_ij,'zr_ij')
			nb_here = sum([i.tolist() in zr_ij for i in nb])
			if nb_here == 4:
				diff_lat.append(pb_lat[pb_ij.index(a.tolist())])
				nb_vec = [ zr_vx[zr_ij.index(i.tolist())] for i in nb]
				#print(nb_vec,'nb_vec')
				v = np.sum(np.array(nb_vec),axis=0)/4
				#print(v,'v')
				#print(pb_vx[pb_ij.index(a.tolist())],'ref')
				diff_vx.append(pb_vx[pb_ij.index(a.tolist())] - v)

		ang_d = [np.arctan2(j,i) for i,j in diff_vx]
		vdist = np.sqrt(np.sum(np.array(diff_vx)**2,axis=1))
		
		np.save(folder+'angles_diff',np.array(ang_d))
		
		std = np.std(abs(np.array(diff_vx)),axis=0)
		delta = np.sum(vdist)/len(vdist)
		compare['delta'] = delta
		plot_stats_rep(vdist,fname_save+'diff')
		plot_stats_rep(ang_d,fname_save+'ang',ang=True)
		plot_stats_rep(ang_d,fname_save+'ang_weighted',ang=True,ang_weights=vdist)
		compare['N'] = len(diff_vx)
		compare['std'] = std
		print(std)
		plot_quiver(fname_save+'diff',diff_lat,diff_vx,ang_d,.075,hd_w=1.5)
		thr = precision_thr/1000.
		plot_quiver(fname_save+'_diff_filt_'+str(precision_thr),
				np.array(diff_lat)[vdist>=thr],np.array(diff_vx)[vdist>=thr],np.array(ang_d)[vdist>=thr],.1)
		data = pd.DataFrame.from_dict(compare,orient='index')
		data.to_csv(folder+fname.split('.')[0]+'_diff.csv',sep='\t')


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

def load_frame(folder,fname,calib_px,calib_size):
	s = hs.load(folder+fname+'.tif')
	metadata = {}
	#'''
	metadata['fname'] = fname
	metadata['orig_px'] = calib_px
	metadata['orig_size'] = calib_size

	imsize_px = (s.axes_manager[0].size,s.axes_manager[1].size)
	#xy directions not checked! has to be verified
	#d0,d1 = imsize[0]/imsize_px[0],imsize[1]/imsize_px[1]
	#print(d0,d1)
	d0 = calib_size/calib_px
	d1 = d0
	#Flaw!!! atomap apparently does not support non-sqare pixels!
	s.axes_manager[0].scale = d0
	s.axes_manager[1].scale = d1
	s.axes_manager[0].units = 'nm'
	s.axes_manager[1].units = 'nm'
	size = d0
	
	return s, size

def plot_lattice(img,sublattice_list,fname,folder,text):
	plt.close('all')
	
	atom_lattice = am.Atom_Lattice(
			image=img,
			sublattice_list=sublattice_list)
	s = atom_lattice.get_sublattice_atom_list_on_image()
	s.plot()
	fig = s._plot.signal_plot.figure
	fig.delaxes(fig.axes[1]) #remove colorbar
	ax_list = fig.axes
	for i in ax_list:
		i.get_xaxis().set_visible(False)
		i.get_yaxis().set_visible(False)
		i.title.set_text('')
	fig.suptitle(text)
	fig.tight_layout()
	fig.savefig(folder+fname+'_'+text+'.png')
	
	plt.close('all')

def refine_obs_lattice(s, x0,y0, folder,fname, ptonn=0.75, metadata={},rerun_detect=True):
	
	atom_positions = [[i,j] for i,j in zip(x0,y0)]
	sublattice = am.Sublattice(atom_positions, image=s)
	sublattice.find_nearest_neighbors()
	###! NB! to add ellipticity and I here
	if rerun_detect:
		sublattice.refine_atom_positions_using_center_of_mass()
		sublattice.refine_atom_positions_using_2d_gaussian(percent_to_nn=ptonn)
		#extract aniso!
		metadata['percent_to_nn']=ptonn
	else:
		metadata['percent_to_nn']=False
	plt.close('all')
	plot_lattice(s,[sublattice],fname,folder,'atoms_found')

	x = sublattice.x_position
	y = sublattice.y_position
	size = sublattice.pixel_size
	#np.save(folder+fname.split('.')[0],np.array([x,y])) #this one is for GUI
	observed_xy = [ (i*size,j*size) for i,j in zip(x,y)]

	return observed_xy


'''
This code relies on the distorted centered 2D Bravais lattice.
Lattice nodes are encoded as i,j integers
Lattice centering vector is (0.5,0.5)
Dumbbells are encoded as (0.1,0.1) vector
'''

def gen_ij(ij_range,centering=False,dumbbells=False):
	
	ij_set = []
	for i in ij_range:
		for j in ij_range:
			ij_set.append((i,j))
			
	#centering happens here
	if centering:
		ij_c = np.array(ij_set)+[.5,.5]
		full_ij = np.concatenate((ij_set,ij_c),axis=0)
	else:
		full_ij = np.array(ij_set)
	
	if dumbbells:
		ext = full_ij + [0.1,0.1]
		full_ij = np.concatenate((full_ij,ext),axis=0)
	

	return full_ij

def recalc_one_spot(q,a,b,phi,shx,shy,da,db,gamma,ddx,ddy):
	i,j = q
	if abs(int(i)-i) > 0.3 :
		x = np.round(i*2)/2*a+da+(np.round(j*2)/2*b+db)*np.cos(gamma)
		y = (np.round(j*2)/2*b+db)*np.sin(gamma)
	else:
		x = np.round(i*2)/2*a + np.round(j*2)/2*b*np.cos(gamma)
		y = np.round(j*2)/2*b*np.sin(gamma)
	if  abs(np.round(abs(i)*2)-abs(i)*2) > 0.01:
		x+=ddx
		y+=ddy
		
	return [x,y]

def get_coords_from_ij(ij,param,no_modulation,only_ortho,max_lim,crop=False):
	#da,db=0,0
	if len(param) == 8:
		a,b,phi,shx,shy,da,db,gamma = param
		dumbbells = False
	else:
		a,b,phi,shx,shy,da,db,gamma,ddx,ddy = param
		dumbbells = True

	if no_modulation:
		da,db=0,0
	if only_ortho:
		gamma = 90
	if not dumbbells:
		ddx,ddy = 0,0
	
	phi = phi/180.*np.pi
	gamma = gamma/180.*np.pi

	#lat = np.array([ (i*a+da+(j*b+db)*np.cos(gamma),(j*b+db)*np.sin(gamma)) if int(i*2)%2==1 else (i*a + j*b*np.cos(gamma),j*b*np.sin(gamma)) for i,j in ij])

	lat = np.array([ recalc_one_spot((i,j),a,b,phi,shx,shy,da,db,gamma,ddx,ddy) for (i,j) in ij])
	lat = np.array([ (x*np.cos(phi) + y*np.sin(phi), y*np.cos(phi) - x*np.sin(phi)) for x,y in lat])
	lat = lat + (shx,shy)
	
	mask = np.all(lat>=(-5,-5),axis=1)*np.all(lat<=max_lim,axis=1)
	#print(mask)
	#print(len(mask),len(lat),len(ij),lat[0],ij[0])
	if crop:
		cr_lat = lat[mask]
		cr_ij = np.array(ij)[mask]
	else:
		cr_lat = lat
		cr_ij = np.array(ij)

	return cr_lat,cr_ij
	
def filter_lat(obs,param,ij,no_modulation,only_ortho,max_lim,max_d=0,extra_data=[False,False,False]):
	
	el0,rot0,I0 = extra_data
	if max_d == 0:
		max_d = np.sqrt((param[0]/4)**2+(param[1]/4)**2)
	#theor = np.array([ get_coords_from_ij([ij],param)[0].tolist() for ij in nm])
	theor,cr_ij = get_coords_from_ij(ij,param,no_modulation,only_ortho,max_lim)
	f_obs = []
	f_nm = []
	f_theor = []
	
	f_el,f_rot,f_I0 = [],[],[]
	
	i = 0
	while i < len(obs):
		at = obs[i]
		#for at in obs:
		dist = np.array(theor)-np.array(at)
		dist = np.sqrt(dist**2)
		dist = np.sum(dist,axis=1)

		if min(dist)<=max_d:
			#print(min(dist))
			#print()
			f_obs.append(at)
			f_el.append(el0[i])
			f_rot.append(rot0[i])
			f_I0.append(I0[i])
									
			#here we presume that just one minimum is present; might not be true
			#print(np.sum(np.all(dist==min(dist))))
			if len(theor[dist<=min(dist)+.000001])>1:
				print('Err! len>1', theor[dist<=min(dist)+.000001])
			f_theor.append(theor[dist<=min(dist)+.000001][0])
			f_nm.append(cr_ij[dist<=min(dist)+.000001][0])
			#print(at,theor[dist<=min(dist)+.000001][0],nm[dist<=min(dist)+.000001][0])
			#f_theor.append()
		i += 1
	#at the moment, all observed are returned, even if the nearest one is far away... tbf
	#print('Sanity check,lens shall be equal',len(f_obs),len(f_theor))
	f_el = np.array(f_el)
	f_rot = np.array(f_rot)
	#f_el = (f_el * np.cos(f_rot), -f_el * np.sin(f_rot))
	f_el = [(i,j) for i,j in zip(f_el * np.cos(f_rot),-f_el * np.sin(f_rot))]
	f_extra = [f_el,f_rot,f_I0]
	if len(f_obs) != len(f_theor):
		raise IOError()
	return f_obs,f_theor,f_nm,f_extra

def calc_total_diff(f_obs,f_theor):
	diff = np.array(f_obs)-np.array(f_theor)
	diff = np.sum(diff**2,axis=1)
	#sqrt - linear dist
	#diff = np.sqrt(diff)
	#square dist now
	tot_dist = sum(diff)/len(diff)
	return tot_dist

def get_diff(par,f_ij_g,f_obs_g,no_modulation,only_ortho,max_lim):
	#a_prime,c_prime,phi,shx,shy,da,db = par
	#print(len(f_ij_g),'len ij')
	art = get_coords_from_ij(f_ij_g,par,no_modulation,only_ortho,max_lim)[0]
	#art = gen_centered_ortho_lattice(a_prime,b_prime,phi,mn_range,(5,5),shx=shx,shy=shy)
	#filt_obs,filt_theor = filter_lat(observed_xy,art,max_d=np.sqrt((a_prime/4)**2+(b_prime/4)**2))
	diff = calc_total_diff(f_obs_g,art)
	return diff

def vector_map_calc(param,fname_save,f_ij,f_obs,no_modulation,only_ortho,max_lim):
	phi = param[2]/180*np.pi
	
	fin_lat = get_coords_from_ij(f_ij,param,no_modulation,only_ortho,max_lim)[0]
	vdiff_xy = f_obs-fin_lat
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
	print('Test dist',sum(vdist)/len(vdiff_xy_corr))
	print(np.mean(vdiff_xy_corr,axis=0))
	
	#print(np.std(abs(vdiff_xy_corr),axis=0))
	ang = [np.arctan2(j,i) for i,j in vdiff_xy]
	ang_corr = [np.arctan2(j,i) for i,j in vdiff_xy_corr] #np.angle(vdiff_xy, deg=True)
	
	

	std_to_report = np.std(abs(vdiff_xy),axis=0)
	vdist = np.sqrt(np.sum(vdiff_xy**2,axis=1))
	str_mean = plot_stats_rep(vdist,fname_save)

	
	return std_to_report,fin_lat,vdiff_xy,vdiff_ref,vdiff_xy_corr,ang,ang_corr,str_mean


def plot_stats_rep(vdist,fname_save,ang=False,ang_weights=None):
	#stats here

	if ang:
		N_st = 46
		a = np.array(vdist)/np.pi*180
		print(len(a),'angles',a[0])
	else:
		N_st=35
		a = np.array(vdist)*1000
	e_min,e_max = min(a),max(a)
	edges = np.linspace(e_min, e_max, N_st, endpoint=True)
	q=st.lognorm.fit(a,scale=9,loc=3)
	mu=st.lognorm.mean(q[0],loc=q[1],scale=q[2])
	sigma=st.lognorm.std(q[0],loc=q[1],scale=q[2])
	
	
		
	pvalx = st.shapiro(np.log(a))[-1]
	pvalx2 = st.normaltest(a)[-1]
	print("p-value for `accepting` lognormality of x-data = ", pvalx)
	print("p-value for `accepting` normality of x-data = ", pvalx2)
	print("\n!!!!!!!Ok: the array come from lognormal distribution!!!!!!!!!\n" if pvalx>0.01  else "Hm...  the array isn't lognormal")
	print("Ok: the array come from normal distribution" if pvalx2>0.01  else "Hm...  the array isn't normal")
	plt.close()
	
	#if ang:
	#	resultant_length = np.abs(np.mean(np.exp(1j * np.array(vdist))))
	#	p_value = rayleigh.sf(resultant_length * len(vdist))
	#	print(f"Resultant vector length: {resultant_length}")
	#	print(f"P-value: {p_value}")
	
	fig, ax = plt.subplots()
	n,bins,p = ax.hist(a, bins=edges, density=True, stacked=True,
				alpha = 0.1, lw=3, hatch='X', color='b', edgecolor='b',label='Averaged',weights=ang_weights)
	mid=[]
	i=1
	while i<len(edges):
		tmp=edges[i]+edges[i-1]
		tmp=tmp/2
		mid.append(tmp)
		i+=1

	gx=np.linspace(e_min,e_max,num=10000)
	dd=np.round(abs(sigma),1)

	#ddd=par[0]
	ddd=np.round(mu,1)
	if not ang:
		ax.plot(gx,st.lognorm.pdf(gx,q[0],q[1],q[2]))
	
		q=n.tolist()
		ax.text(mid[q.index(max(n))]+max(mid)/3,max(n)*.8,'$d_{mean}$='+str(ddd)+'$\pm$'+str(dd)+' pm', fontsize=16)
		ax.text(mid[q.index(max(n))]+max(mid)/3,max(n)*.95,'N = '+str(len(a)), fontsize=16)

	plt.subplots_adjust(right=0.95, left=0.15, top=0.92, bottom=0.18)
	ax.yaxis.grid(True)
	if not ang:
		ax.set_xlim(0,e_max)
	else:
		ax.set_xlim(e_min,e_max)
	
	ax.xaxis.label.set_size(20)
	ax.yaxis.label.set_size(20)
	ax.yaxis.set_visible(False)

	ax.tick_params(labelsize=16)
	[ax.spines[i].set_visible(False) for i in ["top","left","right"]]
	
	if ang:
		plt.xlabel("Direction, $^{\circ}$")	
	else:
		plt.xlabel("Residual distance, pm")
	plt.ylabel("Occurence")
	
	plt.savefig(fname_save+'_hist.png')
	
	ax.set_xlim(e_min,e_max)
	ax.set_xscale('log')
	plt.savefig(fname_save+'_hist_log.png')
	
	return '$d_{mean}$='+str(ddd)+'$\pm$'+str(dd)+' pm'

	
def plot_quiver(fname_save,fin_lat,vdiff_xy,ang,vec_scale,hd_w=2,units_v='$1 \AA$',ell=False):
	ref_angle = 0#np.pi/4
	vx = [i for i,j in fin_lat]
	vy = [j for i,j in fin_lat]
	#print(vdiff_xy)
	vu = [i for i,j in vdiff_xy]
	vv = [j for i,j in vdiff_xy]

	#vproj_a = [i for i,j in vproj]

	plt.close()
	fig1, ax1 = plt.subplots()
	ax1.set_box_aspect(1)
	ax1.set_title('')
	#ax1.scatter(x, y, color='blue', s=5)
	#color = matplotlib.colors.Normalize(vmin=0, vmax=1)
	#M = np.hypot(u, v)
	ax1.yaxis.set_inverted(True)
	
	ang = - np.array(ang)
	
	ang = (ang - ref_angle + np.pi) % (2*np.pi) - np.pi
	
	if ell:
		norm = mcolors.Normalize(vmin=0, vmax=np.pi)
		Q = ax1.quiver(vx, vy, vu, vv, ang, angles='xy', scale_units='xy', scale=vec_scale,cmap='hsv',
				width=.005,headwidth=1,norm=norm,pivot='middle')
		#ax1.quiver(vx, vy, -np.array(vu), -np.array(vv), ang, angles='xy', scale_units='xy', scale=vec_scale,cmap='hsv',
		#		width=.005,headwidth=hd_w,norm=norm,pivot='middle')
	else:
		norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
		Q = ax1.quiver(vx, vy, vu, vv, ang, angles='xy', scale_units='xy', scale=vec_scale,cmap='hsv',
				width=.005,headwidth=hd_w,norm=norm)#,norm=DivergingNorm(ref_angle))#cmap_bwr?
	qk = ax1.quiverkey(Q, 0.6, 0.92, 0.1, r''+units_v, labelpos='E',
					   coordinates='figure', fontproperties={'size':18})#,fontsize=22
	ax1.set_xlabel('nm',fontsize=18)
	ax1.set_ylabel('nm',fontsize=18)
	ax1.tick_params(axis='both', which='major', labelsize=14)
	cb = fig1.colorbar(Q)
	#cb.set_label("arg", rotation=0, ha="center", va="bottom")
	#cb.ax.yaxis.set_label_coords(0.5, 1.01)
	if not ell:
		cb.set_ticks(np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]))
		cb.set_ticklabels(
			[r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
	)
	else:
		cb.set_ticks(np.array([0,np.pi / 4, np.pi / 2, 3*np.pi / 4, np.pi]))
		cb.set_ticklabels(
			[r"$0$", r"$\dfrac{\pi}{4}$", r"$\dfrac{\pi}{2}$", r"$\dfrac{3\pi}{4}$",r"$\pi$"]
	)
	cb.ax.tick_params(labelsize=14)
	#plt.tight_layout()
	plt.savefig(fname_save,dpi=600)

	plt.close()
	fig1, ax1 = plt.subplots()
	ax1.set_box_aspect(1)
	ax1.set_title('')
	#ax1.scatter(x, y, color='blue', s=5)
	#color = matplotlib.colors.Normalize(vmin=0, vmax=1)
	#M = np.hypot(u, v)
	ax1.yaxis.set_inverted(True)
	ax1.scatter(np.array(vu)*1000,np.array(vv)*1000, s=80, facecolors='none', edgecolors='r')#,color=ang,cmap='hsv'
	ax1.spines['left'].set_position('zero')
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_position('zero')
	ax1.spines['top'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	ax1.spines['left'].set_visible(False)
	ax1.xaxis.set_ticks_position('bottom')
	ax1.yaxis.set_ticks_position('left')

	ax1.set_xlabel('pm', fontsize=16, labelpad=10)
	ax1.set_ylabel('pm', fontsize=16, labelpad=10, rotation=0)#, labelpad=10
	ax1.tick_params(axis='both', which='major', labelsize=14)
	print(max(vu)*1000*.9,max(vv)*1000*.75)
	ax1.yaxis.set_label_coords(.6, .95)
	ax1.xaxis.set_label_coords(.95,.45)
	xl = ax1.get_xlim()
	yl = ax1.get_ylim()
	ll = (min(xl[0],-xl[0],yl[0],-yl[0],xl[1],-xl[1],yl[1],-yl[1]),max(xl[0],-xl[0],yl[0],-yl[0],xl[1],-xl[1],yl[1],-yl[1]))
	ax1.set_xlim(ll[0],ll[1])
	ax1.set_ylim(ll[0],ll[1])
	
	ax1.annotate('', xy=(ax1.get_xlim()[1],0), xytext=(ax1.get_xlim()[0], 0), arrowprops=dict(arrowstyle="->", color='black'))#, xycoords=('axes fraction', 'data')
	ax1.annotate('', xy=(0,ax1.get_ylim()[1]), xytext=(0, ax1.get_ylim()[0]), arrowprops=dict(arrowstyle="->", color='black'))#, xycoords=('data', 'axes fraction')

	#ax1.text(ll[1]*.9,ll[1]*.9,'N = '+str(len(a)), fontsize=16)
	#Q = ax1.quiver([0 for i in vu],[0 for i in vu], vu, vv, ang, angles='xy', scale_units='xy', scale=1,cmap='hsv' )#cmap_bwr?
	#qk = ax1.quiverkey(Q, 0.8, 0.92, 0.1, r'$1 \AA$', labelpos='E',
	#				   coordinates='figure')
	plt.savefig(fname_save+'_fr0.png',dpi=600)
	


#'''
#https://stackoverflow.com/questions/43593592/errors-to-fit-parameters-of-scipy-optimize
def get_errors(res):
	ftol = 2.220446049250313e-09
	tmp_i = np.zeros(len(res.x))
	for i in range(len(res.x)):
		tmp_i[i] = 1.0
		hess_inv_i = res.hess_inv[i][i]
		uncertainty_i = np.sqrt(max(1, abs(res.fun)) * ftol * hess_inv_i)
		tmp_i[i] = 0.0
		print('x^{0} = {1:12.4e} ± {2:.1e}'.format(i, res.x[i], uncertainty_i))


###f,f
def refinement_run(folder,fname,metadata,initial_guess,size,s,sublattice,f_ij_g,f_obs_g,max_lim=max_lim,no_modulation=True,only_ortho=True,extra_str=''):
	metadata['only_ortho'] = only_ortho
	metadata['no_modulation'] = no_modulation
	res = scipy.optimize.minimize(get_diff,initial_guess, args=(f_ij_g,f_obs_g,no_modulation,only_ortho,max_lim))
	metadata['converged'] = res.success
	metadata['residual_x1000'] = np.sqrt(res.fun)*1000
	metadata['param'] = res.x
	metadata['a/b'] = res.x[0]/res.x[1]
	#get_errors2(res)
	print('Residual',res.fun)
	fin_diff = res.fun
	fin_par = res.x
	fin_lat = get_coords_from_ij(f_ij_g,fin_par,no_modulation,only_ortho,max_lim)[0]
	print('Probe dist',get_diff(fin_par,f_ij_g,f_obs_g,no_modulation,only_ortho,max_lim))
	new_xy = [[i/size,j/size] for i,j in fin_lat]
	fit_lattice = am.Sublattice(new_xy, image=s, color='b')
	if not only_ortho:
		extra_str += '_nonortho'
	if not no_modulation:
		extra_str += '_modulated'
	fname_save = folder+fname.split('.')[0]+extra_str
	plot_lattice(s,[sublattice,fit_lattice],fname,folder,'fit'+extra_str)
	std,fin_lat,vdiff_xy,vdiff_ref,vdiff_xy_corr,ang,ang_corr,str_mean = vector_map_calc(fin_par,fname_save,f_ij_g,f_obs_g,no_modulation,only_ortho,max_lim)
	metadata['std'] = std
	data = pd.DataFrame.from_dict(metadata,orient='index')
	data.to_csv(folder+fname.split('.')[0]+extra_str+'.csv',sep='\t')

	print(metadata)
	return (fin_par,[fin_par[0],fin_par[1],fin_par[-1],fin_par[-3],fin_par[-2],std[0],std[1]],fin_lat,vdiff_xy,vdiff_ref,vdiff_xy_corr,ang,ang_corr,fname_save)


###############################################
# Here we are starting the refinement
##############################################

folder = '/home/vasily/test_fit_atomap/'
fname = 'test_frame'#the presence of .npy file is essential; tiff file is expected for previews

#Calibrations. Only ratio is important
calib_px = 1024
calib_size = 16*.9 #nm!


'''
#initial_guess: a,b,phi,shx,shy,da,db,gamma,ddx,ddy
#a,b - Bravais lattice params
#phi,shx,shy - rotation angle and shifts for the whole lattice
#da,db - modulations of the center
#gamma - lattice angle
#ddx,ddy - dumbbells split


#rerun_fit=False is important!
#it seems reasoneable to separate image processing and refiement at the moment

#centering defines the (non)centered Bravais lattice

#lat2_is_ref - defines which of two sublattices (centers of bases) will be a reference one for the differential plot

#vec_scale - vector plot visualisation parameter

#!!!!!!!
#max_d defines a distance threshold for pairing between theoretical and observed lattices
#!!!!!!!

#precision_thr - only for extra cleanup of a differential plot

#basis_hor_angle - defines a direction of the projections axis (if one wish to plot projections on a, but not on X)

#dumbbells - allows ddx,ddy variables to be nonzero

#just_centered - if True, one lattice is considered; if False, two sublattices are created
#'''

data_processing(folder,fname,calib_px,calib_size,centering=True,initial_guess=[0.4108,0.187,-96.96,
													0.15,0.21,
													.00,0,90.1,
													0,0],
			lat2_is_ref=False,vec_scale=0.15,max_d=0.19,rerun_fit=False,ptonn=0.75,rerun_detect=False,
			precision_thr=3,basis_hor_angle=0,dumbbells=False,just_centered=True)
			


print('Done!')
