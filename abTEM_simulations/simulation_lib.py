#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as np
from scipy.spatial.transform import Rotation as R

from ase.visualize import view
from ase.cell import Cell
import abtem.atoms as ab_atoms
import diffpy.structure

import ase, abtem, py4DSTEM
import numpy as np
import matplotlib.pyplot as plt

import tifffile, scipy, numpy.linalg
import datetime

def get_params(cif_path):
	#folder + cif_files[s]
	c = ase.io.read(cif_path)
	par = c.cell.lengths()
	ang = c.cell.angles()

	return par[0],par[1],par[2],ang[0],ang[1],ang[2]

def get_supercell(cif_path,sblock_size):
	#folder + cif_files[s] 
	c = ase.io.read(cif_path)
	#here we are increasing the cell to fit as close to the size as we can in ints
	par = ase.geometry.cell_to_cellpar(c.cell, radians=False)[:3]
	multiplier = [ int(sblock_size/i) if int(sblock_size/i) >=1 else 1 for i in par  ]
	c = c*multiplier
	
	#Let's bring center to 1/3 of the volume
	c.translate(-np.array(multiplier,dtype=int)/2*par)

	return c

'''
def gen_hkl(uvw,max_uvw=20):
	print('UVW ',uvw)
	all_hkl = gen_list(max_uvw)
	sel_hkl = np.array([i for i in all_hkl if np.dot(i,uvw)==0])
	simplest = []
	for i in sel_hkl:
		u,c = np.unique(i, return_counts=True)
		cc = dict(zip(u,c))
		if 0 in u and cc[0] == 2:
			simplest.append(i)
	#print('S',simplest)
	m = np.sum(sel_hkl**2,axis=1)
	smallest = sel_hkl[m == min(m)]
	smallest = drop_negatives(smallest.tolist())
	#print(smallest)
	i = 0
	while len(smallest)<2:
		#print('Error - not enough hkl found for uvw given')
		thr = int(min(m)+i)
		#print(type(min(m)),type(thr))
		#print(m == thr)
		#print(len(m),len(sel_hkl))
		sm2 = sel_hkl[m <= thr]
		smallest = drop_negatives(sm2.tolist())
		i+=1
	sel_hkl = drop_negatives(sel_hkl.tolist())
	return smallest
'''

def drop_negatives(hkl):
	new_set = []
	for i in hkl:
		h,k,l = i
		mh,mk,ml = -h,-k,-l
		#print(new_set)
		if [mh,mk,ml] in hkl and not [mh,mk,ml] in new_set and not [h,k,l] in new_set:
			if sum([mh,mk,ml])<=sum(i):
				new_set.append(i)
			else:
				new_set.append([mh,mk,ml])
	return new_set

def uvw_to_hkl(param_list,uvw):
	print('UVW ',uvw)
	#param_list = get_params(s)
	lat = diffpy.structure.Lattice(param_list[0],param_list[1],
							  param_list[2],param_list[3],
							  param_list[4],param_list[5])
	lat_r = lat.reciprocal()
	G = lat.metrics
	Gr = lat_r.metrics

	v = lat.cartesian(uvw)
	out = lat_r.fractional(v)

	print(out)
	#out = [i if i == 0 else 1/i for i in out]
	out = np.array(out)
	
	u = out[abs(out) > 0.0001]
	out = out/min(abs(u))
	print(out)
	m = find_multiplier(out)  
	out = out*m

	out = out.round()
	out = out.astype(int)

	out = out.tolist()
	print('HKL ',out)

	return(out)

def hkl_to_uvw(param_list,hkl,around=True):
	print('HKL ',hkl)
	#param_list = get_params(s)
	lat = diffpy.structure.Lattice(param_list[0],param_list[1],
							  param_list[2],param_list[3],
							  param_list[4],param_list[5])
	lat_r = lat.reciprocal()
	G = lat.metrics
	Gr = lat_r.metrics

	vs = lat_r.cartesian(hkl)
	out = lat.fractional(vs)

	print(out)
	#out = [i if i == 0 else 1/i for i in out]
	out = np.array(out)
	
	u = out[abs(out) > 0.0001]
	out = out/min(abs(u))
	print(out)
	m = find_multiplier(out)  
	out = out*m

	if around:
		out = out.round()
		out = out.astype(int)
	out = out.tolist()
	print('UVW ',out)

	return(out)


	 
def find_multiplier(frac):
	#res,inte = np.modf(frac)
	diff = []
	multipliers = np.arange(1,max_uvw)
	m = 1
	fl = True
	for i in multipliers:
		res = frac*i - np.round(frac*i)
		diff.append(sum(abs(res)))
		#print(i,res)
		if np.all(abs(res) < 0.0001):
			print('Ideal multiplier ',i)
			m = i
			break
		if np.all(abs(res) < 0.1) and fl:
			print('Non-ideal multiplier ',i)
			m = i
			fl = False

	#f = multipliers[diff == min(diff)]

	return m

#Here the rotation magic happens
def get_euler_uvw(param_list,uvw):

	lat = diffpy.structure.Lattice(param_list[0],param_list[1],
							  param_list[2],param_list[3],
							  param_list[4],param_list[5])
	vv = lat.cartesian(uvw)
	vc = lat.cartesian([0,0,1])
	print('Check uvw',uvw)
	
	av,bv,cv = lat.cartesian([1,0,0]),lat.cartesian([0,1,0]),lat.cartesian([0,0,1])

	print('Sanity check')
	sal,sbt,sgm = np.linalg.norm(np.cross(av,[1,0,0])),np.linalg.norm(np.cross(bv,[0,1,0])),np.linalg.norm(np.cross(cv,[0,0,1]))
	print('Angles to axes',sal,sbt,sgm)
	AtoX = R.from_matrix(np.eye(3))
	if sal != 0:
		AtoX = R.from_euler('z',-param_list[5]+90,degrees=True)
		print('Around z by',-param_list[5]+90)
	print(vv,vc)
	#vv = vv/np.linalg.norm(vv)
	#vc = vc/np.linalg.norm(vc)
	
	an_c = lat.angle(uvw,[0,0,1])
	print('Angle uvw to c',an_c)

	#an_c_p = np.arccos(np.dot(vv,vc)/abs(np.linalg.norm(vc))/abs(np.linalg.norm(vv)))/np.pi*180
	#print('Angle uvw to c, direct',an_c_p)
	if param_list[3]-90 !=0:
		print('Careful! might be an issue there; this angle was not tested properly')
	CtoZ_bc = R.from_euler('x',param_list[3]-90,degrees=True) #angle between c and z in bc plane
	print('Around x by',param_list[3]-90)
	CtoZ_XZ = R.from_euler('y',param_list[4]-90,degrees=True) #angle between c and z in XZ plane
	print('Around y by',param_list[4]-90)
	CtoZ = CtoZ_bc*CtoZ_XZ
	rot_v = np.cross(vv,vc)
	if np.linalg.norm(rot_v) != 0:
		rot_v = rot_v/np.linalg.norm(rot_v)
		rot_v = rot_v*an_c
		print('Around axis',rot_v,'by',np.linalg.norm(rot_v))
		VtoC = R.from_rotvec(rot_v,degrees=True)
		#test = R.from_euler('y',an_c,degrees=True)
		rot = VtoC*CtoZ
	else:
		#special case of collinear v and c. we just need to find a sign then
		flip = R.from_matrix(np.eye(3)*(np.dot(vc,vv)/abs(np.dot(vc,vv))))
		rot = flip*CtoZ
	#'''
	rot = rot*AtoX
	
	print(np.round(rot.as_matrix(),2))
	return rot


def make_lamella(cif_path,hkl,sblock_size,lamella_sizes,atom_to_zero,tol,is_uvw=True,inplane_angle=None ):
	sup = get_supercell(cif_path,sblock_size)
	param_list = get_params(cif_path)
	if is_uvw:
		uvw = hkl
	else:
		uvw = hkl_to_uvw(param_list,hkl,around=False)
	rot_matrix = get_euler_uvw(param_list,uvw)
	#rot_matrix = get_euler_hkl(param_list,hkl)
	#print(rot_matrix.as_matrix())
	all_atoms = sup.get_positions()
	#sup.set_cell((lamella_sizes[0],lamella_sizes[1],lamella_sizes[2],90,90,90))
	print(len(all_atoms))
	box = max(param_list[:3])
	box = max(box,10)
	box = np.ones(3)*box
	new_coords = rot_matrix.apply(all_atoms)# inverse=True
	#new_coords = all_atoms
	
	#to add here - rotation around z
	
	#Here I wish to find a lead atom nearby 0 and bring it to 0... on the subset of +-abc
	#print(par)
	sup.set_positions(new_coords)
	#sup.center()
	#print('Looking for a zero shift')
	new_coords = sup.get_positions()
	mask = np.all(new_coords >= -box, axis=1) * np.all(
										 new_coords < box, axis=1 )

	#mask = np.all( abs(new_coords) < np.array(param_list[:3]), axis=1 )
	test_c = sup[mask]
	
	leads = [ i for i,j in zip(test_c.get_positions(), test_c.get_chemical_symbols()) if j == atom_to_zero ]
	#print('Pb',leads)
	dist = ase.geometry.get_distances((0,0,0), p2=leads, cell=test_c.cell, pbc=True )[1][0]
	new_zero = [ i for i,j in zip(leads,dist) if (j > min(dist) - tol) and ( j < min(dist) + tol ) ][0]
	print('Zero moved to',new_zero)
	sup.translate(-new_zero)

	new_coords = sup.get_positions()
	
	
	#'''
	if inplane_angle is None:
		###Here we need to rotate the system in the way that the nearest lead atom will settle on OX
		#if dist are within marg, then 1st quarter
		#
		loc_margin = [0.1,0.1,0.1]
		#take a flat rectagonal area of margin thickness without a margin around zero 
		mask = np.all(abs(new_coords) >= loc_margin, axis=1) * np.all(
												abs(new_coords) < box, axis=1 )
	
		test_c = sup[mask]
		leads = [ i for i,j in zip(test_c.get_positions(), test_c.get_chemical_symbols()) if j == atom_to_zero ]

		proj_XY = [ (x,y,0) for (x,y,z) in leads ]
		if len(proj_XY) >1:
			dot = [ np.dot(i,[1,0,0]) for i in proj_XY ]
			norm = [ np.linalg.norm(i) for i in proj_XY ]
			angle = [ np.arccos(i/j)/np.pi*180 for i,j in zip(dot,norm) ]

			selected = np.atleast_2d(proj_XY[np.all(abs(np.array(angle))==min(abs(np.array(angle))))])
			#fin_selected = selected
			if len(selected)>1:
				dist = [np.dot(i,i) for i in selected]
				selected = np.atleast_2d(selected[np.all(dist==min(dist))])
				#print(selected)
				if len(selected)>1:
						for i in selected:
							found = False
							if i[0]>0 and i[1]>0:
								fin_selected = i
								found = True
								break
							elif i[1]>0:
								fin_selected = i
								found = True
						if not found:
							fin_selected = selected[0]
				else:
							fin_selected = selected[0]
			else:
						fin_selected = selected[0]
		else:
			fin_selected = proj_XY[0]
		print(fin_selected)
	
		rot_angle = np.arccos(np.dot(fin_selected,[1,0,0])/np.linalg.norm(fin_selected))/np.pi*180
		print('Proposed in-plane rotation',rot_angle)
	else:
		rot_angle = inplane_angle
		print('Requested in-plane rotation',rot_angle)
		
		
	rot_matrix = R.from_euler('z',rot_angle,degrees=True)
	new_coords = rot_matrix.apply(new_coords)
	print(np.round(rot_matrix.as_matrix(),5))
	#'''
	sup.set_positions(new_coords)


	
	new_coords = sup.get_positions()
	###Here we are cropping the lamella
	margin = np.ones(3)*tol
	mask = np.all(new_coords >= -margin, axis=1) * np.all(
			new_coords < (np.array(lamella_sizes) + margin), axis=1 )

	cropped = sup[mask]
	print('Atoms in the lamella',len(cropped.get_positions()))
	#atoms_before = cropped.get_positions()
	cropped.set_cell((lamella_sizes[0],lamella_sizes[1],lamella_sizes[2],90,90,90))
	
	return cropped
