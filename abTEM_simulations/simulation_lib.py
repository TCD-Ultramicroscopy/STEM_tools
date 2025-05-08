#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import ase
import numpy as np
import diffpy.structure
from scipy.spatial.transform import Rotation as R

def get_params(cif_path):
	'''
	Reads cif file from the path given
	cif_path - str, existing path to the valid cif file
	Output - tuple of lattice params, (a,b,s,alpha,beta,gamma) as in cif
	'''
	#folder + cif_files[s]
	c = ase.io.read(cif_path)
	par = c.cell.lengths()
	ang = c.cell.angles()

	return par[0],par[1],par[2],ang[0],ang[1],ang[2]

def get_supercell(cif_path,sblock_size):
	'''
	Here we are expanding the unit cell to fit as close to the requested sblock size as we can in ints
	Then the superblock shifted by xyz to bring its center to 0
	
	Inputs:
	cif_path - str, existing path to the valid cif file
	sblock_size - int or float, size of the superblock cube
	
	Output - ase object
	'''

	c = ase.io.read(cif_path)	
	#par = ase.geometry.cell_to_cellpar(c.cell, radians=False)[:3]
	par = get_params(cif_path)[:3]
	multiplier = [ int(sblock_size/i) if int(sblock_size/i) >=1 else 1 for i in par  ]
	c = c*multiplier
	
	#Let's bring center to 1/2 of the volume
	
	#c.translate(-np.array(multiplier,dtype=int)/2*par)#rounded up
	c.translate(-ase.geometry.cell_to_cellpar(c.cell, radians=False)[:3]/2)#precise center

	return c

def drop_negatives(hkl):
	'''
	Filters out Friedels pairs from the set of directions.
	Preference is given to positive values (ones with h+k+l > -h-k-l ) 
	hkl - array of vectors [(h_i),(k_i),(l_i),...]
	'''
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


def hkl_to_uvw(param_list,hkl,max_uvw,around=True):
	'''
	Converts hkl to uvw with respect to the lattice parameters

	around - boolean, defines if we need uvw as ints
			Extremely important parameter!
				if True, structure is aligned by the nearest uvw
				if False, by the normal to hkl
	param_list - (a,b,c,alpha,beta,gamma)
	hkl - tuple of 3 ints is expected
	max_uvw - upper threshold for the equivalent uvw multiplier
	
	out - tuple of 3 ints if uvw, of 3 floats if normal to hkl
	'''
	print('Given HKL ',hkl)

	#First, create lattice and its reciprocal version
	lat = diffpy.structure.Lattice(param_list[0],param_list[1],
							  param_list[2],param_list[3],
							  param_list[4],param_list[5])
	lat_r = lat.reciprocal()
	G = lat.metrics
	Gr = lat_r.metrics
	
	#Convert hkl vector to the real space 
	vs = lat_r.cartesian(hkl)
	out = lat.fractional(vs)

	#print(out)
	#out = [i if i == 0 else 1/i for i in out]
	out = np.array(out)
	
	#Renorm, with respect to zeros
	u = out[abs(out) > 0.0001]
	out = out/min(abs(u))
	#print(out)
	
	#Try to find a multiplier, within a given margins
	m = find_multiplier(out,max_uvw)  
	out = out*m

	#Round up if needed
	if around:
		out = out.round()
		out = out.astype(int)
	out = out.tolist()
	print('Proposed UVW ',out)

	return(out)

def find_multiplier(frac,max_uvw):
	'''
	Ugly way to find the best multiplier with respect to the upper threshold
	frac - uvw vector
	max_uvw - int, threshold
	'''
	diff = []
	multipliers = np.arange(1,max_uvw)
	m = 1
	fl = True
	for i in multipliers:
		res = frac*i - np.round(frac*i)
		diff.append(sum(abs(res)))
		#if there is a way to get ints, there is no point to search further
		if np.all(abs(res) < 0.0001):
			print('Ideal multiplier ',i)
			m = i
			break
		#if somehow reasonable multiplier found, we'd better keep it,
		#but continue with attempts to find an ideal one
		if np.all(abs(res) < 0.1) and fl:
			print('Non-ideal multiplier ',i)
			m = i
			fl = False


	#f = multipliers[diff == min(diff)]
	#!TODO - check if multiplier is not found at all

	return m





#Here the rotation magic happens
def get_euler_uvw(param_list,uvw):
	'''
	This function finds a rotation matrix required to align a given [uvw] with Z
	There were a plenty of issues while I was trying to directly align these vectors,
		so here it is done step by step, one rotation after another
	
	param_list - (a,b,c,alpha,beta,gamma)
	uvw - vector
	
	returns rotation object as in scipy.spatial.transform
	'''
	lat = diffpy.structure.Lattice(param_list[0],param_list[1],
							  param_list[2],param_list[3],
							  param_list[4],param_list[5])
							  
	#Fractional coordinates of the real-space uvw and c vectors
	vv = lat.cartesian(uvw)
	vc = lat.cartesian([0,0,1])
	print('Check uvw',uvw)
	
	#Fractional coordinates of the real-space a,b,c vectors
	av,bv,cv = lat.cartesian([1,0,0]),lat.cartesian([0,1,0]),lat.cartesian([0,0,1])

	#Another way to get angles between a,b,c and x,y,z
	print('Sanity check')
	sal,sbt,sgm = np.linalg.norm(np.cross(av,[1,0,0])),np.linalg.norm(np.cross(bv,[0,1,0])),np.linalg.norm(np.cross(cv,[0,0,1]))
	print('Angles to axes',sal,sbt,sgm)
	
	#First rotation - bring a to OX by rotation around Z
	AtoX = R.from_matrix(np.eye(3))
	if sal != 0:
		AtoX = R.from_euler('z',-param_list[5]+90,degrees=True)
		print('Around z by',-param_list[5]+90)
	print(vv,vc)
	#vv = vv/np.linalg.norm(vv)
	#vc = vc/np.linalg.norm(vc)
	
	#Check
	an_c = lat.angle(uvw,[0,0,1])
	print('Angle uvw to c',an_c)
	
	
	#an_c_p = np.arccos(np.dot(vv,vc)/abs(np.linalg.norm(vc))/abs(np.linalg.norm(vv)))/np.pi*180
	#print('Angle uvw to c, direct',an_c_p)
	
	#Warning for trigonal systems
	if param_list[3]-90 !=0:
		print('Careful! might be an issue there; this angle was not tested properly')
	
	#Second rotation, only for trigonal - around OX (and a), to bring c to XZ plane 
	CtoZ_bc = R.from_euler('x',param_list[3]-90,degrees=True) #angle between c and z in bc plane
	print('Around x by',param_list[3]-90)
	
	#Third rotation, around OY, to bring c to Z within XZ plane
	CtoZ_XZ = R.from_euler('y',param_list[4]-90,degrees=True) #angle between c and z in XZ plane
	print('Around y by',param_list[4]-90)
	
	#Combine 2nd and 3rd
	CtoZ = CtoZ_bc*CtoZ_XZ
	
	#Fourth rotation - align uvw and c
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
	
	#Cumulative rotation
	rot = rot*AtoX
	
	print(np.round(rot.as_matrix(),2))
	return rot


def make_lamella(cif_path,hkl,sblock_size,lamella_sizes,atom_to_zero,tol,max_uvw,is_uvw=True,inplane_angle=None,extra_shift_z=0 ):
	'''
	High-level function; for a given crystal structure, generates the rectangular set of atoms - 'lamella'
		in such a way that the requested uvw is directed upwards (!TODO or downwards... to be confirmed)
	Input:
		cif_path - str, existing path to the valid cif file
		hkl - tuple of three ints; desired orientation uvw (or normal to hkl)
		sblock_size - int or float, size of the superblock cube which later will be rotated and cropped
		lamella_sizes - tuple of 3 ints, XxYxZ sizes of the proposed lamella in Angstroms
		atom_to_zero - str, label of atom to be set to the point of origin after the rotation completed
			!NB not to the corner of the virtual scan; there is a gap
		tol - float, tolerance for atoms on surfaces and near zero
		max_uvw - int, max value of the multiplier for hkl to uvw conversion
		
		is_uvw - boolean, defines if we provided hkl or uvw vector
		
		inplane_angle - float, extra rotation in XY plane, degrees
		extra_shift_z - float, shifts the superblock along Z before cropping
	output - ase object
	'''
	
	#Obtain rotation matrix for hkl/uvw and the structure given
	param_list = get_params(cif_path)
	if is_uvw:
		uvw = hkl
	else:
		uvw = hkl_to_uvw(param_list,hkl,max_uvw,around=False)
	rot_matrix = get_euler_uvw(param_list,uvw)
	
	#Create supercell
	sup = get_supercell(cif_path,sblock_size)
	all_atoms = sup.get_positions()
	print('There are ',len(all_atoms),' atoms in the supercell')
	
	#Here we are rotating x,y,z set
	new_coords = rot_matrix.apply(all_atoms)# inverse=True
	#amending coordinates in the ase object
	sup.set_positions(new_coords)
	#and harmonizing variables, just in case
	new_coords = sup.get_positions()

	#lets select a relatively small test subset of atoms to:
	#	- find the atom of interest nearest to (0,0,0) - say, atom0
	#	- find the angle between OX and vector from the atom0 to the nearest atom of the same type
	box = max(param_list[:3])
	box = max(box,10)
	box = np.ones(3)*box
	mask = np.all(new_coords >= -box, axis=1) * np.all(
						new_coords < box, axis=1 )

	#mask = np.all( abs(new_coords) < np.array(param_list[:3]), axis=1 )
	test_c = sup[mask]
	
	#Here I wish to find an atom of interest nearby 0 and bring it to 0... on the subset of +-abc
	if atom_to_zero is not None:
		leads = [ i for i,j in zip(test_c.get_positions(), test_c.get_chemical_symbols()) if j == atom_to_zero ]
		#print('Pb',leads)
		dist = ase.geometry.get_distances((0,0,0), p2=leads, cell=test_c.cell, pbc=True )[1][0]
		new_zero = [ i for i,j in zip(leads,dist) if (j > min(dist) - tol) and ( j < min(dist) + tol ) ][0]
		print('Zero moved to',new_zero)
		#!NB ase object is shifted here, not (x,y,z) set - it shall be updated
		sup.translate(-new_zero)
		#Extra shift by z applied here
		sup.translate((0,0,-extra_shift_z))

	#Update gathered coordinates
	new_coords = sup.get_positions()
	
	
	if inplane_angle is None and atom_to_zero is not None:
		###Here we need to rotate the system in the way that the nearest atom of interest will settle on OX
		#if dist are within marg, then 1st quarter
		loc_margin = [0.1,0.1,0.1]
		#take a flat rectagonal area of margin thickness with a margin void around zero 
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
		#angle can not be autodefined without a given atom of interest
		if inplane_angle is None:
			inplane_angle = 0
		rot_angle = inplane_angle
		print('Requested in-plane rotation',rot_angle)
		
	#Here we are rotating the full set of coordinates (x,y,z)
	rot_matrix = R.from_euler('z',rot_angle,degrees=True)
	new_coords = rot_matrix.apply(new_coords)
	print(np.round(rot_matrix.as_matrix(),5))
	#applying changes to the ase object
	sup.set_positions(new_coords)
	#and renewing xyz, just in case
	new_coords = sup.get_positions()
	
	###Here we are cropping the lamella, from 0 to lims
	#-a/2,a/2 to be considered
	margin = np.ones(3)*tol
	mask = np.all(new_coords >= -margin, axis=1) * np.all(
			new_coords < (np.array(lamella_sizes) + margin), axis=1 )

	cropped = sup[mask]
	print('Atoms in the lamella',len(cropped.get_positions()))

	#Just in case, to avoid undesired ase magic
	cropped.set_cell((lamella_sizes[0],lamella_sizes[1],lamella_sizes[2],90,90,90))
	
	return cropped

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

'''
def uvw_to_hkl(param_list,uvw,max_uvw):
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
	m = find_multiplier(out,max_uvw)  
	out = out*m

	out = out.round()
	out = out.astype(int)

	out = out.tolist()
	print('HKL ',out)

	return(out)
'''

