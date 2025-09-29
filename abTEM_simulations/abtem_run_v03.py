#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"


#zarr<3 is needed!
import numpy as np
import ase, abtem
import matplotlib.pyplot as plt

#import datetime
from pathlib import Path
import tomli_w

#Our simulations routines
from simulation_lib import *
#Our local config read&validate routines
import confread

#config is expected to be in the same folder as a code
#'config.toml' is in use if None is provided
cfg = confread.load_config('config.toml')



#######
####### Loading params

folder_sim = cfg.paths.folder_sim + cfg.paths.extr #outputs
#extr = cfg.paths.extr
#folder_sim += extr
folder = cfg.paths.folder #for CIF files

###Configuring computational environment

#Here we are deciding if cpu or gpu computing happens
use_gpu = cfg.gpu_related.use_gpu
if use_gpu:
	abtem.config.set({"device": "gpu", "fft": "fftw",'dask.lazy': True})
	abtem.config.set({"cupy.fft-cache-size" : cfg.gpu_related.cupy_fft_cache_size})
	abtem.config.set({"dask.chunk-size-gpu" : cfg.gpu_related.dask_chunk_size_gpu})
	import cupy as cp
else:
	abtem.config.set({"device": "cpu", "fft": "fftw",'dask.lazy': True})	

abtem.config.set({"dask.chunk-size" : cfg.gpu_related.dask_chunk_size})

if use_gpu and cfg.gpu_related.dask_cuda:
	from dask_cuda import LocalCUDACluster
	from dask.distributed import Client
	client = Client("tcp://127.0.0.1:8786")

	from rmm.allocators.cupy import rmm_cupy_allocator


	cp.cuda.set_allocator(rmm_cupy_allocator)
elif cfg.gpu_related.dask_cuda:
	print('dask_cuda can run only if CUDA is allowed; skipping')

#No of threads; limited by video-memory, can fail if No is too high
abtem.config.set(scheduler="processes",num_workers=1)

#####



####################
#Main variables
####################

SEED = 15


#Microscope settings
do_full_run = cfg.simulations.do_full_run 
HT_value = cfg.microscope.HT_value
do_diffraction = cfg.microscope.do_diffraction

#########
#detectors settings

haadf_detector = abtem.AnnularDetector(inner=cfg.microscope.haadfinner, outer=cfg.microscope.haadfouter)
abf_detector = abtem.AnnularDetector(inner=cfg.microscope.abfinner, outer=cfg.microscope.abfouter)
bf_detector = abtem.AnnularDetector(inner=cfg.microscope.bfinner, outer=cfg.microscope.bfouter)


#lamella settings
sample_name = cfg.paths.sample_name

max_uvw = cfg.lamella_settings.max_uvw
sblock_size = cfg.lamella_settings.sblock_size

scan_s = cfg.lamella_settings.scan_s
borders = cfg.lamella_settings.borders
thickness = cfg.lamella_settings.thickness
extra_shift_z = cfg.lamella_settings.extra_shift_z

tol = cfg.lamella_settings.tol

scan_start = (borders*2,borders*2)
scan_stop = (borders*2+scan_s,borders*2+scan_s)
lamella_sizes = (borders*2+scan_s,borders*2+scan_s,thickness)

atom_to_zero = cfg.lamella_settings.atom_to_zero

#Lamella tilt
global_tilt = (cfg.lamella_settings.global_tilt_a,
		cfg.lamella_settings.global_tilt_a)

add_vacancies_toggle = cfg.lamella_settings.add_vacancies_toggle
element_to_remove = cfg.lamella_settings.element_to_remove
probability_of_vac = cfg.lamella_settings.probability_of_vac



#Number of frozen phonons
frozen_phonons = cfg.simulations.frozen_phonons
if frozen_phonons == 'None':
	frozen_phonons = None

fph_sigma = cfg.simulations.fph_sigma
if fph_sigma is bool:
	fph_sigma = None

override_sampling = cfg.simulations.override_sampling

#Short names for cif files
cif_files = {
		'Pbam':'Pbam.cif',
		'Pm3m':'Pm3m.cif',
		'I4cm':'I4cm.cif',
		'Ima2':'Ima2.cif',
		'CC':'CC.cif',
		'R-3c':'R-3c.cif'
}

def save_config(cfg, path: str ):
    path = Path(path)
    with path.open("wb") as f:           # note: binary mode
        tomli_w.dump(cfg.model_dump(), f)

def add_potential(surf):
	# Make the potential
	potential = abtem.Potential(
	surf,
	sampling=0.05,   # real space sampling
	projection='infinite',
	parametrization='kirkland',
	periodic=False,
	).build()#.compute()

	return potential

def add_frozen_phonons_potential(surf):
	frozen = abtem.FrozenPhonons(surf, num_configs=frozen_phonons, sigmas=fph_sigma, seed=100)
	# Make the potential
	potential = abtem.Potential(
	frozen,
	sampling=0.05,   # real space sampling
	projection='infinite',
	parametrization='kirkland',
	periodic=False,
	)#.build().compute() #once-done is faster, but requires a huge amount of memory
	
	return potential

def add_probe(potential):
	# Make the probe
	tilt = (0,0)#(global_tilt[0]/180*np.pi*1000,global_tilt[1]/180*np.pi*1000)
	probe = abtem.Probe(
	energy=HT_value, 
	semiangle_cutoff=30, 
	defocus="scherzer"
	)
	#,
	#tilt=tilt
	# Match probe grid to potential grid
	probe.grid.match(potential)
	print('tilt',global_tilt)
	return probe

def add_scan(probe,pot):
	# Define scan area
	print('Proposed sampling',probe.ctf.nyquist_sampling * .9)
	if not override_sampling:
		sampling = probe.ctf.nyquist_sampling * .9
	else:
		sampling = override_sampling
		print('Overrided sampling',sampling)

	scan = abtem.scan.GridScan(
		start = scan_start, 
		end = scan_stop, 
		sampling = sampling,
		potential=pot
	)
	
	return scan   

def plot_diffraction(pot,fname,ftitle):
	#try:
	#	pot_cpu = pot.copy_to_device('cpu')
	#except:
	#pot_cpu = pot.build()
	#pot_cpu = pot.copy_to_device('cpu')
	#pot_cpu = pot.to_cpu()
	#print('No CPU conversion')
	#print(type(pot_cpu))
	
	initial_waves = abtem.PlaneWave(energy=HT_value,device='cpu')
	exit_waves = initial_waves.multislice(pot).compute()
	#exit_waves_raw = initial_waves.multislice(pot).to_cpu()

	#exit_waves = exit_waves_raw.compute()
	print('Exit waves')
	diffraction_patterns = exit_waves.diffraction_patterns(max_angle="valid", block_direct=True).compute()
	diffraction_patterns.show(
		explode=False,power=0.2,units="mrad",
		figsize=(10, 6),cbar=True,common_color_scale=True,)
		
	fig = plt.gcf()

	#fig.tight_layout(rect=[0, 0, 1, 0.9])#
	fig.suptitle(ftitle, y=1.005)
	plt.savefig(fname,dpi=600)
	plt.close()
	
	if len(diffraction_patterns.shape) > 2:
		diffraction_patterns.mean(axis=0).to_tiff(fname[:-4]+'.tif')
	else:
		diffraction_patterns.to_tiff(fname[:-4]+'.tif')
	
	


def prepare_job(hkl_set,is_uvw=True,inplane_angle=None):
	'''
	This function prepares a set of ase objects to use for the further computations
	Inputs:
		hkl_set - list (Nx3), list of hkl (or uvw) vectors
		is_uvw - boolean, defines are vectors provided as hkl to use a normal to the corresponding plane, or as uvw
		inplane_angle - float | None, inplane slab rotation in degrees (if defined) prior crop
	Output:
		full_dataset - dict, structured in a local format
	'''
	full_dataset = {}
	count = 0
	for i in hkl_set.keys():
		for j in hkl_set[i]:
			print('Generating',i,j)
			#scell = struct_set[i]
			cif_path = folder + cif_files[i]
			surf = make_lamella(cif_path,j,sblock_size,lamella_sizes,atom_to_zero,tol,max_uvw,
						is_uvw=is_uvw,inplane_angle=inplane_angle,
						extra_shift_z=extra_shift_z,vac_xy=borders,vac_z=borders)
						
			if add_vacancies_toggle:
				surf = add_vacancies(surf,element_to_remove,probability_of_vac)
				print('Vacancies applied to '+element_to_remove+ ', probability '+str(probability_of_vac))
			
			potential = add_potential(surf)
			#potential.compute()
			fph_potential = add_frozen_phonons_potential(surf)
			#fph_potential.compute()
			probe = add_probe(potential)
			fph_probe = add_probe(fph_potential)
			data = {
				'symm':i,
				'hkl':j,
				'surface':surf,
				'potential':potential,
				'probe':probe,
				'fph_potential':fph_potential,
				'fph_probe':fph_probe,
				'scan':add_scan(probe,potential)
			}
			full_dataset[count] = data
			count+=1
	return full_dataset   



def simulation_run(s,is_uvw=True,inplane_angle=None,do_full_run=False):
	'''
	Main starter function
	Inputs:
		s - list (Nx3), list of hkl (or uvw) vectors
		is_uvw - boolean, defines are vectors provided as hkl to use a normal to the corresponding plane, or as uvw
		inplane_angle - float | None, inplane slab rotation in degrees (if defined) prior crop
		do_full_run - boolean
	'''

	#cp.cuda.Stream.null.synchronize()
	dataset = prepare_job(s,is_uvw,inplane_angle)
	for i in dataset.keys():
		cfg_out_path = folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'
		cfg_out_path += ''.join([str(q) for q in dataset[i]['hkl']]) +'.cfg'
		save_config(cfg, cfg_out_path )
	
		sg = dataset[i]['symm']
		line_hkl = ''.join([str(q) for q in dataset[i]['hkl']])

		plot_dataset(dataset[i],is_uvw,scan_s,borders,folder_sim,sample_name)
		
		surf_fname = folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'+line_hkl+'surf.xyz'
		ase.io.write(surf_fname, dataset[i]['surface'], 'xyz')

		
		if do_diffraction:
			print('Diffraction - single')
			potential = dataset[i]['potential']
			ttl = sg+', [' + line_hkl +'], '+ str(lamella_sizes[0])+'x'+str(lamella_sizes[1])+'x'+str(lamella_sizes[2])+'$\AA$'
			plot_diffraction(potential,folder_sim+sg+'_'+line_hkl+'_single_diff.png',ttl)
			del potential
			print('Diffraction - fph')
			fph_potential = dataset[i]['fph_potential']
			plot_diffraction(fph_potential,folder_sim+sg+'_'+line_hkl+'_fph_diff.png',ttl+', '+str(frozen_phonons)+' fph')
			del fph_potential
			
						
		if do_full_run:
			potential = dataset[i]['potential']
			probe = add_probe(potential)
			probe.grid.match(potential)		
			scan = add_scan(probe,potential)
			
			#Single img
			#'''
			measurements = probe.scan(potential, scan=scan, detectors=[haadf_detector,abf_detector,bf_detector])
			img = measurements.compute()#(scheduler="threads", num_workers=num_workers)
			
			w = 0
			while w<len(img):
				iimg = img[w].copy()
				det_s = ['haadf','abf','bf'][w]
				iimg.to_tiff(folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'.tif')
				iimg.to_zarr(folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'.zarr',overwrite=True)
				
				for k in [0.025,0.1,0.25]:
					blurred = iimg.gaussian_filter(k,boundary='constant')
					blurred.to_tiff(folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'+line_hkl+'_'+det_s+'_'+str(k).replace('.','-')+'.tif')
				w+=1
			#'''

			#frozen phonon set
			fph_potential = dataset[i]['fph_potential']
			probe.grid.match(potential)
			scan = add_scan(probe,potential)

			fph_measurements = probe.scan(fph_potential, scan=scan, detectors=[haadf_detector,abf_detector])
			img = fph_measurements.compute()#(scheduler="threads", num_workers=num_workers)

			w = 0
			while w<w<len(img):
				iimg = img[w].copy()
				det_s = ['haadf','abf','bf'][w]
				iimg.to_tiff(folder_sim+'fph_'+dataset[i]['symm']+'_'+str(global_tilt)+'_'+line_hkl+'_'+det_s+'.tif')
				iimg.to_zarr(folder_sim+'fph_'+dataset[i]['symm']+'_'+str(global_tilt)+'_'+line_hkl+'_'+ det_s+'.zarr',overwrite=True)
				
				for k in [0.025,0.1,0.25]:
					blurred = iimg.gaussian_filter(k,boundary='constant')
					blurred.to_tiff(folder_sim+'fph_'+dataset[i]['symm']+'_'+str(global_tilt)+'_'+line_hkl +'_'+det_s+'_'+str(k).replace('.','-')+'.tif')
				w+=1

	del dataset
	#gc.collect()
	#cp.get_default_memory_pool().free_all_blocks()
	#cp.get_default_pinned_memory_pool().free_all_blocks()

simulation_run({'CC': [[1,1,2]]},is_uvw=True,inplane_angle=None,do_full_run=do_full_run)

print('Finished')
