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

from dataclasses import dataclass
from copy import deepcopy


#Our simulations routines
import simulation_lib as sim
#Our local config read&validate routines
import confread

from itertools import product
from confread import AppConfig  # so we can re-validate expanded dicts

@dataclass
class RunContext:
	cfg: object

	# resolved paths
	folder_sim: str
	folder: str

	# resolved microscope/sim
	do_full_run: bool
	HT_value: float
	do_diffraction: bool
	override_sampling: float | bool

	# resolved lamella geometry
	scan_start: tuple[float, float]
	scan_stop: tuple[float, float]
	lamella_sizes: tuple[float, float, float]
	global_tilt: tuple[float, float]
	tilt_degrees: bool

	# resolved detectors (abtem objects)
	haadf_detector: object
	abf_detector: object
	bf_detector: object

	element_to_remove: str
	probability_of_vac: float
	add_vacancies_toggle: bool
	
	# frozen phonons settings
	frozen_phonons: int | None
	fph_sigma: float | None


#config is expected to be in the same folder as a code
#'config.toml' is in use if None is provided

#######
####### Loading params

#Short names for cif files
cif_files = {
		'Pbam':'Pbam.cif',
		'Pm3m':'Pm3m.cif',
		'I4cm':'I4cm.cif',
		'Ima2':'Ima2.cif',
		'CC':'CC.cif',
		'R-3c':'R-3c.cif'
}

#No of threads; limited by video-memory, can fail if No is too high
abtem.config.set(scheduler="processes",num_workers=1)

SEED = 15

def effective_cfg_for_run(ctx):
	d = deepcopy(ctx.cfg.model_dump())
	d["lamella_settings"]["global_tilt_a"] = float(ctx.global_tilt[0])
	d["lamella_settings"]["global_tilt_b"] = float(ctx.global_tilt[1])
	return d

def _as_list(v):
	return v if isinstance(v, list) else [v]

def _norm_frozen(v):
	# supports scalar or list entries like 'None'
	if v == "None":
		return None
	return int(v)

def _norm_sigma(v):
	if isinstance(v, bool):
		return None
	return float(v)


def expand_cfg(cfg: AppConfig):
	"""
	Yield AppConfig objects, each with scalar values, for the cartesian product of:
	frozen_phonons, fph_sigma, thickness, (global_tilt_a, global_tilt_b), probability_of_vac, HT_value.
	"""
	base = cfg.model_dump()

	frozen_list = [_norm_frozen(x) for x in _as_list(base["simulations"]["frozen_phonons"])]
	sigma_list  = [_norm_sigma(x)  for x in _as_list(base["simulations"]["fph_sigma"])]
	thick_list  = [float(x) for x in _as_list(base["lamella_settings"]["thickness"])]
	pvac_list   = [float(x) for x in _as_list(base["lamella_settings"]["probability_of_vac"])]
	ht_list	 = [int(x)   for x in _as_list(base["microscope"]["HT_value"])]

	ta_list = [float(x) for x in _as_list(base["lamella_settings"]["global_tilt_a"])]
	tb_list = [float(x) for x in _as_list(base["lamella_settings"]["global_tilt_b"])]

	# tilt pairs: full cartesian of a and b
	tilt_pairs = list(product(ta_list, tb_list))

	for frozen, sigma, thick, (ta, tb), pvac, ht in product(
		frozen_list, sigma_list, thick_list, tilt_pairs, pvac_list, ht_list
	):
		d = deepcopy(base)
		d["simulations"]["frozen_phonons"] = ("None" if frozen is None else int(frozen))
		d["simulations"]["fph_sigma"] = (False if sigma is None else float(sigma))  # preserves your bool->None convention
		d["lamella_settings"]["thickness"] = float(thick)
		d["lamella_settings"]["global_tilt_a"] = float(ta)
		d["lamella_settings"]["global_tilt_b"] = float(tb)
		d["lamella_settings"]["probability_of_vac"] = float(pvac)
		d["microscope"]["HT_value"] = int(ht)

		yield AppConfig.model_validate(d)

def resolve_context(cfg, global_tilt: tuple[float, float] | None = None):
	folder_sim = cfg.paths.folder_sim + cfg.paths.extr
	folder = cfg.paths.folder

	do_full_run = cfg.simulations.do_full_run
	HT_value = cfg.microscope.HT_value
	do_diffraction = cfg.microscope.do_diffraction
	override_sampling = cfg.simulations.override_sampling

	borders = cfg.lamella_settings.borders
	scan_s = cfg.lamella_settings.scan_s
	thickness = cfg.lamella_settings.thickness

	scan_start = (borders * 2, borders * 2)
	scan_stop = (borders * 2 + scan_s, borders * 2 + scan_s)
	lamella_sizes = (borders * 2 + scan_s, borders * 2 + scan_s, thickness)

	haadf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.haadfinner, outer=cfg.microscope.haadfouter
	)
	abf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.abfinner, outer=cfg.microscope.abfouter
	)
	bf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.bfinner, outer=cfg.microscope.bfouter
	)

	if global_tilt is None:
		global_tilt = (cfg.lamella_settings.global_tilt_a, cfg.lamella_settings.global_tilt_b)

	element_to_remove = cfg.lamella_settings.element_to_remove
	probability_of_vac = cfg.lamella_settings.probability_of_vac
	add_vacancies_toggle = cfg.lamella_settings.add_vacancies_toggle
	
	###Configuring computational environment

	#Here we are deciding if cpu or gpu computing happens
	use_gpu = cfg.gpu_related.use_gpu
	if use_gpu:
		abtem.config.set({"device": "gpu", "fft": "cufft",'dask.lazy': True})
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


	#Number of frozen phonons
	frozen_phonons = cfg.simulations.frozen_phonons
	if frozen_phonons == 'None':
		frozen_phonons = None

	fph_sigma = cfg.simulations.fph_sigma
	if isinstance(fph_sigma, bool):
		fph_sigma = None

	return RunContext(
		cfg=cfg,
		folder_sim=folder_sim,
		folder=folder,
		do_full_run=do_full_run,
		HT_value=HT_value,
		do_diffraction=do_diffraction,
		override_sampling=override_sampling,
		scan_start=scan_start,
		scan_stop=scan_stop,
		lamella_sizes=lamella_sizes,
		global_tilt=global_tilt,
		tilt_degrees=cfg.lamella_settings.tilt_degrees,
		haadf_detector=haadf_detector,
		abf_detector=abf_detector,
		bf_detector=bf_detector,
		element_to_remove=element_to_remove,
		probability_of_vac=probability_of_vac,
		add_vacancies_toggle=add_vacancies_toggle,
		frozen_phonons=frozen_phonons,
		fph_sigma=fph_sigma,
	)

def save_config(cfg, path: str ):
	path = Path(path)
	with path.open("wb") as f:		   # note: binary mode
		tomli_w.dump(cfg, f)

def add_potential(surf):
	# Make the potential
	potential = abtem.Potential(
	surf,
	sampling=0.05,   # real space sampling
	projection='infinite',
	parametrization='kirkland',
	periodic=False,
	).build().compute()

	return potential

def add_frozen_phonons_potential(ctx,surf):
	frozen = abtem.FrozenPhonons(surf, num_configs=ctx.frozen_phonons, sigmas=ctx.fph_sigma, seed=100)
	# Make the potential
	potential = abtem.Potential(
	frozen,
	sampling=0.05,   # real space sampling
	projection='infinite',
	parametrization='kirkland',
	periodic=False,
	)#.build().compute() #once-done is faster, but requires a huge amount of memory
	
	return potential

def add_probe(ctx, potential):
	# Make the probe
	#tilt = (0,0) #this one is a beam tilt not the sample one
	probe = abtem.Probe(
	energy=ctx.HT_value, 
	semiangle_cutoff=30, 
	defocus="scherzer"
	)
	#,
	#tilt=tilt
	# Match probe grid to potential grid
	probe.grid.match(potential)
	#print('tilt',global_tilt)
	return probe

def add_scan(ctx, probe,pot):
	# Define scan area
	print('Proposed sampling',probe.ctf.nyquist_sampling * .9)
	if not ctx.override_sampling:
		sampling = probe.ctf.nyquist_sampling * .9
	else:
		sampling = ctx.override_sampling
		print('Overrided sampling',sampling)

	scan = abtem.scan.GridScan(
		start = ctx.scan_start, 
		end = ctx.scan_stop, 
		sampling = sampling,
		potential=pot
	)
	
	return scan   

def plot_diffraction(ctx, pot,fname,ftitle):
	fname = str(fname)
	
	initial_waves = abtem.PlaneWave(energy=ctx.HT_value)
	try:
		exit_waves = initial_waves.multislice(pot.compute()).compute()
	except:
		exit_waves = initial_waves.multislice(pot).compute()
	#exit_waves_raw = initial_waves.multislice(pot).to_cpu()

	#exit_waves = exit_waves_raw.compute()
	print('Exit waves')
	diffraction_patterns = exit_waves.diffraction_patterns(max_angle="valid", block_direct=True).compute().to_cpu()
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

def prepare_job(ctx, hkl_set,is_uvw=True,inplane_angle=None):
	'''
	This function prepares a set of ase objects to use for the further computations
	Inputs:
		hkl_set - list (Nx3), list of hkl (or uvw) vectors
		is_uvw - boolean, defines are vectors provided as hkl to use a normal to the corresponding plane, or as uvw
		inplane_angle - float | None, inplane slab rotation in degrees (if defined) prior crop
	Output:
		full_dataset - dict, structured in a local format
	'''
	
	cfg = ctx.cfg
	full_dataset = {}
	count = 0

	borders = cfg.lamella_settings.borders
	tol = cfg.lamella_settings.tol
	max_uvw = cfg.lamella_settings.max_uvw
	sblock_size = cfg.lamella_settings.sblock_size
	atom_to_zero = cfg.lamella_settings.atom_to_zero
	extra_shift_z = cfg.lamella_settings.extra_shift_z


	for i in hkl_set.keys():
		for j in hkl_set[i]:
			print('Generating',i,j)
			#scell = struct_set[i]
			cif_path = ctx.folder + cif_files[i]
			surf = sim.make_lamella(cif_path,j,sblock_size,ctx.lamella_sizes,atom_to_zero,tol,max_uvw,
						is_uvw=is_uvw,inplane_angle=inplane_angle,
						extra_shift_z=extra_shift_z,vac_xy=borders,vac_z=borders,
						global_tilt=ctx.global_tilt,tilt_degrees=ctx.tilt_degrees)
						
			if ctx.add_vacancies_toggle:
				surf = sim.add_vacancies(surf,ctx.element_to_remove,ctx.probability_of_vac)
				print('Vacancies applied to '+ctx.element_to_remove+ ', probability '+str(ctx.probability_of_vac))
			
			potential = add_potential(surf)
			#potential.compute()
			fph_potential = add_frozen_phonons_potential(ctx,surf)
			#fph_potential.compute()
			probe = add_probe(ctx,potential)
			fph_probe = add_probe(ctx,fph_potential)
			data = {
				'symm':i,
				'hkl':j,
				'surface':surf,
				'potential':potential,
				'probe':probe,
				'fph_potential':fph_potential,
				'fph_probe':fph_probe,
				'scan':add_scan(ctx,probe,potential)
			}
			full_dataset[count] = data
			count+=1
	return full_dataset   

def simulation_run(s,cfg,
	is_uvw=True,
	inplane_angle=None,
	do_full_run=False
	):
	'''
	Main starter function
	Inputs:
		s - list (Nx3), list of hkl (or uvw) vectors
		is_uvw - boolean, defines are vectors provided as hkl to use a normal to the corresponding plane, or as uvw
		inplane_angle - float | None, inplane slab rotation in degrees (if defined) prior crop
		do_full_run - boolean
	'''
	###

	#cfg = confread.load_config(config_path)
	ctx = resolve_context(cfg, global_tilt=None)
	
	#cp.cuda.Stream.null.synchronize()
	dataset = prepare_job(ctx,s,is_uvw,inplane_angle)
	for i in dataset.keys():
		#cfg_out_path = ctx.folder_sim+dataset[i]['symm']+'_'
		#cfg_out_path += ''.join([str(q) for q in dataset[i]['hkl']])+'_' +str(ctx.global_tilt) +'.cfg'
		
		out_dir = Path(ctx.folder_sim)
		sg = dataset[i]['symm']
		line_hkl = ''.join([str(q) for q in dataset[i]['hkl']])
		
		cfg_out_path = out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}.toml"
		
		run_cfg = effective_cfg_for_run(ctx)
		save_config(run_cfg, cfg_out_path )

		sim.plot_dataset(dataset[i],is_uvw,cfg.lamella_settings.scan_s,
			cfg.lamella_settings.borders,str(out_dir)+'/',cfg.paths.sample_name,ctx.global_tilt)
		
		#surf_fname = ctx.folder_sim+dataset[i]['symm']+'_'+line_hkl+'_'+str(ctx.global_tilt)+'surf.xyz'
		surf_fname = out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_surf.xyz"
		ase.io.write(surf_fname, dataset[i]['surface'], 'xyz')

		
		if ctx.do_diffraction:
			print('Diffraction - single')
			ttl = sg+', [' + line_hkl +'], '+ str(ctx.lamella_sizes[0])+'x'+str(ctx.lamella_sizes[1])+'x'+str(ctx.lamella_sizes[2])+'$\AA$'
			#plot_diffraction(ctx,dataset[i]['potential'],ctx.folder_sim+sg+'_'+line_hkl+'_'+str(ctx.global_tilt)+'_single_diff.png',ttl)
			print('Diffraction - fph')
			#plot_diffraction(ctx,dataset[i]['fph_potential'],ctx.folder_sim+sg+'_'+line_hkl+'_'+str(ctx.global_tilt)+'_fph_diff.png',ttl+', '+str(ctx.frozen_phonons)+' fph')
			
			plot_diffraction(ctx,dataset[i]['potential'], out_dir / f"{sg}_{line_hkl}_single_diff.png", ttl)
			plot_diffraction(ctx,dataset[i]['fph_potential'], out_dir / f"{sg}_{line_hkl}_fph_diff.png", ttl+', '+str(ctx.frozen_phonons)+' fph')
						
		if ctx.do_full_run:
			potential = dataset[i]['potential']
			probe = add_probe(ctx,potential)
			probe.grid.match(potential)		
			scan = add_scan(ctx,probe,potential)
			
			#Single img
			#'''
			measurements = probe.scan(potential, scan=scan, detectors=[ctx.haadf_detector,ctx.abf_detector,ctx.bf_detector])
			img = measurements.compute()#(scheduler="threads", num_workers=num_workers)
			
			w = 0
			while w<len(img):
				iimg = img[w].copy()
				det_s = ['haadf','abf','bf'][w]
				#iimg.to_tiff(ctx.folder_sim+dataset[i]['symm']+'_'+str(ctx.global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'.tif')
				#iimg.to_zarr(ctx.folder_sim+dataset[i]['symm']+'_'+str(ctx.global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'.zarr',overwrite=True)
				
				iimg.to_tiff(str(out_dir / f"{sg}_{ctx.global_tilt}_{line_hkl}_{det_s}.tif"))
				iimg.to_zarr(str(out_dir / f"{sg}_{ctx.global_tilt}_{line_hkl}_{det_s}.zarr"), overwrite=True)

				for k in [0.025,0.1,0.25]:
					blurred = iimg.gaussian_filter(k,boundary='constant')
					#blurred.to_tiff(ctx.folder_sim+dataset[i]['symm']+'_'+str(ctx.global_tilt)+'_'+line_hkl+'_'+det_s+'_'+str(k).replace('.','-')+'.tif')
					blurred.to_tiff(str(out_dir / f"{sg}_{ctx.global_tilt}_{line_hkl}_{det_s}_{str(k).replace('.','-')}.tif"))
				w+=1
			#'''

			#frozen phonon set
			fph_potential = dataset[i]['fph_potential']
			probe.grid.match(potential)
			scan = add_scan(ctx,probe,potential)
			fph_measurements = probe.scan(fph_potential, scan=scan, detectors=[ctx.haadf_detector,ctx.abf_detector])
			img = fph_measurements.compute()#(scheduler="threads", num_workers=num_workers)

			w = 0
			while w < len(img):
				iimg = img[w].copy()
				det_s = ['haadf','abf','bf'][w]
				#iimg.to_tiff(ctx.folder_sim+'fph_'+dataset[i]['symm']+'_'+str(ctx.global_tilt)+'_'+line_hkl+'_'+det_s+'.tif')
				#iimg.to_zarr(ctx.folder_sim+'fph_'+dataset[i]['symm']+'_'+str(ctx.global_tilt)+'_'+line_hkl+'_'+ det_s+'.zarr',overwrite=True)	
				iimg.to_tiff(str(out_dir / f"fph_{sg}_{ctx.global_tilt}_{line_hkl}_{det_s}.tif"))
				iimg.to_zarr(str(out_dir / f"fph_{sg}_{ctx.global_tilt}_{line_hkl}_{det_s}.zarr"), overwrite=True)
                 			
				for k in [0.025,0.1,0.25]:
					blurred = iimg.gaussian_filter(k,boundary='constant')
					#blurred.to_tiff(ctx.folder_sim+'fph_'+dataset[i]['symm']+'_'+str(ctx.global_tilt)+'_'+line_hkl +'_'+det_s+'_'+str(k).replace('.','-')+'.tif')
					blurred.to_tiff(str(out_dir / f"fph_{sg}_{ctx.global_tilt}_{line_hkl}_{det_s}_{str(k).replace('.','-')}.tif"))
				w+=1
	del dataset

cfg0 = confread.load_config("config.toml")
for cfg_run in expand_cfg(cfg0):
	simulation_run({'Pm3m': [[1,1,0]]}, cfg_run, do_full_run=False)

print('Finished')
