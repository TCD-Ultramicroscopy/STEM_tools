#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

'''
This code is only for reading and validating the config.toml
Need to be edited only if new variables are added or config file is splitted
'''

from pydantic import BaseModel, Field, ValidationError
from typing import Union
import tomllib
from pathlib import Path
import os

#config_fname = 'config.toml'

#If adding a new class of variables, add it to AppConfig, too!
class Paths(BaseModel):
	folder_sim: str = Field()
	extr: str = Field()
	folder: str = Field()
	sample_name: str = Field()
	
class GPU_related(BaseModel):
	use_gpu: bool = Field()
	dask_cuda: bool = Field()
	cupy_fft_cache_size: str = Field()
	dask_chunk_size_gpu: str = Field()
	dask_chunk_size: str = Field()

class Simulations(BaseModel):
	override_sampling: float | bool = Field()
	frozen_phonons: int | str = Field() #str meant to be only 'None'
	fph_sigma: float | bool = Field() #bool meant to be converted to None
	do_full_run: bool = Field()

class Microscope(BaseModel):
	HT_value: int = Field()
	do_diffraction: bool = Field()
	haadfinner: float = Field()
	haadfouter: float = Field()
	abfinner: float = Field()
	abfouter: float = Field()
	bfinner: float = Field()
	bfouter: float = Field()
	
class Lamella_Settings(BaseModel):
	max_uvw: int = Field()
	sblock_size: float = Field()
	scan_s: float = Field()
	borders: float = Field()
	thickness: float = Field()
	extra_shift_z: float = Field()
	tol: float = Field()
	atom_to_zero: str = Field()
	global_tilt_a: float = Field()
	global_tilt_b: float = Field()
	add_vacancies_toggle: bool = Field()
	element_to_remove: str = Field()
	probability_of_vac: float = Field()

class AppConfig(BaseModel):
	paths: Paths
	gpu_related: GPU_related
	microscope: Microscope
	lamella_settings: Lamella_Settings
	simulations: Simulations

def load_config(path: str | Path = 'config.toml') -> AppConfig:
	here = Path(__file__).resolve()
	#print(path)
	full_path = here.parent / path
	with full_path.open("rb") as f:
		data = tomllib.load(f)
	return AppConfig.model_validate(data)
