#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

'''
This code is only for reading and validating the config.toml
Need to be edited only if new variables are added or config file is splitted
'''

from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Union, Any
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
	
class Job(BaseModel):
	"""
	Job-defining parameters (should be only one per config TOML).
	- phase: CIF filename or a key
	- hkl_to_do: either [h,k,l] or [[h,k,l], ...]
	- is_uvw: whether HKL vectors are interpreted as UVW
	"""
	phase: str = Field()
	hkl_to_do: list[int] | list[list[int]] = Field()
	is_uvw: bool = Field()
	phonons_seed: int = Field(default=0)

	@field_validator("hkl_to_do")
	@classmethod
	def validate_hkl_to_do(cls, v: Any):
		# Accept either [h,k,l] or [[h,k,l], ...]
		if isinstance(v, list) and len(v) == 3 and all(isinstance(x, int) for x in v):
			return v
		if isinstance(v, list) and all(isinstance(row, list) for row in v):
			for row in v:
				if len(row) != 3 or not all(isinstance(x, int) for x in row):
					raise ValueError("Each HKL entry must be a list of 3 integers.")
			return v
		raise ValueError("hkl_to_do must be [h,k,l] or a list of [h,k,l] entries.")
	
class GPU_related(BaseModel):
	use_gpu: bool = Field()
	dask_cuda: bool = Field()
	cupy_fft_cache_size: str = Field()
	dask_chunk_size_gpu: str = Field()
	dask_chunk_size: str = Field()

class Simulations(BaseModel):
	override_sampling: float | bool = Field()
	frozen_phonons: int | str | list[int | str] = Field() #str meant to be only 'None'
	fph_sigma: float | bool | str | list[float | bool | str] = Field() #bool meant to be converted to None
	do_full_run: bool = Field()

class Microscope(BaseModel):
	HT_value: int | list[int ] = Field()
	do_diffraction: bool = Field()
	convergence_angle: float = Field(default=30.0)   # mrad
	cbed_max_angle: float | str = Field(default="valid")
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
	thickness: float | list[float] = Field()
	extra_shift_z: float = Field()
	tol: float = Field()
	atom_to_zero: str = Field()
	global_tilt_a: float | list[float] = Field()
	global_tilt_b: float | list[float] = Field()
	tilt_degrees: bool = Field()
	add_vacancies_toggle: bool = Field()
	element_to_remove: str = Field()
	probability_of_vac: float | list[float] = Field()

class AppConfig(BaseModel):
	paths: Paths
	gpu_related: GPU_related
	microscope: Microscope
	lamella_settings: Lamella_Settings
	simulations: Simulations
	job: Job

def load_config(path: str | Path = 'config.toml') -> AppConfig:
	here = Path(__file__).resolve()
	#print(path)
	full_path = here.parent / path
	with full_path.open("rb") as f:
		data = tomllib.load(f)
	return AppConfig.model_validate(data)
