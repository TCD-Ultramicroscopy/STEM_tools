#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

#import os
import numpy as np
import hyperspy.api as hs

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
