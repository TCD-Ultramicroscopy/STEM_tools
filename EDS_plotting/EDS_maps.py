#%matplotlib inline
import numpy as np
#import exspy
#import hyperspy
import hyperspy.api as hs
import exspy.material as hsm
#import hyperspy.utils.material as hsm

import matplotlib.pyplot as plt
import cv2
import scipy
import math
import os

import cairo
import gi

import skimage

gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo

hs.set_log_level('INFO')

storage = '/home/vasily/test_EDX_map/'
folders = os.listdir(storage)
fs = 30

def add_text(fr,x0,y0,w,h,text):
	#add first label
	abc_figure_bgra = cv2.cvtColor(fr, cv2.COLOR_BGR2BGRA)
	surface = cairo.ImageSurface.create_for_data(
		abc_figure_bgra,
		cairo.FORMAT_ARGB32,
		abc_figure_bgra.shape[1],
		abc_figure_bgra.shape[0],
		abc_figure_bgra.strides[0]
	)
	context = cairo.Context(surface)
	#context.set_source_rgb(1, 1, 1)
	pango_layout = PangoCairo.create_layout(context)
	pango_layout.set_markup(text)
	text_width, text_height = pango_layout.get_size()
	text_width /= Pango.SCALE
	text_height /= Pango.SCALE
	PangoCairo.show_layout(context, pango_layout)
	abc_figure = cv2.cvtColor(abc_figure_bgra, cv2.COLOR_BGRA2BGR)
	return abc_figure

def check_text_size(text,frame_w,frame_h):
	fr = np.ones((frame_h,frame_w,3),dtype=np.uint8)
	abc_figure_bgra = cv2.cvtColor(fr, cv2.COLOR_BGR2BGRA)
	surface = cairo.ImageSurface.create_for_data(
		abc_figure_bgra,
		cairo.FORMAT_ARGB32,
		abc_figure_bgra.shape[1],
		abc_figure_bgra.shape[0],
		abc_figure_bgra.strides[0]
	)
	context = cairo.Context(surface)
	context.set_source_rgb(1, 1, 1)
	pango_layout = PangoCairo.create_layout(context)
	#text = "<span font='Sans 60'>HR-STEM</span>"
	pango_layout.set_markup(text)
	width, height = pango_layout.get_size()
	width /= Pango.SCALE
	height /= Pango.SCALE

	return int(height),int(width)
	
def concat(arr,axis=0):
	if len(arr) <= 1:
		print('None')
		return arr
	elif len(arr) == 2:
		return np.concatenate((arr[0],arr[1]), axis=axis)
	else:
		return concat([arr[0],concat(arr[1:], axis=axis)], axis=axis)

def map_processing(folder):

	SI = hs.load(filenames= folder+'EDS Spectrum Image.dm3', signal_type="EDS_TEM")
	png_im_path = folder+'processed_data/png/' + 'HAADF.png'
	
	
	t = SI
	t.set_microscope_parameters(beam_energy=300)
	t.axes_manager.signal_axes[0].units = 'keV'
	
	t.add_elements(['C','O','Ga','Au','As','Cr','Hf','Si','Cu'])
	t.add_lines()
	#t.change_dtype('unit8')
	
	lines = ['Cu_Ka','C_Ka','O_Ka','Ga_La','Ga_Lb1','Ga_Ka','Si_Ka','Au_La','Cu_La','Cr_Ka','Au_Ma','Ga_Ka',
						'Fe_Ka','Co_Ka','Hf_La','Hf_Lb1','As_Ka']#
	plt_list = ['Ga_La','Cr_Ka','Hf_Lb1','Au_La','O_Ka','Cu_Ka']#'Cu_Ka','Hf_La'
	#plt_list_sum = ['Hf_La','Au_M','Al_Ka','Si_Ka']

	#maps will be colorized correspondingly
	colorcodes = {'Ga_La':'f03ff7','Cr_Ka':'0000FF','Hf_Lb1':'40f73f','Au_La':'f7b83f'}
	#'Cu_Ka':'000000','Au_La':'f7b83f','O_Ka':'db3939', 
	
	#colorcodes = {}
	#color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
	#i = 0
	#while i<len(plt_list):
	#	colorcodes[plt_list[i]] = color_cycle[i][1:]
	#	i+=1

	fin_pixsize_map = 1000
	
	col_rgb = {i:(int(colorcodes[i][0:2],16),int(colorcodes[i][2:4],16),int(colorcodes[i][4:6],16)) for i in colorcodes.keys()}
	col_bgr = {i:(col_rgb[i][2],col_rgb[i][1],col_rgb[i][0]) for i in col_rgb.keys()}
	#to assess other aspect ratios!
	labels_sizes = {i:check_text_size("<span font='Sans "+str(fs)+"'>"+i.replace('_',' ')+"</span>",
				fin_pixsize_map,fin_pixsize_map) for i in col_rgb.keys()}
	
	print(col_rgb,labels_sizes)
	
	#plt_list_sum = ['Hf_La','Au_M','Al_Ka','Si_Ka']
	
	
	##########################################
	#Normalization
	norm = False
	sub_zero = False
	ref_line = 'Cu_Ka'
	zero_line = 'V_Ka'
	norm_sum = False
	##########################################
	
	if norm_sum:
		norm = False
		sub_zero = False
	
	#Output parameters (images stack)
	col_number = 5
	row_number = math.ceil((len(plt_list) + 1)/col_number)
	
	#lines = ['Cu_Ka','Mo_Ka','Te_La','Sb_La','Ga_Ka','Au_La','Ag_La']
	data = [ t.get_lines_intensity([i]) for i in lines ]
	
	#Here there is an issue with pillow&imageio libs
	#data[0]
	i = 0
	while i < len(data):
		tl = data[i][0].metadata['General']['title'].split(': ')[1].split(' at')[0]
		#print(data[i][0])
		#print(tmp_fr)
		norm_d = cv2.normalize(data[i][0].data, None, 0, 255, cv2.NORM_MINMAX)
		cv2.imwrite(folder+'/'+tl+ '.png', norm_d.astype(np.uint8) )
		#data[i][0].save(folder+'/'+tl+ '.png', overwrite = True)
		i+=1
	
	SI.add_lines(lines)
	s_sum = SI.sum()
	s_sum.plot(True)
	s_sum.save(folder+'EDS.msa', overwrite = True)
	s_sum.plot(True)
	fig = s_sum._plot.signal_plot.figure
	fig.savefig(folder+'sum_EDS.png')
	
	plot_total_sp(s_sum,folder)
	#plt.ylim([0,1000])
	#plt.xlim([0,10])
	
	#If there is a need to crop
	'''
	fr1 = [1,29,1,30] #x1,y1,x2,y2
	fr = [fr1]
	
	im = t.to_signal2D()
	im_crop = [ im.isig[i[0]:i[1],i[2]:i[3]] for i in fr ]
	fr_sum = [ i.to_signal1D().sum() for i in im_crop ]
	
	i = 0
	while i<len(fr_sum):
		#fr_sum[i].save('fr'+str(i)+'.msa')
		#fr_sum[i].save(folder+m+'/area'+str(i)+'.png')
		fr_sum[i].plot(True)
		i+=1
	'''
	
	s1 = t.integrate1D(0)
	data1 = [ s1.get_lines_intensity([i]) for i in lines ]
	tmp = s1.data
	data_sum = [ sum(i) for i in tmp ]
	
	#x = data1[0]
	#x
	#'''
	for i in data1:
		plt.close()
		tl = i[0].metadata['General']['title'].split(': ')[1].split(' at')[0]
		print(tl)
		j = i[0].data
		fig, ax = plt.subplots()
		plt.plot(range(0,len(j)),j,".r")
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		ax.yaxis.tick_right()
		ax.yaxis.set_label_position("right")
		ax.set_ylim(0,max(j))
	
		ax.xaxis.label.set_size(20)
		ax.yaxis.label.set_size(20)
		ax.tick_params(labelsize=18)
		plt.savefig(folder+'sp_'+tl+'.png')

	#'''
	
	plt.close()
	
	fig, ax = plt.subplots()
	plt.subplots_adjust(right=0.85, left=0, top=.95, bottom=0.05)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	ax.yaxis.tick_right()
	ax.yaxis.set_label_position("right")
	ax.set_ylim(min(-np.array(range(0,len(j)))),max(-np.array(range(0,len(j)))))
	
	ax.xaxis.label.set_size(20)
	ax.yaxis.label.set_size(20)
	ax.tick_params(labelsize=18)
	
	for i in data1:
		j = i[0].data
		tl = i[0].metadata['General']['title'].split(': ')[1].split(' at')[0]
		if ref_line in tl:
			ref_I = j
		if zero_line in tl:
			zero_I = j
	
	for i in data1:
		j = i[0].data
		tl = i[0].metadata['General']['title'].split(': ')[1].split(' at')[0]
		p = False
		for k in plt_list:
			if k in tl:
				p = True
				if k == ref_line: #and norm
					p = False
				print(k)
		if p:
			I_EDX = j
			if sub_zero:
				I_EDX = I_EDX - zero_I
				ref_I = ref_I - zero_I
			if norm:
				I_EDX = I_EDX / ref_I
			if norm_sum:
				I_EDX = I_EDX / data_sum
			#plt.plot(j/ref_I,-np.array(range(0,len(j))),'o-',mfc='none',label=tl,markersize=3,linewidth=1.2)
			plt.plot(I_EDX,-np.array(range(0,len(j))),'o-',mfc='none',label=tl,markersize=3,linewidth=1.2)
			np.savetxt(folder+'linescan_'+tl+'.csv',I_EDX)
			
	plt.legend()
	plt.ylabel("$x$, nm")
	ax.set_title("EDX intensity, rel.u.", fontsize=16, style='italic')
	plt.savefig(folder+'sum'+'_all.png',dpi=300)
	plt.close()
	
	lin_im_path = folder+'sum_all.png'
	#png_im_path = folder + 'DF.png'#'HAADF.png'#to be generalized
	#png_im_path = folder+'processed_data/png/' + 'DF.png'
	fin_im_path = folder+'1_combined.png'
	scale_factor=.9
	lin = cv2.imread(lin_im_path)
	print(lin.shape)
	png = cv2.imread(png_im_path)
	print(png.shape)
	scale_factor = scale_factor/(png.shape[0]/lin.shape[0])
	print(scale_factor)
	png2 = cv2.resize(png, None,fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)
	lin2 = cv2.resize(lin, None,fx=1/scale_factor, fy=1/scale_factor, interpolation = cv2.INTER_CUBIC)
	
	h1 = lin.shape[0]
	h2 = png2.shape[0]
	l1 = lin.shape[1]
	l2 = png2.shape[1]
	#print(lin2.shape,h1,l1)
	
	final_image = np.ones((max(h1,h2),l1+l2,3),dtype=np.uint8)
	final_image = final_image*255
	final_image[-h1:,-l1:] = lin
	final_image[int(abs(h1-h2)/2):int(abs(h1-h2)/2)+h2,:l2] = png2
	
	cv2.imwrite(fin_im_path,final_image)
	cv2.destroyAllWindows()
	
	
	#Cumulative maps
	
	font				   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,75)
	fontScale			  = 2.5
	fontColor			  = (0,0,255)
	thickness			  = 3
	lineType			   = 2
	im_height = 1024
	
	#
	#'Cu_Ka'
	elist = sorted(plt_list)
	
	imlist = []
	img = cv2.imread(png_im_path)
	print(img.shape)
	img = cv2.resize(img, None,fx=im_height/img.shape[0], fy=im_height/img.shape[0], interpolation = cv2.INTER_LINEAR)
	print(img.shape)
	cv2.putText(img,'DF', 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			thickness,
			lineType)
	
	imlist.append(img)
	
	#for i in elist:
	#	img = cv2.imread(folder+"_label"+i+".png")
	#	imlist.append(img)
	
	for i in elist:
		print(i)
		img = cv2.imread(folder+i+'.png')
		img = cv2.resize(img, None,fx=im_height/img.shape[0], fy=im_height/img.shape[0], interpolation = cv2.INTER_LINEAR)
		cv2.putText(img,i.split('_')[0]+' '+i.split('_')[1], 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			thickness,
			lineType)
		cv2.imwrite(folder+"_label"+i+".png", img)
		imlist.append(img)
		if i in col_bgr.keys():
			img_gray = cv2.imread(folder+i+'.png',cv2.IMREAD_GRAYSCALE)
			img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
			img_norm = img_gray/255.
			color_fr = np.zeros_like(img_gray_rgb)
			color_fr[:, :] = col_bgr[i]
			img = (color_fr * img_norm[:, :, None]).astype(np.uint8)
			img = cv2.resize(img, None,fx=fin_pixsize_map/img.shape[0], fy=fin_pixsize_map/img.shape[0], interpolation = cv2.INTER_LINEAR)
			cv2.putText(img,i.split('_')[0]+' '+i.split('_')[1], 
				bottomLeftCornerOfText, 
				font, 
				fontScale,
				fontColor,
				thickness,
				lineType)
			cv2.imwrite(folder+"_c_label"+i+".png", img)
		
	
	
	cv2.imwrite(folder+"test.png", imlist[1])

	###########Overlay maps########
	label_h = int(max([labels_sizes[i][0] for i in labels_sizes.keys()]))
	label_w = {i:labels_sizes[i][1] for i in labels_sizes.keys()}
	print(label_h,label_w,'label_w')
	img_overlay = np.zeros((fin_pixsize_map,fin_pixsize_map,3),dtype=np.uint8)
	#will not work for non-square! to be fixed
	#fref = cv2.imread(fft_ref_path) #if reference is available
	
	for i in col_bgr.keys():
		img_gray = cv2.imread(folder+i+'.png',cv2.IMREAD_GRAYSCALE)
		img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
		img_norm = img_gray/255.
		color_fr = np.zeros_like(img_gray_rgb)
		color_fr[:, :] = col_bgr[i]
		img = (color_fr * img_norm[:, :, None]).astype(np.uint8)
		img = cv2.resize(img, None,fx=fin_pixsize_map/img.shape[0], fy=fin_pixsize_map/img.shape[0], interpolation = cv2.INTER_LINEAR)
		img_overlay = cv2.add(img_overlay, img)

	backgrounds = {}
	for i in col_bgr.keys():
		img = np.ones((label_h,label_w[i],3),dtype=np.uint8)*0
		col = str(colorcodes[i])
		print(col)
		img = add_text(img,0,0,label_w[i],label_h,"<span font='Sans "+str(fs)+"' color='#"+col+"'>"+i.replace('_',' ')+"</span>")
		backgrounds[i] = img
		'''
		cv2.putText(img,i.split('_')[0]+' '+i.split('_')[1], 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			thickness,
			lineType)
		cv2.imwrite(folder+"_label"+i+".png", img)
		'''
	
	gap = int(fin_pixsize_map*0.005)

	dist = int(gap)
	
	for i in col_bgr.keys():
		print(backgrounds[i].shape,gap,dist,label_h,label_w[i])
		print(img_overlay[-dist-label_w[i]:-dist,-gap-label_h:-gap].shape)
		img_overlay[-gap-label_h:-gap,-dist-label_w[i]:-dist] = backgrounds[i]
		
		dist+=gap+label_w[i]
	#img_overlay = cv2.add(img_overlay, fref)
	cv2.imwrite(folder+"overlay.png", img_overlay)
	
	blank_im = np.ones(imlist[0].shape,dtype=np.uint8)
	blank_im = blank_im*255
	
	col_number = 3
	row_number = math.ceil((len(plt_list) + 1)/col_number)
	
	fin_map = concat(imlist,axis=1)
	#print(fin_map.shape)
	cv2.imwrite(folder+"map_test.png", fin_map)

def plot_total_sp(sp,savename):

    E_FeKa = hsm.elements.Fe.Atomic_properties.Xray_lines.Ka['energy (keV)']
    E_CrKa = hsm.elements.Cr.Atomic_properties.Xray_lines.Ka['energy (keV)']
    E_CrKb = hsm.elements.Cr.Atomic_properties.Xray_lines.Kb['energy (keV)']
    E_CrLa = hsm.elements.Cr.Atomic_properties.Xray_lines.La['energy (keV)']
    E_CKa = hsm.elements.C.Atomic_properties.Xray_lines.Ka['energy (keV)']
    E_GaKa = hsm.elements.Ga.Atomic_properties.Xray_lines.Ka['energy (keV)']
    E_GaKb = hsm.elements.Ga.Atomic_properties.Xray_lines.Kb['energy (keV)']
    E_GaLa = hsm.elements.Ga.Atomic_properties.Xray_lines.La['energy (keV)']

    E_HfKa = hsm.elements.Hf.Atomic_properties.Xray_lines.Ka['energy (keV)']
    E_HfKb = hsm.elements.Hf.Atomic_properties.Xray_lines.Kb['energy (keV)']
    E_HfLa = hsm.elements.Hf.Atomic_properties.Xray_lines.La['energy (keV)']
    E_HfLb1 = hsm.elements.Hf.Atomic_properties.Xray_lines.Lb1['energy (keV)']
    E_HfMa = hsm.elements.Hf.Atomic_properties.Xray_lines.Ma['energy (keV)']

    E_AuLa = hsm.elements.Au.Atomic_properties.Xray_lines.La['energy (keV)']
    E_AuLb1 = hsm.elements.Au.Atomic_properties.Xray_lines.Lb1['energy (keV)']
    E_AuMa = hsm.elements.Au.Atomic_properties.Xray_lines.Ma['energy (keV)']
    
    E_OKa = hsm.elements.O.Atomic_properties.Xray_lines.Ka['energy (keV)']
    
    
    plt.close('all')
    if 1:        
        ax = hs.plot.plot_spectra(sp, style='cascade', padding=-1)#,legend=names
        
        plt.axvline(E_OKa, c='C2', ls=':', lw=0.5) 
        plt.text(x=E_OKa+0.1, y=0.85, s='O-K', color='C2') 

        plt.axvline(E_CrKa, c='C1', ls=':', lw=0.5) 
        plt.text(x=E_CrKa+0.1, y=0.85, s='Cr-K$_\\alpha$', color='C1')
        plt.axvline(E_CrLa, c='C1', ls=':', lw=0.5) 
        plt.text(x=E_CrLa+0.1, y=0.7, s='Cr-L$_\\alpha$', color='C1')

        plt.axvline(E_CrKb, c='C1', ls=':', lw=0.5) 
        plt.text(x=E_CrKb+0.1, y=0.7, s='Cr-K$_\\beta$', color='C1')
        
        plt.axvline(E_GaKa, c='C3', ls=':', lw=0.5) 
        plt.text(x=E_GaKa+0.1, y=0.85, s='Ga-K$_\\alpha$', color='C3') 
        
        plt.axvline(E_GaKb, c='C3', ls=':', lw=0.5) 
        plt.text(x=E_GaKb+0.1, y=0.7, s='Ga-K$_\\beta$', color='C3')
        
        plt.axvline(E_GaLa, c='C3', ls=':', lw=0.5) 
        plt.text(x=E_GaLa+0.1, y=0.55, s='Ga-L$_\\alpha$', color='C3')

        plt.axvline(E_HfMa, c='C4', ls=':', lw=0.5) 
        plt.text(x=E_HfMa+0.1, y=0.4, s='Hf-M$_\\alpha$', color='C4') 
        plt.axvline(E_HfLb1, c='C4', ls=':', lw=0.5) 
        plt.text(x=E_HfLb1+0.1, y=0.55, s='Hf-L$_\\beta 1$', color='C4')
        plt.axvline(E_HfLa, c='C4', ls=':', lw=0.5) 
        plt.text(x=E_HfLa+0.1, y=0.7, s='Hf-L$_\\alpha$', color='C4') 

        plt.axvline(E_AuMa, c='C5', ls=':', lw=0.5) 
        plt.text(x=E_AuMa+0.1, y=0.85, s='Au-Ma', color='C5') 
        plt.axvline(E_AuLb1, c='C5', ls=':', lw=0.5) 
        plt.text(x=E_AuLb1+0.1, y=0.4, s='Au-L$_\\beta 1$', color='C5')
        plt.axvline(E_AuLa, c='C5', ls=':', lw=0.5) 
        plt.text(x=E_AuLa+0.1, y=0.4, s='Au-L$_\\alpha$', color='C5') 
        
        plt.axvline(8.040, c='k', ls=':', lw=0.5) 
        plt.text(x=8.14, y=0.35, s='Cu-K$_\\alpha$', color='k')
        
        #plt.title(savename)
        
        plt.xlabel('Energy, keV')
        
        ax.figure.savefig(savename+'_total_sp.png')

maps,areas = [],[]

for q in folders:
	p = storage + q
	for root,d_names,f_names in os.walk(p, topdown=True, onerror=None, followlinks=False):
		for i in d_names:
			if 'map' in i or i.startswith('m'):
				print('map',root + '/' + i)
				maps.append(root + '/' + i+'/')
			elif 'area' in i or i.startswith('a'):
				print('area',root + '/' + i)
				areas.append(root + '/' + i+'/')  
		
for f in maps:
	map_processing(f)
