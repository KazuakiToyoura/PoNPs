#!/usr/bin/env python
import numpy as np
import converter as cvt
import sys

# Define the functions for space and time grids

# Estimate debye length (ldebye)
def mkLdebye(ldebye_str,epsilon,T_K,z,ci,n_elem):
	R = 8.3144598 # gas const. [J/K/mol]
	F = 96485.3329 # Faraday const. [C/mol]
	n_layer = ldebye_str.shape[0] # the number of layers
	ldebye = np.zeros(n_layer) # Debye length list for all layers
	for i in range(n_layer):
		if i == 0:
			ldebye[i] = np.sqrt(epsilon[i]*R*T_K/F**2/np.mean(np.dot(z**2,ci[:,1:n_elem[i]+1])))
		else:
			ldebye[i] = np.sqrt(epsilon[i]*R*T_K/F**2/np.mean(np.dot(z**2,ci[:,np.sum(n_elem[:i])+1:np.sum(n_elem[:i+1])+1])))
	print("-"*30)
	print("Estimated_ldebye[cm]:", ldebye)
	for i in range(n_layer):
		if not ldebye_str[i].lower() == 'auto':
			ldebye[i] = float(ldebye_str[i])
	print("Set_ldebye[cm]:", ldebye)
	print("-"*30,"\n\n")
	return ldebye


# Space grid
def mkGrid_space(spaceGrid,n_elem,ldebye_,n_elem_ldebye,n_ldebye,d_,K_,ci_,z,l0):
	print("-"*30)
	dx_ = [] # dx_ is a list for lengths of elements at each layer
	for i in range(n_elem.shape[0]):
		if spaceGrid[i].lower() == 'auto':
			# The constant grid near both ends:
			# n_elem_ldebye: Number of elements per Debye length.
			# n_ldebye: The region for the constant grid (n-times Debye length).
			# The geometric series grid in the bulk regions:
			# n_elem is odd, leading to c_def defined at the center of layer.
			dx_tmp_ = np.zeros(n_elem[i])
			dx_tmp_[0:n_elem_ldebye[i]*n_ldebye[i]] = ldebye_[i]/n_elem_ldebye[i]
			dx_tmp_[-n_elem_ldebye[i]*n_ldebye[i]:] = ldebye_[i]/n_elem_ldebye[i]
			d_bulk_half_ = (d_[i]-ldebye_[i]*n_ldebye[i]*2.0)/2.0
			n_half = int((n_elem[i]-n_elem_ldebye[i]*n_ldebye[i]*2-1)/2)
			
			# Determine r_grid (geometric ratio) by binary search.
			r_grid_min = 1.0
			r_grid_max = 99.0
			r_grid = 50.0
			diff = 1000.0
			dx_min_ = 0.0
			while np.abs(diff) > 1.0e-5:
				dx_min_ = d_bulk_half_ / ((r_grid**n_half-1)/(r_grid-1)+r_grid**n_half/2.0)
				diff = (dx_min_-ldebye_[i]/n_elem_ldebye[i])/ldebye_[i]
				if diff > 0.0:
					r_grid_min = r_grid
				elif diff < 0.0:
					r_grid_max = r_grid
				r_grid = (r_grid_min+r_grid_max)/2.0

			# Determine the width of each element.
			dx_min_ = d_bulk_half_ / ((r_grid**n_half-1)/(r_grid-1)+r_grid**n_half/2.0)
			for j in range(n_half):
				dx_tmp_[n_elem_ldebye[i]*n_ldebye[i]+j] = dx_min_*r_grid**j
				dx_tmp_[-n_elem_ldebye[i]*n_ldebye[i]-j-1] = dx_min_*r_grid**j
			dx_tmp_[n_elem_ldebye[i]*n_ldebye[i]+n_half] = dx_min_*r_grid**n_half
			dx_.append(dx_tmp_)
			
		else: # Read a specified file for element widths.
			dx_tmp = np.loadtxt(spaceGrid[i],dtype="float",delimiter=",",comments="#")
			dx_.append(cvt.d2dl_l(dx_tmp,l0))
			if dx_[-1].shape[0] != n_elem[i]:
				print("Warning! layer", i, ":")
				print("# of grids in spaceGrid.csv is not consistent to the number of elements in layer.csv.")
				print("Abort program!")
				sys.exit()
			
	# Make grids dx_b_, dx_m_. dx_b_ is equivalent to dx_.
	# Make dx_b_
	dx_b_ = np.concatenate(dx_,0)
	dx_b_ = np.concatenate([[0.0],dx_b_,[0.0]],0)
	print("d_total[cm]:", np.round(cvt.dl2d_l(np.sum(dx_b_),l0),decimals=16))
	print("dx_b[cm]:") 
	print("The first and last values are the widths of Reservoirs L & R.")
	print(cvt.dl2d_l(dx_b_,l0))
	print("")
	# Make dx_m_
	dx_m_ = np.zeros(np.sum(n_elem)+1)
	for i in range(np.sum(n_elem)+1):
		dx_m_[i] = (dx_b_[i]+dx_b_[i+1])/2.0
	print("dx_m[cm]:")
	print("The distances between the middles of adjacent elements.")
	print(cvt.dl2d_l(dx_m_,l0))
	print("-"*30,"\n\n")

	return dx_b_, dx_m_


# time grid
def mkGrid_time(timeGrid,n_dt_max,dt_max,dt_min,n_time,slowDefectID,d_,D_,K_,z,ci_,layerID_elem,D0,l0):
	print("-"*30)
	# Estimate td (debye time)
	D_slow_ = []
	for i in range(D_.shape[1]):
		D_slow_.append(D_[slowDefectID[i]-1,i])
	D_slow_ = np.array(D_slow_)
	td_ = np.max(d_**2/D_slow_)
	print("tau_debye[s]:", cvt.dl2d_t(td_,D0,l0))

	# Estimate tinf
	z_mat = []
	ci_mat_ = []
	for i in range(D_.shape[0]):
		z_mat.append(np.full(D_.shape[1],z[i]))
		ci_mat_.append([])
		for j in range(D_.shape[1]):
			ci_mat_[-1].append(np.mean(ci_[i,np.where(layerID_elem==j+1)]))
	z_mat = np.array(z_mat)
	ci_mat_ = np.array(ci_mat_)
	tinf_ = np.min(K_/np.sum(z_mat**2*D_*ci_mat_,axis=0))
	print("tau_infinity[s]:", cvt.dl2d_t(tinf_,D0,l0))

	if timeGrid.lower() == 'auto':
		# Geometric series grid from tinf_/100 to td_.
		# The geometric ratio is determined automatically.
		if dt_max.lower() == 'auto':
			dt_max_ = td_
		else:
			dt_max_ = cvt.d2dl_t(float(dt_max),D0,l0)
		if dt_min.lower() == 'auto':
			dt_min_ = tinf_/100.0
		else:
			dt_min_ = cvt.d2dl_t(float(dt_min),D0,l0)
			if dt_min_ > tinf_ :
				print("Warning!")
				print("The minimum time grid is longer than tau_infinity. Decrease dt_min.")

		dt_ = np.full(n_time, dt_max_)
		ratio = 10.0**(np.log10(dt_max_/dt_min_)/(n_time-n_dt_max-1))
		for i in range(n_time-n_dt_max):
			dt_[i] = dt_min_*ratio**i
	
	else:
		# Read a specified file
		dt = np.loadtxt(fname=timeGrid,dtype="float",delimiter=",",comments="#")
		dt_ = d2dl_t(dt)
		if dt_.shape[0] != n_time:
			print("Warning!")
			print("The number of time intervals in timeGrid.csv is not equal to the number of time grids in comp.csv.")
			print("Abort program!")
			sys.exit()
	
	# Estimate total simulation time
	dt_sum_ = np.sum(dt_)
	print("t_total[s]:", cvt.dl2d_t(np.sum(dt_),D0,l0))
	if dt_sum_ < td_ :
		print("Warning!")
		print("The total simulation time is shorter than tau_debye. Extend simulations.")
	print("dt[s]:")
	print(cvt.dl2d_t(dt_,D0,l0))
	print("-"*30,"\n\n")
	return dt_


# frequency grid
def mkGrid_freq(freqGrid,freq_max,freq_min,n_freqSteps,D0,l0):
	if freqGrid.lower() == 'auto':
		# Geometric series grid from freq_min to freq_max.
		# The geometric ratio is determined automatically.
		ratio = 10.0**(np.log10(freq_max/freq_min)/(n_freqSteps))
		tmp = np.full(n_freqSteps+1,ratio)
		tmp[0] = freq_min
		freq = np.cumprod(tmp)
	else:
		freq = np.loadtxt(fname=freqGrid,dtype="float",delimiter=",",comments="#")
	w = freq*2.0*np.pi
	w_ = cvt.d2dl_w(w,D0,l0)
	return w_,w,freq


