#!/usr/bin/env python
import numpy as np

# Calculate impedance spectra from V profile by slight change in current I --> I+delta_I.
def calcImp(w_,dt_,delta_I_,voltage_):
	voltage_ = np.abs(voltage_)
	t_ = np.cumsum(dt_)
	t_ = np.concatenate([np.array([0.0]),t_])
	delta_V_re = np.zeros(w_.shape[0])
	delta_V_im = np.zeros(w_.shape[0])
	for i in range(w_.shape[0]):
		for j in range(t_.shape[0]-1):
			delta_V_re[i] = delta_V_re[i] + (voltage_[j+1]-voltage_[j])*(np.cos(w_[i]*t_[j+1])-np.cos(w_[i]*t_[j]))/w_[i]**2/dt_[j] \
					            + ((voltage_[j+1]-voltage_[-1])*np.sin(w_[i]*t_[j+1])-(voltage_[j]-voltage_[-1])*np.sin(w_[i]*t_[j]))/w_[i]
			delta_V_im[i] = delta_V_im[i] - (voltage_[j+1]-voltage_[j])*(np.sin(w_[i]*t_[j+1])-np.sin(w_[i]*t_[j]))/w_[i]**2/dt_[j] \
					            + ((voltage_[j+1]-voltage_[-1])*np.cos(w_[i]*t_[j+1])-(voltage_[j]-voltage_[-1])*np.cos(w_[i]*t_[j]))/w_[i]
		delta_V_im[i] = delta_V_im[i] - (voltage_[-1]-voltage_[0])/w_[i]

	z_re_ = - delta_V_im * w_ / delta_I_
	z_im_ = delta_V_re * w_ / delta_I_
	return z_re_, z_im_



