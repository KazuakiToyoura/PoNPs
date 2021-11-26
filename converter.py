#!/usr/bin/env python
import numpy as np

#Define physical and chemical constants
##############################################################
R = 8.3144598 # [J/K/mol]
F = 96485.3329 # [C/mol]
NA = 6.02214086 * 10**23 #[/mol]
e = 1.60217662*10**(-19)  # Elementary charge [C]
##############################################################

# Define conversion functions between variables with and without unit.
def d2dl_flux(flux,D0,l0,c0): # flux [mol/cm2/s]
	flux_dl = flux * l0 / D0 / c0
	return flux_dl
def dl2d_flux(flux_dl,D0,l0,c0):
	flux = flux_dl * D0 * c0 / l0
	return flux

def d2dl_E(E,T_K,l0): # E [V/cm]
	E_dl = E * l0 * F / R / T_K
	return E_dl
def dl2d_E(E_dl,T_K,l0):
	E = E_dl * R * T_K / l0 / F
	return E

def d2dl_D(D,D0): # D [cm2/s]
	D_dl = D / D0 
	return D_dl
def dl2d_D(D_dl,D0):
	D = D_dl * D0 
	return D

def d2dl_c(c,c0): # c [mol/cm3]
	c_dl = c / c0 
	return c_dl
def dl2d_c(c_dl,c0):
	c = c_dl * c0 
	return c

def d2dl_I(I,D0,l0,c0): # I [A/cm2]
	I_dl = I * l0 / F / D0 / c0 
	return I_dl
def dl2d_I(I_dl,D0,l0,c0):
	I = I_dl * F * D0 * c0 / l0
	return I

def d2dl_imp(imp,T_K,D0,l0,c0): # impedance [ohm]
	imp_dl = imp * D0 * c0 * F**2 / l0 / R / T_K
	return imp_dl
def dl2d_imp(imp_dl,T_K,D0,l0,c0):
	imp = imp_dl * l0 * R * T_K / D0 / c0 / F**2
	return imp

def d2dl_t(t,D0,l0): # t [s]
	t_dl = t * D0 / l0**2
	return t_dl
def dl2d_t(t_dl,D0,l0):
	t = t_dl * l0**2 / D0
	return t

def d2dl_w(w,D0,l0): # frequency w [/s]
	w_dl = w * l0**2 / D0
	return w_dl
def dl2d_w(w_dl,D0,l0):
	w = w_dl * D0 / l0**2
	return w

def d2dl_phi(phi,T_K): # phi [V]
	phi_dl = phi * F / R / T_K
	return phi_dl
def dl2d_phi(phi_dl,T_K):
	phi = phi_dl * R * T_K / F
	return phi

def d2dl_q(q,l0,c0): # q [C/cm2]
	q_dl = q / c0 / l0 / F 
	return q_dl
def dl2d_q(q_dl,l0,c0):
	q = q_dl * c0 * l0 * F
	return q

def d2dl_l(l,l0): # l [cm]
	l_dl = l / l0
	return l_dl
def dl2d_l(l_dl,l0):
	l = l_dl * l0
	return l

def d2dl_eps(epsilon,T_K,l0,c0): # epsilon [F/cm]
	K_ = (epsilon*R*T_K)/(c0*F**2*l0**2)
	return K_
def dl2d_eps(K_,T_K,l0,c0):
	epsilon = K_ * (c0*F**2*l0**2) / (R*T_K)
	return epsilon

#def d2dl_k(k,T_K,D0,l0,c0): # rate constant (old definition)
#	k_dl = k * l0 * R * T_K / D0 / c0
#	return k_dl
#def dl2d_k(k_dl,T_K,D0,l0,c0):
#	k = k_dl * D0 * c0 / l0 / R / T_K
#	return k

def d2dl_k_inter(k,T_K,D0,l0,c0): # rate constant [mol/cm2/s] for interlayer reactions
	k_dl = k * l0 / D0 / c0
	return k_dl
def dl2d_k_inter(k_dl,T_K,D0,l0,c0):
	k = k_dl * D0 * c0 / l0
	return k

def d2dl_k_intra(k,T_K,D0,l0,c0): # rate constant [mol/cm3/s] for intralayer reactions
	k_dl = k * l0**2 / D0 / c0
	return k_dl
def dl2d_k_intra(k_dl,T_K,D0,l0,c0):
	k = k_dl * D0 * c0 / l0**2
	return k

def d2dl_dG(dG,T_K): # dG [kJ/mol]
	dG_dl = dG * 1000.0 / R / T_K
	return dG_dl
def dl2d_dG(dG_dl,T_K):
	dG = dG_dl * R * T_K / 1000.0
	return dG

def E_2phi_(E_,dx_b_):
	phi_ = [0.0]
	n = dx_b_.shape[0]
	for i in range(1,n-1):
		phi_.append(phi_[-1]-dx_b_[i]*(E_[i-1]+E_[i])/2.0)
	return np.array(phi_)

