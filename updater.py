#!/usr/bin/env python
import numpy as np
import scipy.linalg as linalg
from copy import deepcopy
import datetime

import mkNRMat as mknr
import converter as cvt

np.set_printoptions(threshold=np.inf)
# Define function to estimate the maximum change ratios for c_ and E_.
def converge_cE(c_tmp_,delta_c_,E_tmp_,delta_E_):
	# Estimate the maximum change ratio of concentration for nonzero values
	neg_c_ = 0.0
	nonzero_c = np.where(np.abs(c_tmp_)>np.abs(neg_c_))
	if len(nonzero_c) == 0:
		diff_c = 0.0
	else:
		diff_c = np.max(np.abs(delta_c_[nonzero_c]/c_tmp_[nonzero_c]))
	# Estimate the maximum change ratio of electric field for nonzero values
	neg_E_ = 0.0
	nonzero_E = np.where(np.abs(E_tmp_)>np.abs(neg_E_))[0]
	if len(nonzero_E) == 0:
		diff_E = 0.0
	else:
		diff_E=np.max(np.abs(delta_E_[nonzero_E]/E_tmp_[nonzero_E]))
	return diff_c,diff_E


# Define function for aborting program.
#def abort_potPro(c_tmp_,E_tmp_,c0,T_K,l0):
def abort_comment():
	print("  Check conc_fin.csv and efield_fin.csv.")
	print("  If defect concentrations in bulk region reach the lower bound, Space grid may be rough.")
	print("  Increase n_elements for making a finer space grid.")
	print("  If defect concentrations near interface reach the lower or upper bound,")
	print("  reaction rates may be too fast. Moderate the reactions in reaction.csv.")
	print("  If your simulation is aborted at a long time interval, Time grid may be rough.")
	print("  Decrease dt_max and increase n_dt_max.")
	print("  If not improved by combining the above changes, reconsider the initial conditions,")
	print("  boundary conditions, and/or chemical reaction parameters.")
	print("  Avoid abrupt change spatially and temporally in defect concentrations & electric field.")


# Updater for c_ and E_ based on the Newton-Raphson method
def update(c_,E_,phi_,dx_m_,dx_b_,dt_,D_,z,reacts,I_,K_B_,iter_max,\
		       prec_c,prec_E,n_switch,prec_Volt_,n_elem,n_def,lid,T_K,n,cLB_,cUB_,D0,l0,c0):
	# Define numpy arrays
	c0_ = deepcopy(c_) # c_ at the previous time step 
	E0_ = deepcopy(E_) # E_ at the previous time step
	c_tmp_ = deepcopy(c_) # c_ at the current NR step
	E_tmp_ = deepcopy(E_) # E_ at the current NR step
	voltage_prev_ =  phi_[-1] # voltage at the previous NR step.
	
	# Newton-Raphson iterations
	print("  NR_step     maxChgRate_c        maxChgRate_E        diff_Volt[V]")
	status=0 # 0 and 1 mean NR success and failure, respectively.
	for i in range(iter_max):
		# Make NR matrix and vector.
		NRMat,NRVec,fLMat_,fRMat_ = mknr.mkNRMat(c_tmp_,c0_,E_tmp_,E0_,dx_m_,dx_b_,dt_,D_,z,reacts,I_,K_B_,n_def,n_elem,lid)
		
		# Check nan and inf in NRMat and NRvector
		chk1 = np.isinf(NRMat)
		chk2 = np.isnan(NRMat)
		chk3 = np.isinf(NRVec)
		chk4 = np.isnan(NRVec)
		if np.any(chk1==True) or np.any(chk2==True) or \
			 np.any(chk3==True) or np.any(chk4==True):
			status = 1
			print("Warning!")
			print("-"*68)
			print("  Newton-Raphson matrix and/or vector include nan or inf.")
			abort_comment()
			print("-"*68, "\n")
			break

		# Estimate the delta_c_ and delta_E_ by LU decompositon
		LU = linalg.lu_factor(NRMat)
		delta_cE_ = linalg.lu_solve(LU, NRVec)
		delta_c_ = np.reshape(delta_cE_[:c_.shape[0]*(c_.shape[1]-2)],[c_.shape[0],c_.shape[1]-2])
		delta_E_ = delta_cE_[c_.shape[0]*(c_.shape[1]-2):]
		
		#cL and cR are fixed, i.e., delta_c_[:,0] and delta_c_[:,-1] are zero.
		delta_c_ = np.hstack([np.zeros((n_def,1)),delta_c_,np.zeros((n_def,1))])
		c_tmp1_ = deepcopy(c_tmp_)
		c_tmp_ = c_tmp_ + delta_c_
		E_tmp_ = E_tmp_ + delta_E_

		# Modify the concentrations out of the specified ranges in defect.csv.
		chk_LB = 0
		chk_UB = 0
		for j in range(n_def):
			if np.any(c_tmp_[j,:] < cLB_[j]): 
				chk_LB = chk_LB + 1
				c_tmp_[j,:] = np.where(c_tmp_[j,:] < cLB_[j], cLB_[j], c_tmp_[j,:])
			if np.any(c_tmp_[j,:] > cUB_[j]):
				chk_UB = chk_UB + 1
				c_tmp_[j,:] = np.where(c_tmp_[j,:] > cUB_[j], cUB_[j], c_tmp_[j,:])
		delta_c_ = c_tmp_ - c_tmp1_
		if chk_LB >= 1:
			print("Warning: Some concentrations reach the lower bound.")
		if chk_UB >= 1:
			print("Warning: Some concentrations reach the upper bound.")

		# Estimate maximum change ratio of c_ and E_ at the current NR step
		diff_c,diff_E = converge_cE(c_tmp_,delta_c_,E_tmp_,delta_E_)
		phi_tmp_ = cvt.E_2phi_(E_tmp_,dx_b_)
		voltage_tmp_ = phi_tmp_[-1]
		diff_V_ = voltage_tmp_-voltage_prev_
		print("{:>6d}     {: .10e}   {: .10e}   {: .10e}".format(i+1,diff_c,diff_E,cvt.dl2d_phi(diff_V_,T_K)))
		voltage_prev_ = voltage_tmp_

		# Check convergence.
		if diff_c < prec_c and diff_E < prec_E:
			break
		elif i+1 >= n_switch and np.abs(diff_V_) < prec_Volt_:
			break
		if i == iter_max-1:
			status = 1
			print("Warning!")
			print("-"*68)
			print("  Newton-Raphson does not converge.")
			abort_comment()
			print("-"*68, "\n")
			break

		# Check NR process stability
		chk5 = np.isinf(diff_c)
		chk6 = np.isnan(diff_c)
		chk7 = np.isinf(diff_E)
		chk8 = np.isnan(diff_E)
		if np.any(chk5==True) or np.any(chk6==True) or \
			 np.any(chk7==True) or np.any(chk8==True):
			status = 1
			print("Warning!")
			print("-"*68)
			print("  Newton-Raphson process is unstable.")
			abort_comment()
			print("-"*68, "\n")
			break

	return c_tmp_, E_tmp_, phi_tmp_,fLMat_,fRMat_,status

