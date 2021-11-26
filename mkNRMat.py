#!/usr/bin/env python
import numpy as np

np.set_printoptions(threshold=np.inf)
#Define functions for creating the Newton-Raphson matrix and vector.

# Change in Gibbs energy for a given intralayer reaction
def dG_homo(j,c_,stoich_coef,dG0_):
	id_nonzero = np.where(np.abs(stoich_coef)>1.0e-10)[0]
	dG_ = dG0_ + np.dot(stoich_coef[id_nonzero],np.log(c_[id_nonzero,j]))
	return dG_




# Change in Gibbs energy for a given interlayer reaction
# L and R mean left and right sides of the boudnary, respectively.
def dG_hetero(j,c_,E_,z,stoich_coef_L,stoich_coef_R,dG0_,dx_m_):
	id_nonzero_L = np.where(np.abs(stoich_coef_L)>1.0e-10)[0]
	id_nonzero_R = np.where(np.abs(stoich_coef_R)>1.0e-10)[0]
	dG_ = dG0_ + np.dot(stoich_coef_L[id_nonzero_L],np.log(c_[id_nonzero_L,j]))\
			       + np.dot(stoich_coef_R[id_nonzero_R],np.log(c_[id_nonzero_R,j+1]))\
						 + dx_m_[j]*E_[j]*np.dot(stoich_coef_L,z)
	return dG_




# Flow of defect i on emelent boundary j
# Flows on the left and right sides of the boundary (fL, fR) can be
# different in the case of interlayer reactions.
# For example, Had(electrode) --> H+(electrolyte) + e-(electrode)
def flow(i,j,c_,E_,D_,z,reacts,dx_m_,lid):
	if lid[j] == lid[j+1]: # in the case of defect flow in the single layer, fL = fR.
		fL = - D_[i,lid[j]-1] * ( (c_[i,j+1]-c_[i,j])/dx_m_[j] - z[i]*(c_[i,j+1]+c_[i,j])*E_[j]/2.0 )
		fR = fL
	else: # in the case of defect flow between the different layers, fL != fR.
		fL = 0.0
		fR = 0.0
		for key in reacts[i].keys(): # loop for reactions involving defect i
			if key[:2] == (lid[j],lid[j+1]): # if the reaction corresponds to the interlayer between lid[j] and lid[j+1]
				# Add the flow by the reaction
				dG_ = dG_hetero(j,c_,E_,z,reacts[i][key]['stoich_L'],reacts[i][key]['stoich_R'],reacts[i][key]['dG0_'],dx_m_)
				fL = fL + reacts[i][key]['stoich_L'][i]*reacts[i][key]['k_const_']*dG_
				fR = fR - reacts[i][key]['stoich_R'][i]*reacts[i][key]['k_const_']*dG_
	return fL, fR




# Differentiation of fij against ckl
# fij: flow of defect i on emelent boundary j
# ckl: concentration of defect k on emelent l
def dfij_dckl(i,j,k,l,c_,E_,D_,z,reacts,dx_m_,lid):
	dfL_dc = 0.0
	dfR_dc = 0.0
	if lid[j] == lid[j+1]: # in the case of intralayer flow
		# nonzero only when k = i and l = j, j+1
		if k == i:
			if l == j:
				dfL_dc = -D_[i,lid[j]-1]*(-1.0/dx_m_[j]-z[i]*E_[j]/2.0)
				dfR_dc = dfL_dc
			elif l == j+1:
				dfL_dc = -D_[i,lid[j]-1]*(1.0/dx_m_[j]-z[i]*E_[j]/2.0)
				dfR_dc = dfL_dc
	else: # int the case of interlayer flow
		for key in reacts[i].keys():
			if key[:2] == (lid[j],lid[j+1]):
				if np.abs(reacts[i][key]['stoich_L'][i]) > 1.0e-5:
					if l == j and np.abs(reacts[i][key]['stoich_L'][k]) > 1.0e-5:
						dfL_dc = dfL_dc + reacts[i][key]['stoich_L'][i] * reacts[i][key]['k_const_'] * reacts[i][key]['stoich_L'][k]/c_[k,l]
					elif l == j+1 and np.abs(reacts[i][key]['stoich_R'][k]) > 1.0e-5:
						dfL_dc = dfL_dc + reacts[i][key]['stoich_L'][i] * reacts[i][key]['k_const_'] * reacts[i][key]['stoich_R'][k]/c_[k,l]
				if np.abs(reacts[i][key]['stoich_R'][i]) > 1.0e-5:
					if l == j and np.abs(reacts[i][key]['stoich_L'][k]) > 1.0e-5:
						dfR_dc = dfR_dc - reacts[i][key]['stoich_R'][i] * reacts[i][key]['k_const_'] * reacts[i][key]['stoich_L'][k]/c_[k,l]
					elif l == j+1 and np.abs(reacts[i][key]['stoich_R'][k]) > 1.0e-5:
						dfR_dc = dfR_dc - reacts[i][key]['stoich_R'][i] * reacts[i][key]['k_const_'] * reacts[i][key]['stoich_R'][k]/c_[k,l]
	return dfL_dc, dfR_dc




# Differentiation of fij against El
# fij: flow of defect i on emelent boundary j
# El: electric field on emelent boundary l
def dfij_dEl(i,j,l,c_,D_,z,reacts,dx_m_,lid):
	# nonzero only when l = j
	dfL_dE = 0.0
	dfR_dE = 0.0
	if l == j:
		if lid[j] == lid[j+1]:
			dfL_dE = D_[i,lid[j]-1]*z[i]*(c_[i,j+1]+c_[i,j])/2.0
			dfR_dE = dfL_dE
		else:
			for key in reacts[i].keys():
				if key[:2] == (lid[j],lid[j+1]):
					if np.abs(reacts[i][key]['stoich_L'][i]) > 1.0e-5:
						dfL_dE = dfL_dE + reacts[i][key]['stoich_L'][i] * reacts[i][key]['k_const_'] * np.dot(reacts[i][key]['stoich_L'],z) * dx_m_[j]
					if np.abs(reacts[i][key]['stoich_R'][i]) > 1.0e-5:
						dfR_dE = dfR_dE - reacts[i][key]['stoich_R'][i] * reacts[i][key]['k_const_'] * np.dot(reacts[i][key]['stoich_L'],z) * dx_m_[j]
	return dfL_dE, dfR_dE




def mkNRMat(c_,c0_,E_,E0_,dx_m_,dx_b_,dt_,D_,z,reacts,I_,K_B_,n_def,n_elem,lid):
	n_elem_sum = np.sum(n_elem)
	# Make fLMat_ and fRMat_ shape:(n_def,n_elem_sum+1)
	fLMat_ = np.zeros((n_def,n_elem_sum+1))
	fRMat_ = np.zeros((n_def,n_elem_sum+1))
	for i in range(n_def): # Loop for defect species i
		for j in range(n_elem_sum+1): # Loop for each boundary j
			fLMat_[i,j],fRMat_[i,j]=flow(i,j,c_,E_,D_,z,reacts,dx_m_,lid)

	# Make NR matrix and vector
	dim = n_elem_sum*(n_def+1)+1
	NRMat = np.zeros((dim,dim))
	NRVec = np.zeros(dim)
	m=0
	for i in range(n_def): # Loop for defect species i
		for j in range(1,n_elem_sum+1): # Loop for each element j
			#NRVecij
			Fij = c_[i,j]-c0_[i,j] + dt_/dx_b_[j]*(fLMat_[i,j]-fRMat_[i,j-1])
			for key in reacts[i].keys():
				if key[:2] == (lid[j],lid[j]):
					dG_ = dG_homo(j,c_,reacts[i][key]['stoich_L'],reacts[i][key]['dG0_'])
					Fij = Fij + dt_*reacts[i][key]['stoich_L'][i]*reacts[i][key]['k_const_']*dG_
			NRVec[m] = -Fij
			#print(i,j,Fij)
			
			# for Fij vs. ckl
			for k in range(n_def): 
				for l in [j-1,j,j+1]:
					if l == 0 or l == n_elem_sum+1:
						continue
					n=k*n_elem_sum+l-1
					if k == i and l == j:
						NRMat[m,n] = 1.0
					if l == j:
						for key in reacts[i].keys():
							if key[:2] == (lid[j],lid[j]) and np.abs(reacts[i][key]['stoich_L'][i]*reacts[i][key]['stoich_L'][k]) > 1.0e-5:
								NRMat[m,n] = NRMat[m,n] + dt_ * reacts[i][key]['stoich_L'][i] * reacts[i][key]['k_const_'] * reacts[i][key]['stoich_L'][k]/c_[k,l]
					fLij_ckl, fRij_ckl = dfij_dckl(i,j,k,l,c_,E_,D_,z,reacts,dx_m_,lid)
					fLij_1_ckl, fRij_1_ckl = dfij_dckl(i,j-1,k,l,c_,E_,D_,z,reacts,dx_m_,lid)
					NRMat[m,n] = NRMat[m,n] + dt_/dx_b_[j]*(fLij_ckl-fRij_1_ckl)

			# for Fij vs. El
			for l in [j-1,j]:
				n=n_def*n_elem_sum+l
				fLij_El, fRij_El = dfij_dEl(i,j,l,c_,D_,z,reacts,dx_m_,lid)
				fLij_1_El, fRij_1_El = dfij_dEl(i,j-1,l,c_,D_,z,reacts,dx_m_,lid)
				NRMat[m,n] = dt_/dx_b_[j] * (fLij_El-fRij_1_El)
			
			m=m+1

	for j in range(0,n_elem_sum+1):
		#NRVecj
		Gj = K_B_[j]*(E_[j]-E0_[j])-I_*dt_+ np.dot(fLMat_[:,j],z)*dt_
		NRVec[m] = -Gj

		#for Gj vs. ckl
		for k in range(n_def):
			for l in [j,j+1]:
				if l == 0 or l == n_elem_sum+1:
					continue
				n=k*n_elem_sum+l-1
				for i in range(n_def):
					fLij_ckl, tmp = dfij_dckl(i,j,k,l,c_,E_,D_,z,reacts,dx_m_,lid)
					NRMat[m,n] = NRMat[m,n] + dt_*z[i]*fLij_ckl

		#for Gj vs. El
		l = j
		n=n_def*n_elem_sum+l
		NRMat[m,n] = K_B_[j]
		for i in range(n_def):
			fLij_El, tmp = dfij_dEl(i,j,l,c_,D_,z,reacts,dx_m_,lid)
			NRMat[m,n] = NRMat[m,n] + dt_*z[i]*fLij_El

		m=m+1
	
	return NRMat,NRVec,fLMat_,fRMat_




