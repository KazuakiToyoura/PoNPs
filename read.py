#!/usr/bin/env python
import numpy as np
np.set_printoptions(threshold=np.inf)

# Define the function of each input file one by one.

def read_comp(comp_file):
	data = np.loadtxt(comp_file,dtype="str",delimiter=",",comments="#")
	calcType, T_K, I, n_print, n_figs = data[0,:]
	iter_max, prec_c, prec_E, n_switch, prec_Volt = data[1,:]
	timeGrid, dt_max, dt_min, n_time, n_dt_max = data[2,:]
	delta_I, freqGrid, freq_max,freq_min,n_freqSteps = data[3,:]

	# Convert variable types.
	T_K = float(T_K)
	I = float(I)
	n_print = int(n_print)
	n_figs = int(n_figs)
	iter_max = int(iter_max)
	prec_c = float(prec_c)
	prec_E = float(prec_E)
	n_switch = int(n_switch)
	prec_Volt = float(prec_Volt)
	n_time = int(n_time)
	n_dt_max = int(n_dt_max)
	delta_I = float(delta_I)
	freq_max = float(freq_max)
	freq_min = float(freq_min)
	n_freqSteps = int(n_freqSteps)

	# Output input infomation
	################################
	print("-"*30)
	print("calcType:", calcType)
	print("T_K[K]:",T_K)
	print("I[A/cm2]:",I)
	print("n_print:",n_print)
	print("n_figs:",n_figs)
	print("iter_max:",iter_max)
	print("prec_c:",prec_c)
	print("prec_E:",prec_E)
	print("n_switch:",n_switch)
	print("prec_Volt[V]:",prec_Volt)
	print("mkTimeGrid:",timeGrid)
	print("dt_max[s]:",dt_max)
	print("dt_min[s]:",dt_min)
	print("n_timeSteps:",n_time)
	print("n_dt_max:",n_dt_max)
	print("delta_I[A/cm2]:",delta_I)
	print("mkFreqGrid",freqGrid)
	print("freq_max[/s]",freq_max)
	print("freq_min[/s]",freq_min)
	print("n_freqSteps",n_freqSteps)
	print("-"*30,"\n\n")
	################################
	return calcType, T_K, I, n_print, n_figs,\
			   iter_max, prec_c, prec_E, n_switch, prec_Volt,\
				 timeGrid, dt_max, dt_min, n_time, n_dt_max,\
				 delta_I, freqGrid, freq_max,freq_min,n_freqSteps


def read_defect(defect_file):
	defect_info = np.loadtxt(defect_file,dtype="str",delimiter=",",comments="#")
	defect_info = np.atleast_2d(defect_info)
	defect_name = defect_info[:,0] # defect name
	z = defect_info[:,1].astype(int) # valence
	cL = defect_info[:,2].astype(float) # concentrations in the reservoir on the left side [mol/cm3]
	cR = defect_info[:,3].astype(float) # concentrations in the reservoir on the right side [mol/cm3]
	cLB = defect_info[:,4].astype(float) # Lower bound for each defect concentration [mol/cm3]
	cUB = defect_info[:,5].astype(float) # Upper bound for each defect concentration [mol/cm3]
	D = defect_info[:,6:].astype(float) # diffusion constants at individual layers [cm2/s]
	n_def = z.shape[0]
	###################################
	print("-"*30)
	print("defect name:", defect_name)
	print("z:",z)
	print("cL[mol/cm3]:")
	print(cL)
	print("cR[mol/cm3]:")
	print(cR)
	print("D[cm2/s]:")
	print(D)
	print("-"*30,"\n\n")
	###################################
	return defect_name, z, cL, cR, cLB, cUB, D, n_def


def read_layer(layer_file):
	layer_info = np.loadtxt(layer_file,dtype="str",delimiter=",",comments="#")
	layer_info = np.atleast_2d(layer_info)
	layer_name = layer_info[:,0] # layer name
	relativePermittivity = layer_info[:,1].astype(float) # relativePermittivity at each layer
	d = layer_info[:,2].astype(float) # length of each layer
	spaceGrid = layer_info[:,3] # type of space grid
	n_elem = layer_info[:,4].astype(int) # number of elements for each layer
	ldebye = layer_info[:,5] # debye length at each layer (negative values are neglected)
	n_elem_ldebye = layer_info[:,6].astype(int) # number of elements in debye length at each layer
	n_ldebye = layer_info[:,7].astype(int) # number of debye length at each layer
	slowDefectID = layer_info[:,8].astype(int) # slow defect ID at each layer
	for i in range(n_elem.shape[0]): # Check and modify the number of emelents in each layer.
		if n_elem[i]%2 == 0:
			n_elem[i] = nelem[i]+1
			print("Warning: n_elem is changed into odd number!")
	# Make list of layer ID for each element (lid)
	lid = []
	for i in range(n_elem.shape[0]):
		lid.append(np.full(n_elem[i],i+1))
	#lid = np.array(lid)
	lid = np.concatenate(lid)
	lid = np.append(0,lid)
	lid = np.append(lid,len(n_elem)+1)

	###################################
	print("-"*30)
	print("layer name:", layer_name)
	print("relative permittivity:", relativePermittivity)
	print("layer length d[cm]:", d)
	print("mkSpaceGrid:", spaceGrid)
	print("n_elem:", n_elem)
	print("ldebye[cm]:", ldebye)
	print("n_elem_ldebye:", n_elem_ldebye)
	print("n_ldebye:", n_ldebye)
	print("slowDefectID:", slowDefectID)
	print("lid (layer IDs for all elements):")
	print(lid)
	print("-"*30,"\n\n")
	###################################
	return layer_name, relativePermittivity, d, spaceGrid, n_elem, ldebye, n_elem_ldebye, n_ldebye, slowDefectID, lid


def read_reaction(reaction_file,n_def):
	reaction_info = np.loadtxt(reaction_file,dtype="str",delimiter=",",comments="#")
	reaction_lids = reaction_info[:,0:2].astype(int) # reaction layer ids
	stoich_coef_L = reaction_info[:,2:2+n_def].astype(float) # change in defect  at each reaction in left part
	stoich_coef_R = reaction_info[:,2+n_def:2+2*n_def].astype(float) # relativePermittivity at each reaction
	k_const = reaction_info[:,2+2*n_def].astype(float) # kinetic constant [cm/s]
	dG0 = reaction_info[:,3+2*n_def].astype(float) # change in standard chemical poteitnal of each defect [Right-Left]
	###################################
	print("-"*30)
	print("reaction layer ids:")
	print(reaction_lids)
	print("stoich_coef_L:")
	print(stoich_coef_L)
	print("stoich_coef_R:")
	print(stoich_coef_R)
	print("k_const[mol/cm2/s or mol/cm3/s]:")
	print(k_const)
	print("deltaG0[kJ/mol]:")
	print(dG0)
	print("-"*30,"\n\n")
	###################################
	return reaction_lids,stoich_coef_L,stoich_coef_R,k_const,dG0


def read_conc(c_file,z,n_elem,cL,cR):
	ci = np.loadtxt(c_file,dtype="float",delimiter=",",comments="#")
	c_smooth = 'false'
	print("-"*30)
	if ci.shape[0] != z.shape[0] or ci.shape[1] != np.sum(n_elem):
		print("Warning!")
		print("conc.csv is not consistent with the n_elem in layer.csv.")
		print("They are interporated between cL and cR based on log_ci.")
		ci = np.zeros((z.shape[0],np.sum(n_elem)))
		c_smooth = 'true'
		for i in range(z.shape[0]):
			ci[i,:] = cL[i]
	else:
		print("The initial concentration data is read successfully.")
	print("-"*30,"\n\n")
	return ci, c_smooth


def read_efield(efield_file,n_elem):
	Ei = np.loadtxt(efield_file,dtype="float",delimiter=",",comments="#")
	print("-"*30)
	if Ei.shape[0] != np.sum(n_elem)+1:
		print("Warning!")
		print("efield.csv is not consistent with n_elem in layer.csv.")
		print("All values are set to zero.")
		Ei = np.zeros(np.sum(n_elem)+1)
	else:
		print("The initial electric field data is read successfully.")
	print("-"*30,"\n\n")
	return Ei


