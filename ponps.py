"""
The PNPS code is a Numerical simulator for the one-dimensional Poisson-Nernst-Planck (1D-PNP) systems. The 1D-PNP equations are numerically solved, to acquire the time dependences of charge-carrier concentration profiles and electric potential profile. A steady state (constant current), a transition process from a steady state to another, and an impedance spectrum can be treated. The diffusion coefficients of all charge carriers should be specified for each of multiple layers. In addition, the rate constants and standard Gibbs energy changes of inter-layer and intra-layer reactions should also be specified. Both ends of the given system contact reservoirs of charge carriers, where the concentration of each charge carrier is constant at a given value. The following paper was referred to write this code, in which several advanced features are newly added for enhanced functionality.

Timothy R. Brumleve and Richard P. Buck, Numerical solution of the Nernst-Planck and Poisson equation system with applications to membrane electrochemistry and solid state physics, Journal of Electroanalytical Chemistry and Interfacial Electrochemistry 90, 1-31 (1978).

The PNPS code was written by Kazuaki Toyoura in Nov. 2021. Dr. Katsuhiro Ueno has great contribution to the code development with fruitful discussions and many bug reports.

"""

# Import modules (Other modules are imported afterwards.)
import argparse
import datetime
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
	# Start main program.
	########################################
	print("#"*30)
	date = datetime.datetime.now()
	print(date)
	print("Start progmram!")
	print("#"*30, "\n")
	########################################

	# Parse arguments
	parser = argparse.ArgumentParser()
	#parser.add_argument("--omp", type=int, default=1, \
	#		               help="OMP_NUM_THREADS")
	parser.add_argument("--comp", type=str, default='comp.csv', \
			               help="Computational conditions in the csv format")
	parser.add_argument("--defect", type=str, default='defect.csv', \
			               help="Information of each defect in the csv format")
	parser.add_argument("--layer", type=str, default='layer.csv', \
			               help="Information of each layer in the csv format")
	parser.add_argument("--reaction", type=str, default='reaction.csv', \
			               help="Information of each layer in the csv format")
	parser.add_argument("--conc", type=str, default='conc.csv', \
			               help="Initial concentration data of each defect in the csv format")
	parser.add_argument("--efield", type=str, default='efield.csv', \
			               help="Initial electric field data in the csv format")
	args = parser.parse_args()


	# Import other modules
	import os
	#n_omp=args.omp
	#os.environ["OMP_NUM_THREADS"] = str(n_omp)
	os.environ["OMP_NUM_THREADS"] = "1"
	import numpy as np
	np.set_printoptions(threshold=np.inf)
	from copy import deepcopy
	
	# Import defined functions in separated files
	import read
	import write
	import converter as cvt
	import mkGrid as mkgr
	import updater as upd
	import ploter as pltr
	import mkImpedance as mkimp

	# Read input data
	print("Read comp.csv")
	comp_file = args.comp
	calcType, T_K, I, n_print, n_figs,\
	iter_max, prec_c, prec_E, n_switch, prec_Volt,\
	timeGrid, dt_max, dt_min, n_time, n_dt_max,\
	delta_I, freqGrid, freq_max,freq_min,n_freqSteps\
	= read.read_comp(comp_file)

	print("Read defect.csv")
	defect_file = args.defect
	defect_name, z, cL, cR, cLB, cUB, D, n_def\
	= read.read_defect(defect_file)

	print("Read layer.csv")
	layer_file = args.layer
	layer_name, epsr, d, spaceGrid, n_elem,\
	ldebye_str, n_elem_ldebye, n_ldebye, slowDefectID, lid\
	= read.read_layer(layer_file)
	# Convert relative permittivity to absolute permittivity [F/cm]
	epsilon = epsr * 8.854187812813e-14 # 8.854187812813e-12 * 1e-2
	
	print("Read reaction.csv")
	reaction_file = args.reaction
	react_lids,stoich_coef_L,stoich_coef_R,k_const,dG0\
	= read.read_reaction(reaction_file,n_def)

	print("Read conc.csv")
	c_file = args.conc
	ci, c_smooth = read.read_conc(c_file,z,n_elem,cL,cR)
	ci = np.concatenate([cL.reshape([cL.shape[0],1]),ci,cR.reshape([cR.shape[0],1])],1)

	print("Read efield.csv")
	efield_file = args.efield
	Ei = read.read_efield(efield_file,n_elem)


	# Estimate debye length in each layer i (ldebye)
	print("Estimate debye length in each layer.")
	ldebye = mkgr.mkLdebye(ldebye_str,epsilon,T_K,z,ci,n_elem)


	# Specify dimensionless constants
	print("Specify dimensionless constants")
	print("-"*30)
	l0 = np.min(ldebye) #[cm]
	c0 = 1.0 #[mol/cm3]
	D0 = np.min((l0**2/d[:]**2)*D[slowDefectID-1,:]) #[cm2/s]
	print("l0[cm]:", l0)
	print("c0[mol/cm3]:", c0)
	print("D0[cm2/s]:", D0)
	print("-"*30,"\n\n")


	# Convert into dimensionless quantities (denoted by "_")
	cL_ = cvt.d2dl_c(cL,c0)
	cR_ = cvt.d2dl_c(cR,c0)
	cLB_ = cvt.d2dl_c(cLB,c0)
	cUB_ = cvt.d2dl_c(cUB,c0)
	ci_ = cvt.d2dl_c(ci,c0)
	Ei_ = cvt.d2dl_E(Ei,T_K,l0)
	prec_Volt_ = cvt.d2dl_phi(prec_Volt,T_K)
	D_ = cvt.d2dl_D(D,D0)
	d_ = cvt.d2dl_l(d,l0)
	ldebye_ = cvt.d2dl_l(ldebye,l0)
	dG0_ = cvt.d2dl_dG(dG0,T_K)
	I_ = cvt.d2dl_I(I,D0,l0,c0)
	delta_I_ = cvt.d2dl_I(delta_I,D0,l0,c0)
	K_ = cvt.d2dl_eps(epsilon,T_K,l0,c0)
	k_const_ = cvt.d2dl_k_inter(k_const,T_K,D0,l0,c0)
	# The units are different between intralayer & interlayer reactions.
	# mol/cm3/s vs. mol/cm2/s
	for i in range(react_lids.shape[0]):
		if react_lids[i,0] == react_lids[i,1]:
			k_const_[i] = cvt.d2dl_k_intra(k_const[i],T_K,D0,l0,c0)
	
	
	# Make K_ for each boudnary between elements.
	K_B_ = np.zeros(np.sum(n_elem)+1)
	K_B_[0] = K_[0]
	K_B_[-1] = K_[-1]
	#print(lid.shape)
	#print(lid)
	for j in range(1,np.sum(n_elem)):
		#print(j,lid[j],lid[j+1])
		K_B_[j] = (K_[lid[j]-1]+K_[lid[j+1]-1])/2.0


	# Reconstruct reaction information by dictionary.
	reacts = []
	for i in range(n_def):
		reacts.append({})
		n=0
		for j in range(react_lids.shape[0]):
			if np.abs(stoich_coef_L[j,i]) > 0.01\
			or np.abs(stoich_coef_R[j,i]) > 0.01:
				reacts[-1][tuple(np.append(react_lids[j],n))] \
				= {'stoich_L':stoich_coef_L[j,:],'stoich_R':stoich_coef_R[j,:],\
				   'k_const_':k_const_[j],'dG0_':dG0_[j]}
				n=n+1


	# Make space grid.
	print("Make space grid")
	dx_b_,dx_m_ = mkgr.mkGrid_space(spaceGrid,n_elem,ldebye_,n_elem_ldebye,\
			                            n_ldebye,d_,K_,ci_,z,l0)
	
	# Make time grid.
	print("Make time grid")
	dt_ = mkgr.mkGrid_time(timeGrid,n_dt_max,dt_max,dt_min,n_time,\
			                   slowDefectID,d_,D_,K_,z,ci_,lid,D0,l0)

	
	# When conc.csv is insufficient, ci_ are generated by linearly interpolating log_ci
	if c_smooth == 'true':
		print("Interpolate defect concentrations between cL and cR.")
		print("-"*30)
		print("Initial concentration profiles are created automatically.")
		x=np.cumsum(dx_m_)/np.sum(dx_m_)
		x=np.append(0.0,x)
		for i in range(n_def):
			ci_[i,1:-1] = ci_[i,0]*(ci_[i,-1]/ci_[i,0])**x[1:-1]
		print("ci[mol/cm3]")
		print(cvt.dl2d_c(ci_,c0))
		print("-"*30,"\n\n")


	# Start iteration (time step)
	########################################
	print("#"*30)
	date = datetime.datetime.now()
	print(date)
	print("Start main loop")
	print("#"*30, "\n")
	########################################
	
	# All the defect concentrations, electric fields, and electric potentials are
	# stored in the lists of data_c_, data_E_, and data_phi_, respectively.
	# Note that these values are dimensionless.
	data_c_ = [ci_]
	data_E_ = [Ei_]
	data_phi_ = [cvt.E_2phi_(Ei_,dx_b_)]
	data_fMat_ = [np.zeros((n_def,np.sum(n_elem)+1))]

	# Setting a constant current.
	Iconst_ = I_
	if calcType.lower() == 'impedance':
		Iconst_ = I_ + delta_I_

	# Plot the initial concentrations and electric potential.
	pltr.plot(cvt.dl2d_l(dx_m_,l0),cvt.dl2d_c(ci_,c0),cvt.dl2d_l(dx_b_,l0),cvt.dl2d_phi(data_phi_[0],T_K),n_elem,n_elem_ldebye,n_ldebye,'0',defect_name,0.0)
	
	# Start iterations
	for n in range(dt_.shape[0]):
		# Convert dt and t [s].
		dt = cvt.dl2d_t(dt_[n],D0,l0)
		t = cvt.dl2d_t(np.sum(dt_[:n+1]),D0,l0)
		print("time step: {:<5d} (t[s]: {:.5e}, dt[s]: {:.5e})".format(n+1,t,dt))

		# update c_, E_, and phi_  by Newton-Raphson method
		c_,E_,phi_,fLMat_,fRMat_,status = upd.update(data_c_[n],data_E_[n],data_phi_[n],\
				                                         dx_m_,dx_b_,dt_[n],D_,z,reacts,Iconst_,\
																								 K_B_,iter_max,prec_c,prec_E,n_switch,\
																								 prec_Volt_,n_elem,n_def,lid,T_K,n,\
																								 cLB_,cUB_,D0,l0,c0)
		data_c_.append(c_)
		data_E_.append(E_)
		data_phi_.append(phi_)
		#If fL_ != fR_, the value with greater magnitude is selected as f_.
		fMat_ = np.where(np.abs(fLMat_)>=np.abs(fRMat_),fLMat_,fRMat_)
		data_fMat_.append(fMat_)
		
		# Print total differences in c, E, and Voltage
		dc_max = cvt.dl2d_c(np.max(np.abs(data_c_[-1]-data_c_[-2])),c0)
		dE_max = cvt.dl2d_E(np.max(np.abs(data_E_[-1]-data_E_[-2])),T_K,l0)
		voltage = cvt.dl2d_phi(data_phi_[-1][-1],T_K)
		dVolt = cvt.dl2d_phi(data_phi_[-1][-1]-data_phi_[-2][-1],T_K)
		print(" "*27,"-"*40)
		print(" "*27,"  dc_max[mol/cm3] = {: .10e}".format(dc_max))
		print(" "*27,"     dE_max[V/cm] = {: .10e}".format(dE_max))
		print(" "*27,"         dVolt[V] = {: .10e}".format(dVolt))
		print(" "*27,"       Voltage[V] = {: .10e}".format(voltage))
		print(" "*27,"-"*40,"\n")

		# Plot profile figures.
		if (n+1)%(int(dt_.shape[0]/n_figs)) == 0 or n+1 == dt_.shape[0]:
			pltr.plot(cvt.dl2d_l(dx_m_,l0),cvt.dl2d_c(c_,c0),cvt.dl2d_l(dx_b_,l0),cvt.dl2d_phi(phi_,T_K),n_elem,n_elem_ldebye,n_ldebye,str(n+1),defect_name,t)

		# Break when update is failure (status == 1)
		if status == 1:
			break

	########################################
	print("#"*30)
	date = datetime.datetime.now()
	print(date)
	if status == 0:
		print("Finish iterations successfuly")
	elif status == 1:
		print("Abort iterations")
	print("#"*30, "\n")
	########################################
	
	
	# Output several files
	print("Output files")
	write.output(data_c_,data_E_,data_phi_,data_fMat_,dx_m_,dx_b_,dt_,T_K,z,D0,l0,c0,n_print)
	
	
	# Calculate impedance spectra (calcType = 'impedance' only).
	if status == 0 and calcType.lower() == 'impedance':
		########################################
		print("#"*30)
		date = datetime.datetime.now()
		print(date)
		print("Calculate impedance spectra")
		print("#"*30, "\n")
		########################################
		w_,w,freq = mkgr.mkGrid_freq(freqGrid,freq_max,freq_min,n_freqSteps,D0,l0)
		z_re_, z_im_ = mkimp.calcImp(w_,dt_,delta_I_,np.array(data_phi_)[:,-1])
		pltr.imp_plot(cvt.dl2d_imp(z_re_,T_K,D0,l0,c0),cvt.dl2d_imp(z_im_,T_K,D0,l0,c0),freq)
		write.output_imp(freq,z_re_,z_im_,T_K,D0,l0,c0)
	

	########################################
	print("#"*30)
	date = datetime.datetime.now()
	print(date)
	if status == 0:
		print("Finish pogram!")
	elif status == 1:
		print("Abort program!")
	print("#"*30)
	########################################


