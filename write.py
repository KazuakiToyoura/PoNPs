import numpy as np
np.set_printoptions(threshold=np.inf)	
import converter as cvt


def output(data_c_,data_E_,data_phi_,data_fMat_,dx_m_,dx_b_,dt_,T_K,z,D0,l0,c0,n_print,hetero_ids):
	print("-"*30)
	# Convert into quantities with original dimension.
	data_c = cvt.dl2d_c(np.array(data_c_),c0)
	data_E = cvt.dl2d_E(np.array(data_E_),T_K,l0)
	data_phi = cvt.dl2d_phi(np.array(data_phi_),T_K)
	x_m = np.append(0.0,np.cumsum(cvt.dl2d_l(dx_m_,l0)))
	x_m_2d = x_m.reshape(1,x_m.shape[0])
	x_b = np.append(0.0,np.cumsum(cvt.dl2d_l(dx_b_[1:-1],l0)))
	x_b_hetero = np.array([0.0,0.0])
	for i in range(1,hetero_ids.shape[0]):
		x_b_hetero = np.append(x_b_hetero,x_b[hetero_ids[i-1]+1:hetero_ids[i]+1])
		x_b_hetero = np.append(x_b_hetero,x_b[hetero_ids[i]])
	x_b_2d = x_b.reshape(1,x_b.shape[0])
	x_b_hetero_2d = x_b_hetero.reshape(1,x_b_hetero.shape[0])
	t = np.append(0.0,np.cumsum(cvt.dl2d_t(dt_,D0,l0)))
	t_2d = t.reshape(t.shape[0],1)
	voltage_2d = data_phi[:,-1].reshape(data_phi.shape[0],1)
	n_def = data_c.shape[1]

	# Convert fMat_ into IMat[A/cm2]
	F = 96485.3329 # [C/mol]
	data_IMat = cvt.dl2d_flux(np.array(data_fMat_),D0,l0,c0)
	for i in range(n_def):
		data_IMat[:,i,:] = z[i]*F*data_IMat[:,i,:]

	
	# Save the files in the final state
	# Save the final electric field in efield_final.csv.
	print("Output efield_fin.csv")
	np.savetxt('efield_fin.csv',data_E[-1,:].reshape(1,data_E.shape[1]),fmt='%.10e',delimiter=',',header='Electric field in each element [V/cm]')

	# Save the final concentration of each defect in conc_final_num.csv.
	print("Output conc_fin.csv")
	np.savetxt('conc_fin.csv',data_c[-1,:,1:-1],fmt='%.10e',delimiter=',',header='Concentration of each defect in each element [mol/cm3].cL & cR are not included.')

	
	# Same the profiles with decimation filter
	# Determine time step IDs for printing.
	n_tstep = data_phi.shape[0]
	printID = []
	if n_print == 0:
		date = datetime.datetime.now()
		n_print = n_tstep-1
	if n_print > 0:
		for i in range(n_tstep):
			if i%np.int((n_tstep-1)/n_print) == 0 or i == n_tstep-1:
				printID.append(i)
	elif n_print < 0:
		# step ids with large difference in voltage are sampled.
		n_print = -n_print
		diff_V = np.abs(np.diff(data_phi[:,-1],n=1))
		diff_V_LB = np.max(diff_V)/((n_tstep-1)/n_print)
		diff_V = np.where(diff_V<diff_V_LB,diff_V_LB,diff_V)
		diff_V = np.append(0.0,diff_V)
		Vint = (np.floor(np.cumsum(diff_V)/np.sum(diff_V)*n_print)).astype(int)
		V_uniq, indices = np.unique(Vint,return_index=True)
		printID = indices.tolist()
		if not n_tstep-1 in printID:
			printID.append(n_tstep-1)

	# Save the voltage profile between the two ends in voltage_profile_rough.csv.
	print("Output voltage_profile.csv")
	voltage_profile = np.concatenate([t_2d[printID,:],voltage_2d[printID,:]],axis=1)
	np.savetxt('voltage_profile.csv',voltage_profile,fmt='%.10e',delimiter=',',header='time[s],voltage[V]')
	
	# Save the electric potential profile in phi_profile.csv.
	print("Output phi_profile.csv")
	t_2d_v = np.concatenate([np.array([[np.nan]]),t_2d[printID,:]])
	xb_phi = np.concatenate([x_b_2d,data_phi[printID,:]])
	phi_profile = np.concatenate([t_2d_v,xb_phi],axis=1)
	np.savetxt('phi_profile.csv',phi_profile,fmt='%.10e',delimiter=',',header='phi(time[s],x[cm]) [V]')

	# Save the electric field profile in efield_profile.csv.
	print("Output efield_profile.csv")
	xb_efield = np.concatenate([x_b_2d,data_E[printID,:]],axis=0)
	efield_profile = np.concatenate([t_2d_v,xb_efield],axis=1)
	np.savetxt('efield_profile.csv',efield_profile,fmt='%.10e',delimiter=',',header='efield(time[s],x[cm]) [V/cm]')
	
	# Save the concentration profile in conc*_profile.csv for each defect.
	print("Output conc_profile.csv")
	for i in range(n_def):
		xm_conc_tmp = np.concatenate([x_m_2d[:,1:-1],data_c[printID,i,1:-1]])
		conc_profile_tmp = np.concatenate([t_2d_v,xm_conc_tmp],axis=1)
		np.savetxt('conc'+str(i+1)+'_profile.csv',conc_profile_tmp,fmt='%.10e',delimiter=',',header='concentration(time[s],x[cm]) [mol/cm3]. cL & cR are not included.')

	# Save the partial current profile in partI*_profile.csv for each defect.
	print("Output partialI_profile.csv")
	for i in range(n_def):
		xb_Iij_tmp = np.concatenate([x_b_hetero_2d,data_IMat[printID,i,:]])
		Iij_profile_tmp = np.concatenate([t_2d_v,xb_Iij_tmp],axis=1)
		np.savetxt('partialI'+str(i+1)+'_profile.csv',Iij_profile_tmp,fmt='%.10e',delimiter=',',header='partialI(time[s],x[cm]) [A/cm2]')
	
	print("-"*30,"\n\n")




def output_imp(freq,z_re_,z_im_,T_K,D0,l0,c0):
	# Convert into quantities with original dimension.
	freq_2d = freq.reshape(freq.shape[0],1)
	z_re = cvt.dl2d_imp(z_re_,T_K,D0,l0,c0)
	z_re_2d = z_re.reshape(z_re.shape[0],1)
	z_im = cvt.dl2d_imp(z_im_,T_K,D0,l0,c0)
	z_im_2d = z_im.reshape(z_im.shape[0],1)

	# Output the information on the impedance.
	print("Output impedance.csv", "\n\n")
	impedance_profile = np.concatenate([freq_2d,z_re_2d,z_im_2d],axis=1)
	np.savetxt('impedance.csv',impedance_profile,fmt='%.10e',delimiter=',',header='freq[/s],z_re[ohm],z_im[ohm]')



