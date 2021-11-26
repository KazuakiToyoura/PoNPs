#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Output profiles of defect concentrations and electric potential.
def plot(dx_m,c,dx_b,phi,n_elem,n_elem_ldebye,n_ldebye,file_name,defect_name,t):
	# Total number of elements. 
	n_elem_sum = np.sum(n_elem)
	# Check the lower and upper bounds of concentrations
	c_positive = c[np.where(c>0.0)]
	logc_LB = np.floor(np.log10(np.min(c_positive)))
	logc_UB = np.ceil(np.log10(np.max(c_positive)))

	# Plots vs. position x [cm].
	xm = np.append(0.0,np.cumsum(dx_m)) # Coordinates of element boundaries
	xb = np.append(0.0,np.cumsum(dx_b[1:-1])) # Coordinates of element centers
	
	# The initial and final coordinates for each Layer
	x_layers = []
	for i in range(n_elem.shape[0]):
		x_layers.append([xb[np.sum(n_elem[:i])],xb[np.sum(n_elem[:i+1])]])

	fig=plt.figure()
	# Plot concentration profiles
	ax1=fig.add_subplot(2,1,1)
	for i in range(c.shape[0]):
		ax1.plot(xm,np.log10(np.where(c[i,:]>0.0,c[i,:],10.0**(logc_LB-10))),marker='x',markersize=4,label=defect_name[i],linewidth=0.5)
	ax1.set_xlabel("x [cm]")
	ax1.set_ylabel("Log(c/[mol/cm3])")
	ax1.set_ylim(logc_LB,logc_UB)
	ax1.legend(ncol=defect_name.shape[0], bbox_to_anchor=(1,1), loc='lower right', borderaxespad=0,frameon=True,facecolor='white',edgecolor='black',fontsize=8)
	ax1.text(-0.12,1.05,"t = "+str(t)+" [s]",fontweight='bold',transform=ax1.transAxes,fontsize=10)
	# Plot electric potential profile
	ax2=fig.add_subplot(2,1,2)
	ax2.plot(xb,phi,marker='x',markersize=4,linewidth=0.5)
	ax2.set_xlabel("x [cm]")
	ax2.set_ylabel("phi [V]")
	# Fill layer area.
	for i in range(len(x_layers)):
		if i%2==0:
			ax1.axvspan(x_layers[i][0], x_layers[i][1], color = "whitesmoke")
			ax2.axvspan(x_layers[i][0], x_layers[i][1], color = "whitesmoke")

	fig.tight_layout()	
	#plt.savefig("profile_"+file_name+".png", format='png', dpi=150, bbox_inches="tight")
	plt.savefig("profile_"+file_name+".pdf", bbox_inches="tight")


	# Plots vs. elem IDs.
	xm = np.append(0.0,0.5+np.array(range(n_elem_sum)))
	xm = np.append(xm,float(n_elem_sum))
	xb = np.array(range(n_elem_sum+1))

	# The initial and final coordinates for each Layer
	x_layers = []
	for i in range(n_elem.shape[0]):
		x_layers.append([xb[np.sum(n_elem[:i])],xb[np.sum(n_elem[:i+1])]])

	fig=plt.figure()
	# Plot concentration profiles
	ax3=fig.add_subplot(2,1,1)
	for i in range(c.shape[0]):
		ax3.plot(xm,np.log10(np.where(c[i,:]>0.0,c[i,:],10.0**(logc_LB-10))),marker='x',markersize=4,label=defect_name[i],linewidth=0.5)
	ax3.set_xlabel("element id")
	ax3.set_ylabel("Log(c/[mol/cm3])")
	ax3.set_ylim(logc_LB,logc_UB)
	ax3.legend(ncol=defect_name.shape[0], bbox_to_anchor=(1,1), loc='lower right', borderaxespad=0,frameon=True,facecolor='white',edgecolor='black',fontsize=8)
	ax3.text(-0.12,1.05,"t = "+str(t)+" [s]",fontweight='bold',transform=ax3.transAxes,fontsize=10)
	# Plot electric potential profile
	ax4=fig.add_subplot(2,1,2)
	ax4.plot(xb,phi,marker='x',markersize=4,linewidth=0.5)
	ax4.set_xlabel("element id")
	ax4.set_ylabel("phi [V]")
	# Fill layer area.
	for i in range(len(x_layers)):
		if i%2==0:
			ax3.axvspan(x_layers[i][0], x_layers[i][1], color = "whitesmoke")
			ax4.axvspan(x_layers[i][0], x_layers[i][1], color = "whitesmoke")

	fig.tight_layout()	
	#plt.savefig("profile_xid_"+file_name+".png", format='png', dpi=150, bbox_inches="tight")
	plt.savefig("profile_xid_"+file_name+".pdf", bbox_inches="tight")




# Output Nyquist plot and Bode diagram.
def imp_plot(z_re,z_im,freq):
	print("Output impedance.pdf & Bode.pdf")
	fig=plt.figure()
	# Nyquist plot
	ax1=fig.add_subplot(1,1,1)
	ax1.set_aspect('equal')
	ax1.plot(z_re,-z_im,marker='x',markersize=1,linewidth=0.1)
	ax1.set_xlabel("z_real [ohm]")
	ax1.set_ylabel("-z_imaginary [ohm]")
	plt.xlim(0,)
	fig.tight_layout()	
	#plt.savefig("impedance.png", format='png', dpi=150, bbox_inches="tight")
	plt.savefig("impedance.pdf", bbox_inches="tight")

	# Bode diagram
	fig=plt.figure()
	# |z| vs. log_f
	ax2=fig.add_subplot(2,1,1)
	#ax2.plot(np.log10(freq),np.sqrt(z_re**2+z_im**2),marker='x',markersize=1,linewidth=0.1)
	ax2.plot(np.log10(freq),np.log10(np.sqrt(z_re**2+z_im**2)),marker='x',markersize=1,linewidth=0.1)
	ax2.set_xlabel("log(f/[/s])")
	#ax2.set_ylabel("|z| [ohm]")
	ax2.set_ylabel("log(|z|/[ohm])")
	# phase angle vs. log_f
	ax3=fig.add_subplot(2,1,2,ylim=(-90,90))
	ax3.plot(np.log10(freq),np.rad2deg(np.arctan(z_im/z_re)),marker='x',markersize=1,linewidth=0.1)
	ax3.set_xlabel("log(f/[/s])")
	ax3.set_ylabel("phase angle [degree]")
	ax3.set_yticks([-90,-60,-30,0,30,60,90])

	fig.tight_layout()	
	#plt.savefig("Bode.png", format='png', dpi=150, bbox_inches="tight")
	plt.savefig("Bode.pdf", bbox_inches="tight")


