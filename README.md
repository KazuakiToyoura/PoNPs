# Poisson-Nernst-Planck-Simulator
Numerical Simulator for the Poisson-Nernst-Planck systems with multiple ions

# Demo
A steady state (constant current), a transition process from a steady state to another, and an impedance spectrum can be treated in PoNPs.

* Time evolution of defect concentrations and electric potential 
![profile](https://user-images.githubusercontent.com/93914342/142352821-9c66bdfb-6317-452a-8758-1bca05ddfb7e.gif)

* Nyquist plot
![impedance-1](https://user-images.githubusercontent.com/93914342/142379603-315a1925-762d-4cd2-b916-bfbf480d83fe.png)
* Bode diagram
![Bode-1](https://user-images.githubusercontent.com/93914342/142375814-73c40b34-6633-4530-bac9-6c0506c4485e.png)

# Programming language
PoNPs uses python 3 (installed by anaconda3)

Imported modules: numpy, scipy, matplotlib, argparse, copy, os, sys, datetime

# Installation
Just download all files in your preferred directory.

# Usage
Just type the following command at the directory with required six input files (+optional files).
Default names of input files (comp.csv, defect.csv, layer.csv, reaction.csv, conc.csv, efield.csv) can be skipped.
```
python ponps.py --comp comp.csv --defect defect.csv --layer layer.csv --reaction reaction.csv --conc conc.csv --efield efield.csv
```

# Input files
All input files should be prepared in the csv format. The last three are optional.

comp.csv: Computational conditions

defect.csv: Information on all defect species (charge carriers)

layer.csv: Information on all layers in a system

reaction.csv: Information on all interlayer and intralayer reactions.

conc.csv: Initial profiles of all defect species

efield.csv: Initial profiles of electric field

spaceGrid_layerID.csv (optional): Space grid intervals in a layer (File name is specified in layer.csv. Separated files are required for individual defect species.)

timeGrid.csv (optional): Information on the time grid (File name is specified in comp.csv.)

freqGrid.csv (optional): Information on the frequency for impedance spectroscopy (File name is specified in comp.csv.)

# Output files
conc_fin.csv: Final profiles of all defect species

efield_fin.csv: Final profiles of electric field

voltage_profile.csv: Time dependence of voltage between both ends

phi_profile.csv: Time dependence of electric potential profile

conc_defectID_profile.csv: Time dependence of concentration profile of each defect species

partialI_defectID_profile.csv: The profile of partial current for each defect species

impedance.csv: Information on impedance spectrum (Only for impedance simulations)

profile_timeStep.pdf: Figures for the profiles of defect concentrations and electric potential at selected time 
steps. The x axis is shown in the length scale in cm.

profile_xid_timeStep.pdf: Same information on profile_timeStep.pdf. The x axis is shown by the space grid id.

impedance.pdf: Nyquist plot (Only for impedance simulations)

Bode.pdf: Bode diagram (Only for Impedance simulations)

# Author

* Kazuaki Toyoura, PhD.
  Department of Mater. Sci. & Eng., Kyoto Univ.

# Collaborator

* Katsuhiro Ueno, PhD.
  Department of Mater. Sci. & Eng., Kyoto Univ.

# License

PoNPs is under [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause)
