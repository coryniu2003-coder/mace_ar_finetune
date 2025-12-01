import numpy as np
import sys
import math
import time
from ase.io import read
from ase.geometry.analysis import Analysis
from ase.ga.utilities import get_rdf
from ase.units import fs
import ase.md.analysis
import matplotlib.pyplot as plt


########################
# User settings below
########################

# Settings for atom probability projections:
############################################
# MD supercell size
supercell=np.array([1,1,1])
# Grid size for atomic positions
grid=np.array([80,80,80])
# MD equilibration snapshots to ignore for atom mapping
prob_equilibration = 0
# Interval between snapshots to be considered
interval = 1

# Settings for RDF/PDF:
#######################
# Number of timesteps between taking RDF/PDF data:
rdf_interval = 10
# Resolution of RDF/PDF in Angstrom:
rdf_resolution = 0.01
# MD equilibration snapshots to ignore for RDF/PDF:
rdf_equilibration = 0

# Settings for (relative) MSD's:
################################
# Time between successive data snapshots, in fs 
#  (MD timestep * DFT writeout interval)
#  (VASP: POTIM * NBLOCK)
snapshot_interval = 2.0
# MD equilibration snapshots to ignore for MSD
msd_equilibration = 0

# End of user settings
########################

# Collect timing information
start_time = time.time()

print("Reading in "+sys.argv[1])
a=read(sys.argv[1],index=':')

for i in a:
    i.set_pbc(True)

ana = Analysis(a)

print("Finished reading in data.")
print("Done [%.3f s]." % (time.time() - start_time))

###############################
# Save cell parameter evolution
###############################

# Function for minimum image distance
# Taken from Militzer supercell design code
# Input: 3x3 numpy array of lattice vectors
def dmin(m):

    # rework the lattice vectors according to
    # Militzer's paper

    still_changing = True
    while still_changing:
        still_changing = False
        # sort by length
        m = m[np.argsort(np.linalg.norm(m,axis=1))]

        ap = m[0]
        bp = m[1]
        cp = m[2]

        # introduce an epsilon to avoid oscillations
        eps = 1.e-8
        # re-assign if needed
        if round(np.dot(ap,bp)/np.dot(ap,ap)+eps)!=0:
            bp = bp - ap*round(np.dot(ap,bp)/np.dot(ap,ap)+eps)
            still_changing = True
        if round(np.dot(ap,cp)/np.dot(ap,ap)+eps)!=0:
            cp = cp - ap*round(np.dot(ap,cp)/np.dot(ap,ap)+eps)
            still_changing = True
        if round(np.dot(bp,cp)/np.dot(bp,bp)+eps)!=0:
            cp = cp - bp*round(np.dot(bp,cp)/np.dot(bp,bp)+eps)
            still_changing = True

        m = np.array( [ap,bp,cp] )

    # now get minimum image distance
    d_min = max(np.linalg.norm(m,axis=1))
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if i**2+j**2+k**2>0:
                    d_image = np.linalg.norm(i*ap+j*bp+k*cp)
                    if d_image < d_min:
                        d_min = d_image

    return d_min


d_min = np.inf
lattice_file = open("lattice_vs_time.dat","w")
for i, atoms in enumerate(ana.images):
    #lattice_file.write(str(i+1) + " " + str(atoms.cell.cellpar()) + "\n")
    c = atoms.cell.cellpar()
    lattice_file.write("{0:d} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f} {6:.4f}\n".format(i+1,c[0],c[1],c[2],c[3],c[4],c[5]))
    d = dmin(np.array(atoms.cell))
    if d < d_min:
        d_min = d
rdf_max = float(0.49*d_min)
rdf_n_bins = int(rdf_max/rdf_resolution)
print("rdf_max from min image analysis: {0:.2f}\n".format(rdf_max))
lattice_file.close()

########################################
# Calculate MSD 
########################################

# Get all atomic species lists
atomic_numbers = ana.images[0].get_atomic_numbers()
chemical_symbols = ana.images[0].get_chemical_symbols()
unique_numbers = np.unique(atomic_numbers)

num_species = len(unique_numbers)
num_elements = np.zeros(num_species)
print("MSD analysis, found following {0:d} elements:".format(num_species))

unique_symbols = []
atoms_lists = [ [] for _ in range(num_species)]
for i in range(num_species):
    unique_symbols.append(chemical_symbols[np.where(atomic_numbers == unique_numbers[i])[0][0]])
    atoms_lists[i] = np.where(atomic_numbers == unique_numbers[i])
    num_elements[i] = np.shape(atoms_lists[i])[1]
    #print(unique_numbers[i], unique_symbols[i], ": ", str(int(num_elements[i])), " atoms")


print("Calculating MSD's...")

if len(ana.images)>msd_equilibration:
    
    # reference atomic positions (relative scale)
    msdpos_ini = []
    for i in range(num_species):
        msdpos_ini_species = ana.images[msd_equilibration].get_scaled_positions(wrap=False)[atoms_lists[i]]
        msdpos_ini.append(msdpos_ini_species)
    #print(msdpos_ini)
    
    # MSD lists
    timesteps = []
    msd_all = [ [] for _ in range(num_species) ]

    # Acquire MSD data
    # compare unwrapped positions to initial positions in the *current* lattice
    for i in range(msd_equilibration, len(ana.images)):
        
        cell = ana.images[i].get_cell()
        for j in range(num_species):
            msdpos_ini_species = msdpos_ini[j].dot(cell)
            msdpos_species = ana.images[i].get_positions()[atoms_lists[j]]

            delta_pos = msdpos_species - msdpos_ini_species
            msd_species = np.sum(delta_pos*delta_pos)/num_elements[j]

            msd_all[j].append(msd_species)
        
        timesteps.append(snapshot_interval*(i-msd_equilibration))

    legends = []
    for i in range(num_species):
        plt.plot(timesteps, msd_all[i])
        legends.append(unique_symbols[i])
    plt.legend(legends)
    plt.xlabel('Time (fs)')
    plt.ylabel('MSD (Ang2)')
    plt.savefig('msd-all.pdf')
    plt.clf()
    
    msd_file = open("msd-all.dat","w")
    labels=""
    for i in range(num_species):
        labels = labels + "MSD(" + unique_symbols[i] + ") "
    msd_file.write("Time "+labels+"\n")
    for r in range(len(timesteps)):
        line = str(timesteps[r])+" "
        for i in range(num_species):
            line = line + str(msd_all[i][r]) + " "
        msd_file.write(line+"\n")
    msd_file.close()
    
    print("Done. [%.3f s]" % (time.time()-start_time))

########################################
# Calculate relevant PDF's
########################################
print("Calculating PDF's...")

# Get all atomic species data
atomic_numbers = ana.images[0].get_atomic_numbers()
chemical_symbols = ana.images[0].get_chemical_symbols()
unique_numbers = np.unique(atomic_numbers)
num_species = len(unique_numbers)

num_elements = np.zeros(num_species)

unique_symbols = []
for i in range(num_species):
    unique_symbols.append(chemical_symbols[np.where(atomic_numbers == unique_numbers[i])[0][0]])
    num_elements[i] = np.shape(np.where(atomic_numbers == unique_numbers[i]))[1]
print("PDF analysis, found following {0:d} elements:".format(num_species))
for i in range(num_species):
    print(unique_numbers[i], unique_symbols[i], ": ", str(int(num_elements[i])), " atoms")

##### Directly use ase.ga.utilities.get_rdf

ls_rdf = []
for i in range(num_species):
    row = []
    for j in range(num_species):
        row.append([])
    ls_rdf.append(row)

for image in ana.images[slice(rdf_equilibration,None,rdf_interval)]:
    dm = image.get_all_distances(mic=True)
    for i in range(num_species):
        for j in range(i,num_species):
            rdf = get_rdf(image, rdf_max, rdf_n_bins, distance_matrix=dm, elements=[unique_numbers[i], unique_numbers[j]], no_dists=False)
            ls_rdf[i][j].append(rdf)

# Average over snapshots, renormalize by partial atom count
pdf_X_X = np.empty((num_species, num_species, 2, rdf_n_bins))
for i in range(num_species):
    for j in range(i,num_species):
        pdf_X_X[i][j][:] = np.mean(ls_rdf[i][j],axis=0)
        pdf_X_X[i][j][0] *= len(ana.images[0])/num_elements[j]

# Plot all together now
legends = []
for i in range(num_species):
    for j in range(i,num_species):
        plt.plot(pdf_X_X[i][j][1],pdf_X_X[i][j][0])
        legends.append(unique_symbols[i]+'-'+unique_symbols[j])
plt.legend(legends)
plt.xlim([0,rdf_max])
plt.xlabel('r (Angstrom)')
plt.ylim([0,8])
plt.ylabel('g(r)')
plt.savefig('pdf-X-X.pdf')
plt.clf()

pdf_file = open("pdf-X-X.dat","w")
labels=""
for i in range(num_species):
    for j in range(i,num_species):
        labels = labels + "PDF(" + unique_symbols[i] + "-" + unique_symbols[j] + ") "
pdf_file.write("Distance "+labels+"\n")
for r in range(rdf_n_bins):
    line = str(pdf_X_X[0][0][1][r])+" "
    for i in range(num_species):
        for j in range(i,num_species):
            line = line + str(pdf_X_X[i][j][0][r]) + " "
    pdf_file.write(line+"\n")
pdf_file.close()

print("Done. [%.3f s]" % (time.time()-start_time))

##########################################
# Calculate atom probability distributions
##########################################
print("Calculating atomic distributions...")
# Get all unique atom types
atom_symbols=list(set(ana.images[0].get_chemical_symbols()))
print(atom_symbols)

# Store atomic symbols and numbers
atom_types={}
atomic_numbers=[]
for i in range(len(atom_symbols)):
        atom_types[i] = atom_symbols[i]
        atomic_numbers.append(ana.images[0].get_atomic_numbers()[ana.images[0].get_chemical_symbols().index(atom_symbols[i])])
#print(atom_types)
#print(atomic_numbers)

# Get all atom lists
atoms_list=[]
for i in range(len(atom_symbols)):
    atoms_list.append(list(np.where(ana.images[0].get_atomic_numbers()==atomic_numbers[i])[0]))
#print(atoms_list)

# Get initial unit cell and atomic positions
cell_initial = ana.images[prob_equilibration].get_cell()
pos_initial = ana.images[prob_equilibration].get_scaled_positions()

# Get average unit cell and atomic positions
cell_average = np.mean(np.array(list(map(lambda x: x.get_cell(), ana.images[prob_equilibration:]))),axis=0)/supercell
pos_average = np.mean(np.array(list(map(lambda x: x.get_positions(), ana.images[prob_equilibration:]))),axis=0)

# Set up counting grids
counts = np.zeros(np.append(grid,len(atom_types)))
counts_prim = np.zeros(np.append(grid//supercell,len(atom_types)))
#print(counts.shape)

# Add counts of atomic positions
nsteps=0
for image in ana.images[slice(prob_equilibration,None,interval)]:
    nsteps = nsteps+1

    for at in range(len(atom_types)):
        indices = (image.get_scaled_positions()[atoms_list[at]]*grid).astype(int)%grid
        indices_list = list(map(tuple,indices))
        #print(nsteps, atom_types[at],len(indices_list))
        # index reversal below needed for VASP-style output ordering: (z,y,x)
        for indx in indices_list:
            counts[tuple(reversed(indx))+tuple([at])] = counts[tuple(reversed(indx))+tuple([at])] + 1
            #print(indx+tuple([at]))

        indices_prim = indices%(grid//supercell)
        prim_list = list(map(tuple,indices_prim))
        for indx in prim_list:
            counts_prim[tuple(reversed(indx))+tuple([at])] = counts_prim[tuple(reversed(indx))+tuple([at])] + 1
    #print(".",end='',flush=True)

# renormalise counts
print("Number of geometry steps found: "+str(nsteps)+"\n")
counts = counts/nsteps
counts_prim = counts_prim/nsteps

for at in range(len(atom_types)):
    #print(counts[:,:,:,at].shape)
    counts_flattened = counts[:,:,:,at].flatten()
    prim_flattened = counts_prim[:,:,:,at].flatten()

    print("Atom "+atom_types[at]+":\n")
    print("Grid sites occupied throughout run: "+str(len(np.nonzero(counts_flattened)[0]))+"\n")

    # write data to file
    probcar = open("PROBCAR_"+atom_types[at]+".vasp","w")
    probcar.write("Probability distribution\n")
    probcar.write("1.00\n")
    probcar.write("{:.6f} {:.6f} {:.6f}\n".format(cell_initial[0][0],cell_initial[0][1],cell_initial[0][2]))
    probcar.write("{:.6f} {:.6f} {:.6f}\n".format(cell_initial[1][0],cell_initial[1][1],cell_initial[1][2]))
    probcar.write("{:.6f} {:.6f} {:.6f}\n".format(cell_initial[2][0],cell_initial[2][1],cell_initial[2][2]))
    for ii in range(len(atom_types)):
        probcar.write("{:s} ".format(atom_types[ii]))
    probcar.write("\n")
    for ii in range(len(atom_types)):
        probcar.write("{:d} ".format(len(atoms_list[ii])))
    probcar.write("\n")
    probcar.write("Direct\n")
    for jj in pos_initial:
        probcar.write("{:f} {:f} {:f}\n".format(jj[0],jj[1],jj[2]))
    probcar.write(str(grid[0])+" "+str(grid[1])+" "+str(grid[2])+"\n")
    for i, pr in enumerate(counts_flattened):
        probcar.write("{:.6f} ".format(pr))
        if i%5==4: probcar.write("\n")
    probcar.close()
    
    # project back onto primitive unit cell
    probprim = open("PROBPRIM_"+atom_types[at]+".vasp","w")
    probprim.write("Probability distribution\n")
    probprim.write("1.00\n")
    probprim.write("{:.6f} {:.6f} {:.6f}\n".format(cell_average[0][0],cell_average[0][1],cell_average[0][2]))
    probprim.write("{:.6f} {:.6f} {:.6f}\n".format(cell_average[1][0],cell_average[1][1],cell_average[1][2]))
    probprim.write("{:.6f} {:.6f} {:.6f}\n".format(cell_average[2][0],cell_average[2][1],cell_average[2][2]))
    for ii in range(len(atom_types)):
        probprim.write("{:s} ".format(atom_types[ii]))
    probprim.write("\n")
    for ii in range(len(atom_types)):
        probprim.write("{:d} ".format(len(atoms_list[ii])))
    probprim.write("\n")
    probprim.write("Cartesian\n")
    for jj in pos_average:
        probprim.write("{:f} {:f} {:f}\n".format(jj[0],jj[1],jj[2]))
    probprim.write(str(grid[0]//supercell[0])+" "+str(grid[1]//supercell[1])+" "+str(grid[2]//supercell[2])+"\n")

    for i, pr in enumerate(prim_flattened):
        probprim.write("{:.6f} ".format(pr))
        if i%5==4: probprim.write("\n")
    probprim.close()

print("Done. [%.3f s]" % (time.time()-start_time))

