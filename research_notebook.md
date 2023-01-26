#MECH 461 Research Notebook

##Introduction
This is the research notebook for the dataset generation of moment tensor potentials (MTP) for potassium, and the subsequent application in molecular dynamics simulations. Included is a week-by-week breakdown of the progress and findings of each session.

##Week 1
On the first week, consisted primarily of setup and understanding the MTP techniques used in the machine learning model. I started by meeting, Hao Sun, a post-doctorate student who is part of the Nuclear Materials group. His prior work involved the training of an MTP potential and thus his experience would be invaluable.

The first bit of setup focused on Compute Canada HPC cluster access. Compute Canada is a resource of the Digital Research Alliance of Canada. Registering with my supervisor Laurent Béland as the Principal Investigator, gave me access to the more computational model which would be important for the training and active learning of the MTP. In particular, SSH access to Calcul Quebec's Narval supercomputer will be used for most of the simulation.

With access to Narval, I then took a closer examination of the mathematics and theory behind the MTP approach before I started work on developing training sets and running training the system.  This took much of the remainder of week 1 and a brief overview of my understanding is outlined below. A more in-depth description will be provided for the final report.

### The MTP interatomic model
As an atomistic potential, the MTP method describes the energy of a system as a function of the configuration of its atoms. The MTP potential does this by considering the sum of the energies associated with each of the atoms within the system.

The energy of each atom can be defined as the weighted summations through a set of basis functions:

$$
V(n_i) = \sum_{\alpha} \xi_\alpha   \Beta_\alpha
$$

The weightings, $\xi_\alpha$, are trainable parameters in the machine learning of the algorithm. $\Beta_\alpha$ are the members of the basis function to the level specified as a model hyperparameter.

The basis functions are constructed on moment tensor descriptors. These moment tensor descriptors contain radial and angular components which capture the geometric representation of the system environment local to the atom whose energy is being calculated (as defined by a cutoff radius). 

Moment tensor descriptors are differentiated by two different parameters $\nu$ and $\mu$. The former can be conceptually thought of as the depth of the angular data that the particular moment tensor captures. The latter allows the system to  


| Component | Description                            |
| --------- | -------------------------------------- |
| Radial    | $f_\mu (r_{ij},z_i,z_j)$               |
| Angular   | $r_{ij} \otimes \dots \otimes r_{ij} $ |
Where,  the important properties of the atoms in local enviroment are described by:

| Component | Description                            |
| :---------: | -------------------------------------- |
| $r_{ij}$    | Vector from the originating atom to its $j$th neighbour |
| $z_{i}$  | Species of the originating atom|
| $z_{j}$  | Species of the $j$th neighbour|


####Radial Component of the Moment Tensor Descriptor
The radial component, $f_\mu (r_{ij},z_i,z_j)$, is described as the summation of the product of the members of the radial basis set, $Q^{(\Beta)(r_{ij})}$, and the corresponding trainable radial parameters $c^{\Beta} _ {\mu,z_i,z_j}$ .

$$f_\mu (r_{ij},z_i,z_j) = \sum ^ {N_o} _ {\Beta = 1} c^{(\Beta)} _ {\mu,z_i,z_j}  Q^{(\Beta)(r_{ij})}$$

The number of members of the radial basis set, $N_o$ is chosen as a model hyperparameter. The basis set is conditionally evaluated based on the chosen cutoff radius and the minimum distance between atoms in the system, using Chebyshev polynomials on the interval $[R_{min}, R_{cut}]$.

$$Q^{(\Beta)(r_{ij})} =  \begin{cases}
    \phi ^(\Beta)(|r_{ij}|) (R_{cut} - |r_{ij}|)^2& |r_{ij}| < R_{cut} \\
    0 & wk
\end{cases}$$

Where $\phi^(n)$ represents the $n$th Chebyshev polynomial.

####Angular Component of the Moment Tensor Descriptors
The angular component is a series of $\nu$ outer products performed on the position vector between the originating atom and its $j$th neighbour. The value is dependent on the exact moment tensor. This angular component works to capture the angular information between two atoms and results in a tensor whose rank is equivalent to $\nu$.

####Determining Basis Functions from Moment Tensor Descriptors
A moment tensor descriptor is described by two parameters, 



#References
https://iopscience.iop.org/article/10.1088/2632-2153/abc9fe
/home/zjm/mlip-2/bin/mlp train 08.mtp mlip_input.cfg --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=pot.mtp
