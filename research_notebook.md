#MECH 461 Research Notebook

##Introduction
This is the research notebook for the dataset generation of moment tensor potentials (MTP) for potassium, and the subsequent application in molecular dynamics simulations. Included is a week-by-week breakdown of the progress and findings of each session.

##Week 1
On the first week, consisted primarily of setup and understanding the MTP techniques used in the machine learning model. I started by meeting, Hao Sun, a post-doctorate student who is part of the Nuclear Materials group. His prior work involved the training of an MTP potential and thus his experience would be invaluable.

The first bit of setup focused on Compute Canada HPC cluster access. Compute Canada is a resource of the Digital Research Alliance of Canada. Registering with my supervisor Laurent Béland as the Principal Investigator, gave me access to the more computational model which would be important for the training and active learning of the MTP. In particular, SSH access to Calcul Quebec's Narval supercomputer will be used for most of the simulation.

With access to Narval, I then took a closer examination of the mathematics and theory behind the MTP approach before I started work on developing training sets and running training the system. As an atomistic potential, the MTP method describes the energy of a system as a function of the configuration of its atoms. The MTP potential does this by considering the sum of the energies associated with each of the atoms within the system.

The energy of each atom can be defined as the weighted summations through a set of basis functions:

$$
V(n_i) = \sum_{\alpha} \xi_\alpha   \Beta_\alpha
$$

The weightings, $\xi_\alpha$, are trainable parameters in the machine learning of the algorithim. $\Beta_\alpha$ are the members of the basis function to the level specified as a model hyperparameter.

The basis functions are based on moment tensor descriptors. These moment tensors descriptors contains radial and angular components which caputure the geometric representation of the system enviroment local to the atom whose energy is being calculated (as defined by a cutoff radius).

| Component | Description |
| --------- | ----------- |
| Radial    | Title       |
| Angular   | Text        |

#References
https://iopscience.iop.org/article/10.1088/2632-2153/abc9fe
/home/zjm/mlip-2/bin/mlp train 08.mtp mlip_input.cfg --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=pot.mtp
