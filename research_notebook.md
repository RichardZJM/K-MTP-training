# MECH 461 Research Notebook
https://github.com/RichardZJM/K-MTP-training

##Introduction
This is the research notebook for the dataset generation of moment tensor potentials (MTP) for potassium, and the subsequent application in molecular dynamics simulations. Included is a week-by-week breakdown of the progress and findings of each session.

## Week 1
On the first week, consisted primarily of setup and understanding the MTP techniques used in the machine learning model. I started by meeting, Hao Sun, a post-doctorate student who is part of the Nuclear Materials group. His prior work involved the training of an MTP potential and thus his experience would be invaluable.

The first bit of setup focused on Compute Canada HPC cluster access. Compute Canada is a resource of the Digital Research Alliance of Canada. Registering with my supervisor Laurent Béland as the Principal Investigator (PI), gave me access to the more computational model which would be important for the training and active learning of the MTP. In particular, SSH access to Calcul Quebec's Narval supercomputer will be used for most of the simulation.

With access to Narval, I then took a closer examination of the mathematics and theory behind the MTP approach before I started work on developing training sets and running training the system.  This took much of the remainder of week 1 and a brief overview of my understanding is outlined below. A more in-depth description will be provided for the final report.

### The MTP interatomic model
<p style ="font-size:smaller">
Please note that the following are personal notes taken from [REF] that explain the MTP method for my personal reference and future understanding.
</p>

As an atomistic potential, the MTP method describes the energy of a system as a function of the configuration of its atoms. The MTP potential does this by considering the sum of the energies associated with each of the atoms within the system.

The energy of each atom can be defined as the weighted summations through a set of basis functions:

$$
V(n_i) = \sum_{\alpha} \xi_\alpha   \Beta_\alpha
$$

The weightings, $\xi_\alpha$, are trainable parameters in the machine learning of the algorithm. $\Beta_\alpha$ are the members of the basis function to the level specified as a model hyperparameter.

The basis functions are constructed with moment tensor descriptors. These moment tensor descriptors contain radial and angular components which capture the geometric representation of the system environment local to the atom whose energy is being calculated (as defined by a cutoff radius). 

Moment tensor descriptors are differentiated by two different parameters $\nu$ and $\mu$. The former can be conceptually thought of as the depth of the angular data that the particular moment tensor captures. The latter allows the system to exhibit more trainable radial parameters. The moment tensor descriptor for the $ith$ atom, described by $\nu$ and $\mu$, is the summation of the products of the corresponding radial and angular components between the $i$th atom and its $j$ neighbours. 

$$M_{\mu,\nu} (n_i)= \sum_{j} f_\mu (r_{ij},z_i,z_j) r_{ij} \otimes \dots \otimes r_{ij} $$

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

As moment tensors descriptor are described by $\nu$ and $\mu$, we can define a measure the complexity of a moment tensor descriptor based on these two values—the so-called *level* of a moment tensor descriptor. This is given by:

$$\textrm{lev}M_{\mu,\nu} = 2 + 4\mu + \nu$$

Where there the coefficients were determined through experimentation.

Additionally, we can perform binary tensor operations and contractions between various moment tensor descriptors to form additional expressions as long as the operations are valid and one of the following:

1. Dot Product
2. Frobenius Product
3. Scalar Multiplication

The level of an expression of moment tensor descriptors is given by the sum of the level of its constituents. 

We ultimately choose the basis set of the MTP model based on a *maximum level*, $\textrm{lev}_{\max}$ which serves as one of the model's hyperparameters. The basis set consists of all combinations of moment tensor descriptors that use the above operands to contract down to a single scalar value such that the formed expression has a level no more than $\textrm{lev}_{\max}$.  Accordingly, the number of trainable parameters is heavily dependent on $\textrm{lev}_{\max}$, scaling exponentially.

Now, consider the radial and angular components as characterized by the $\nu$ and $\mu$ of the particular moment tensor descriptor.

#### Radial Component of the Moment Tensor Descriptor
The radial component, $f_\mu (r_{ij},z_i,z_j)$, is described as the summation of the product of the members of the radial basis set, $Q^{(\Beta)(r_{ij})}$, and the corresponding trainable radial parameters $c^{\Beta} _ {\mu,z_i,z_j}$.

$$f_\mu (r_{ij},z_i,z_j) = \sum ^ {N_o} _ {\Beta = 1} c^{(\Beta)} _ {\mu,z_i,z_j}  Q^{(\Beta)(r_{ij})}$$

The number of members of the radial basis set, $N_o$ is chosen as a model hyperparameter. The basis set is conditionally evaluated based on the chosen cutoff radius and the minimum distance between atoms in the system, using Chebyshev polynomials on the interval $[R_{min}, R_{cut}]$.

$$Q^{(\Beta)(r_{ij})} =  \begin{cases}
    \phi ^(\Beta)(|r_{ij}|) (R_{cut} - |r_{ij}|)^2& |r_{ij}| < R_{cut} \\
    0 & wk
\end{cases}$$

Where $\phi^(n)$ represents the $n$th Chebyshev polynomial. This generates a Chebyshev polynomial sequence that smoothly decays to zero at the cutoff radius.

#### Angular Component of the Moment Tensor Descriptors
The angular component is a series of $\nu$ outer products performed on the position vector between the originating atom and its $j$th neighbour. The value of $\nu$ is dependent on the exact moment tensor descriptor. This angular component works to capture the angular information between two atoms and results in a tensor whose rank is equivalent to $\nu$.

#### MTP Model Overview
Overall,  the MTP potential provides a framework atop which radial and angular components are considered. Additionally, the tensor operations performed on the descriptors maintain the model's invariance to permutations, rotations, reflection, and translation. MTP has two hyperparameters which affect its accuracy and computational cost: $N_o$ and $\textrm{lev}_{\max}$ which determine the number of basis functions expressions and the number of Chebyshev polynomials evaluated in the radial basis set. 

The trainable parameters of the model are weightings with respect to the moment tensor basis sets and the radial basis sets, $\xi_\alpha$ and $c^{\Beta} _ {\mu,z_i,z_j}$ respectively. Collectively, the vector of trainable parameters will be further expressed as $x$.



### Training
To prepare an MTP for usage in MD simulations like LAMMPS, training is generally performed on quantum-mechanical databases. The initial training is performed using the quantum-mechanical energies $E^{qm}$, forces $f_i^{qm}$, and $\sigma^{qm}$ stress tensors. This is performed using a loss function which considers the sum of the square errors of the energies of a configuration, the sum of the squared magnitude of the force errors for each atom in a configuration, and the error of the Frobenius norm of the stress tensors of a configuration, for all configurations in the training data set.

The loss function is mathematically expressed for the number of configurations in the dataset, $K$

$$\sum_{k=1}^K w_e \delta E + w_f \delta F + w_s \delta S $$

Where, $w_e$, $w_f$, and $w_s$ represent weighting factors (user-chosen) that express the relative importance of each the energies, forces, and stress tensors respectively. Further, the error contributions for each property can be expressed for the $k$th configuration and trainable parameter vector, $x$.

| Component | Mathematical Expression | Description|
| :---------: | -------------------------------------- | -|
| $\delta E$    |$(E^\textrm{mtp}(\textrm{cfg}_k, x) - E^\textrm{qm}(\textrm{cfg}_k, x))^2$| Loss contribution from energy inaccuracies|
| $\delta F$    |$(E^\textrm{mtp}(\textrm{cfg}_k, x) - E^\textrm{qm}(\textrm{cfg}_k, x))^2$  | Loss contribution from force inaccuracies in each atom in the configuration||
| $\delta S$    |$(E^\textrm{mtp}(\textrm{cfg}_k, x) - E^\textrm{qm}(\textrm{cfg}_k, x))^2$ | Species of the $j$th neighbour| |

The mathematical optimization of this loss function doesn't use any special approach for gradient solving like backpropagation for more traditional ML techniques. Instead, we use a BFGS approach against the training set. Afterwards, an estimation of the trained potential accuracy can be obtained using root mean square error on the energies, forces, and stress tensors. RSME is shown for the energies below.


 $\textrm{RSME} (E)^2 = \frac{1}{K} \sum ^{K}_{k=1} (\frac{E^\text{mtp}(\text{cfg}_g,x)}{N^{(k)}}-\frac{E^\text{qm}(\text{cfg}_g,x)}{N^{(k)}})$

## Week 2
The validity of the MTP (and other machine learning potentials ) is predicated on the availability of high-fidelity training data. For the MTP, these calculations consist of DFT calculations using Quantum Espresso. Most of Week 2 focused on familiarizing myself with the Narval HPC enviroment. 

This all starts by connecting to Narval through my newly-minted Compute Canada account and SSH. Then, I follow the prompts, entering my password to gain access.

```
ssh -Y zjm@narval.computecanada.ca
```
Hao initially recommend me the MobaXterm terminal for automatic reconnection and SFTP (file transfer). However, I am currently running a Linux-based personal machine, and am using Tabby Terminal to perform the same.

Upon SSH'ing into Narval, I'm greeted with the home directory in one of the login nodes (narval3 in this case).

![Narval Login Node](notebook_images/narval_home.png)

This is a Linux-based terminal environment with no GUI. I have some experience with similar environments having run Linux natively for CFD purposes.  I immediately started by setting up a git repository for the project. This is to hold version-controlled scripts, output files, and the research notebook. Moreover, I can easily perform edits on my local machine and push them to Narval without SCP or SFTP. Graphical applications such are Ovito are also unavailable on Narval. The Github link is available below.

https://github.com/RichardZJM/K-MTP-training

Narval utilizes the job scheduler Slurm Workload manager for intensive computations. Only tasks smaller than 10 CPU-minutes and 4 RAM are permissible on the login nodes. Slurm is essentially a priority queue for Narval's nodes. Users submit job requests for system resources in job requests. Priority can be allocated based on the relative importance allocated to the project and the principal researcher. A job request resembles the below.

```
#!/bin/bash                                                 
#SBATCH --account=def-belandl1                  // PI's account
#SBATCH --ntasks=1                              // Number of CPUs for job
#SBATCH --time=0-2:00 # time (DD-HH:MM)         // Estimated job duration
#SBATCH --mem-per-cpu=9G                        // Memory per CPU requirement

//Code to run on the Narval compute nodes with allocated resources
// ...
```

A job is defined in the format above in a text file with the .qsub extension. They are submitted to the Slurm system using the sbatch command.

```
sbatch <file>
```
Listed below are some additional Slurm commands which are generally useful.

```
squeue                  // Shows the current jobs in queue and running
squeue -u zjm           // Shows the curret jobs associated with user
scancel <job number>    // Cancels the specified job
scancel -u zjm          // Cancels all jobs associated with user
```
With these tools to perform computationally intensive calculations on Narval, I began to assemble the initial DFT dataset. I received several scripts from Hao from which I adapted my first set of jobs for periodic potassium cells at under shear and hydrostatic expansion/compression. 

The first step is to utilize Quantum Expresso to determine the lattice parameter of BCC potassium from this value. This represents a realistic baseline from which the initial dataset can be built around. We start by determining DFT parameters which produce well-converged results, including the plane wave cutoff energy and the number of uniform k-points. From previous experience with convergence testing with Potassium in Quantum Espresso, I settled on the following parameters for all future Potassium DFT simulations.

| Parameter                | Value |
| :----------------------- | ----- |
| Plane Wave Cutoff Energy | 60 Ry |
| K-Point Count            | 8     |
| Pseudopotential          | K.pbe-mt-fhi.UPF 

Pseudopotential Source: https://github.com/buck54321/pyspresso/tree/master/pseudo



#References
https://iopscience.iop.org/article/10.1088/2632-2153/abc9fe
/home/zjm/mlip-2/bin/mlp train 08.mtp mlip_input.cfg --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=pot.mtp


module load    StdEnv/2020  gcc/9.3.0  openmpi/4.0.3
module load    quantumespresso/6.6