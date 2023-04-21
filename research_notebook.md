# MECH 461 Research Notebook
## Introduction
This is the research notebook for the dataset generation of moment tensor potentials (MTP) for potassium, and the subsequent application in molecular dynamics simulations. Included is a week-by-week breakdown of the progress and findings of each session.

https://github.com/RichardZJM/K-MTP-training

## Terminology
| Term | Description                            |
| :---------: | -------------------------------------- |
|QM| Quantum mechanical|
| DFT   | Density Functional Theory: quantum mechanical approach to calculating the energies and forces of an atomic configuration|
| MD  | Molecular Dynamics: uses a classical representation of atoms to solve the equations of motion. Needs a description of the forces exerted on atoms|
|ML | Machine Learning|
|MTP| Moment Tensor Potential: a ML model of interatomic forces and energies|
|MLIP| Machine Learning Interatomic Potential: a software package that implmenets the MTP|
|CAC|Centre for advanced computing: the Queen's computational cluster|
|DRAC| Digital Reserach Alliance of Canada: authority which grants acces to canadian research clusters|
|QE| Quantum Espresso: a software which performs plane-wave DFT calculations|
|LAMMPS| Large-scale Atomic/Molecular Massively Parallel Simulator: a popular MD software|
|SSH| Secure Shell, a network protocol to connect to remote computers securely|



## Table of Contents
- [MECH 461 Research Notebook](#mech-461-research-notebook)
  - [Introduction](#introduction)
  - [Terminology](#terminology)
  - [Table of Contents](#table-of-contents)
  - [Week 1](#week-1)
      - [Monday, January 9th](#monday-january-9th)
      - [Tuesday, January 10th](#tuesday-january-10th)
      - [Thursday, January 11th - Saturday, January 14th](#thursday-january-11th---saturday-january-14th)
  - [Week 2](#week-2)
      - [Monday, January 16th](#monday-january-16th)
      - [Tuesday, January 17th](#tuesday-january-17th)
      - [Wednesday, January 18th](#wednesday-january-18th)
      - [Thursday, January 19th](#thursday-january-19th)
      - [Friday, January 20th - Saturday, January 22nd](#friday-january-20th---saturday-january-22nd)
  - [Week 3](#week-3)
      - [Monday, January 24th](#monday-january-24th)
      - [Tuesday, January 25th](#tuesday-january-25th)
      - [Wednesday, January 26th](#wednesday-january-26th)
      - [Friday, January 28th - Saturday, January 29th](#friday-january-28th---saturday-january-29th)
- [General Notes](#general-notes)
    - [The MTP interatomic model](#the-mtp-interatomic-model)
      - [Radial Component of the Moment Tensor Descriptor](#radial-component-of-the-moment-tensor-descriptor)
      - [Angular Component of the Moment Tensor Descriptors](#angular-component-of-the-moment-tensor-descriptors)
      - [MTP Model Overview](#mtp-model-overview)
    - [Training](#training)
    - [Narval Cluster](#narval-cluster)
    - [Slurm Job Manager](#slurm-job-manager)
    - [Quantum Espresso](#quantum-espresso)
      - [Plane Wave Function](#plane-wave-function)
      - [AI MD](#ai-md)
    - [LAMMPS](#lammps)
    - [MLIP](#mlip)
      - [MLIP commands](#mlip-commands)
    - [Preparing the first DFT calculations](#preparing-the-first-dft-calculations)
  - [Week 3](#week-3-1)
    - [Format of the MTP File](#format-of-the-mtp-file)
    - [Format of Atomic Configurations](#format-of-atomic-configurations)
- [References](#references)


## Week 1
#### Monday, January 9th
Having made arrangements for a meeting the next day with my supervisor, I started by applying for cluster access with the DRACAccount and the Queen's CAC Frontenac platform. This involved following the tutorials available below.

[DRAC](https://alliancecan.ca/en/services/advanced-research-computing/account-management/apply-account)
[CAC](https://cac.queensu.ca/wiki/index.php/Access:Frontenac)

The clusters are essentially large computers that can handle multiple or single tasks in parallel, which allows me to proceed with the computations at greater speed and stability.

#### Tuesday, January 10th
Today, I met with my supervisor, Laurent Béland. He also introduced me to Hao Sun, a post-doctorate researcher who is part of the Nuclear Materials group. His prior work involved the training of an MTP potential for sodium and thus his experience would be invaluable in helping guide much of my work and avoid some of the common pitfalls that I might encounter.

We outlined a general plan for the progression of the project as a whole:
1. Read the literature and understand the model
2. Setup up the software packages on the cluster and understand job scheduling
3. Make some initial DFT calculations and prepare baselines for mass DFT calculations
4. Passive train some MTP on simple DFT
5. Run some MTP molecular dynamics
6. Run and automate the active learning process

#### Thursday, January 11th - Saturday, January 14th
During this time, I spend most of my time reading through the theory of the moment tensor potential architecture and the documentation of the MLIP package which implements the potential in C++, and would need to be set up on the Cluster. There is an additional interface that is required to run the MTP in LAMMPS. The links are below.

[MLIP](https://iopscience.iop.org/article/10.1088/2632-2153/abc9fe#mlstabc9fes2)
[MTP](https://epubs.siam.org/doi/abs/10.1137/15M1054183?casa_token=RzGStb-dQuEAAAAA:doul_FY1J2XILDG6YjSQCC-WirCG1ZalUc48Z8jSozpAam_pww8D2E55JWZuNY_BsqLfGFjC)
[Interface](https://gitlab.com/ashapeev/interface-lammps-mlip-2)


My extended notes on the MTP architecture can be found in the General Notes section.


## Week 2
#### Monday, January 16th
This week was mostly focused on getting the software environment set up on the Narval, on the cluster operated through the DRAC. Today, I started by setting up a meeting with Hao for 11:30 AM the following day, This would be a recurring meeting to check up on my progress each week.

#### Tuesday, January 17th
During the meeting with Hao, we started by connecting to Narval for the first time through SSH. This was done through the below command.

```bash
ssh -Y zjm@narval.computecanada.ca
```

Alternatively, it was also possible to connect more easily through a program called MobaXterm which would also allow easier file transfer between a local environment and the cluster. However, it was Windows-based, and since I was running Linux natively, I ended up switching to SSH through the Linux terminal and used Git for version control and file transfer.

For the rest of the session, we focused on the installation of the MLIP package which I have detailed in the General Notes section. At the end of this session, Hao left me with some of the example scripts which I could digest later throughout the week.

#### Wednesday, January 18th
Today, I completed the installation of the MLIP package and LAMMPS interface on the cluster. The last command (detailed in General Notes) had taken too long for our session, and I had to close and run it at home. Not much else happened today. The full installation instructions which I followed are available below: 

[MLIP Instalation](https://gitlab.com/ashapeev/interface-lammps-mlip-2)


#### Thursday, January 19th 
I started digesting some of the scripts that Hao had given me the other day. There were mostly bash scripts:

| Script | Description                            |
| :---------: | -------------------------------------- |
|K_e0bcc.txt| Reference file containing important constants for a QE input|
|create_expansion_files_scf.sh|Shell script which automates the creation of 1 atom BCC primitive cells under triaxial strain|
|create_shear_files_scf.sh  |Shell script which automates the creation of 1 atom BCC primitive cells under triaxial strain|
|kp | Reference file containing important parameters for a QE input|
|pseudo| Moment Tensor Potential: a ML model of interatomic forces and energies|

These form an automation framework which serves to provide a range of initial parameters to generate different variations on an initial training set of primitive cells in triaxial strain and in shear.  

The main idea revolves around the create scripts. The scripts reference the auxiliary files, copy-pasting the constants and modifying the value to generate a set of QE inputs. I have outlined the general process of these create scripts below.

```sh
basefile = "K_e0bcc.txt";           # File name of the baseline lattice vectors
matl="K";etype="expansion_bcc";nat=1;   # Values for naming conventions

mkdir ../${matl}_${etype}_runs          # Generate an uncle directory to hold runs 

for e in `seq 0 26`; do         # Create runs with the specified offsets

a=$(echo "1+0.05*$e" | bc -l);          # For Expansion vary the length of 
b=$(echo "1+0.05*$e" | bc -l);          # the lattice vectors by 5% per degree of offset
c=$(echo "1+0.05*$e" | bc -l);

cat > top << EOF            # Generate a QE file
&control
    disk_io = 'none',
    prefix = '${matl}_expansion$e',        
    calculation ='scf',             # Self-consistet field calculation
    outdir = './out',
    pseudo_dir = '/home/zjm'            # Directory of pseudopotential
    tstress = .true.
    tprnfor = .true.
 /
 &system
    ibrav=0,            # Type of lattice = lattice vector specified
    nat=$nat,           # Number of atom in cell
    ntyp=1,             # Number of Species
    ecutwfc=60,         # Plane wave cutoff energy (Ry)
    occupations='smearing',     # Next three are smearing parameters
    smearing = 'gaussian',
    degauss = 0.01,

 /
 &electrons
    mixing_mode='plain',
    diagonalization='david',
/
 &ions
    ion_dynamics = 'bfgs'
 /
 &cell
 /
CELL_PARAMETERS
EOF

# Appends the contents of the baseline file and scales the lattice vectors
sed -n '3,5p' $basefile > cell
awk -v a=$a -v b=$b -v c=$c '{print a*$1,b*$2,c*$3}' cell > newcell

# Similar but with the pseudo and kp files (these are constants)
cat top newcell  pseudo kp > ${matl}_${etype}${e}.relax.in

# Makes a new directory to hold the new files and perfoms clean up 
mkdir ../${matl}_${etype}_runs/${matl}_${etype}${e}
mv ${matl}_${etype}${e}.relax.in ../${matl}_${etype}_runs/${matl}_${etype}${e}/
rm top cell newcell #clean-up
```
The auxiliary files are kp, pseudo, and the baseline file.

The baseline file (K_e0bcc.txt), contains the lattice vector as determined by the previous energy minimization calculations. A slight offset is introduced to prevent symmetry although it may not be strictly necessary.  It also includes the atom positions, but for shear and expansion/contraction in 1 atom BCC, any position is valid, so the origin is chosen.

```txt
1     
CELL_PARAMETERS {bohr} 
   4.83583   4.83589   4.835813             # Vector 1
  -4.83582   4.83585   4.8358231            # Vector 2
  -4.83581  -4.83586   4.83583111           # Vector 3
Atom Positions {Angstrom}
K       0.000000000   0.000000000   0.000000000
```
The kp file simply specifics the number of automatic k-points.

```txt
K_POINTS automatic
8 8 8 0 0 0
```

The pseudo file is used to define the specific information including the atomic weight and the pseudopotential to use. 

```txt
ATOMIC_SPECIES
K  39.0983 K.pbe-mt_fhi.UPF         # Potassium, atomic weight = 39.0983
ATOMIC_POSITIONS angstrom
K  0   0   0
```
Hao also recommended me a source for the pseudopotential to use which is a necessary part of QE plane-wave DFT calculations. It is the UPF file included in the table.

At this time, I was looking to start performing some QE runs on the cluster although I needed to get some more baseline measurements and familiarize myself with the job submission process on the cluster. However, the bash scripts and generation framework Hao provided me will be useful for the development of further automation scripts.

#### Friday, January 20th - Saturday, January 22nd
Final year ski trip with friends ⛷️. No real progression.

## Week 3
####  Monday, January 24th
Today, I started making further preparations for the later DFT calculations. On my local machine, I started by performing convergence testing on the parameters for the DFT calculations. This essentially involves testing different ranges of k-points and plane-wave cutoff energies and finding the lowest possible resolution that provides a reliable prediction. For a representative cell size, I used a Python script to generate these calculations, The general framework for this was developed by myself in a previous term and involves a generation script and a template. The automation idea is to write a QE input file with the highest possible level of completion that serves as the template. We leave a marker in the template and use regular expressions to replace the marker in copies of the template that we generate with the Python script. The OS package can then be used to initiate QE runs on the generated input files. Further detail is available in the General Notes.

Using the process, I used a reference cell size of BCC potassium metal at a lattice parameter of 10 Ry. I performed convergence testing with respect to k-points in a uniform distribution from 1-12, and found that 8 yielded strong convergence. The same was performed for the plane-wave cutoff energy, finding that 60 Ry worked well.  I performed the calculations for the k-points again to confirm there was no significant dependency, and upon finding the same result, settled on the below parameters for the rest of the DFT calculations.

| Parameter                | Value |
| :----------------------- | ----- |
| Plane Wave Cutoff Energy | 60 Ry |
| K-Point Count            | 8     |
| Basis Set | QE PWF Basis Sets|
| Pseudopotential          | K.pbe-mt-fhi.UPF (Packaged with QE)|

####  Tuesday, January 25th
This week the meeting with Hao got pushed to Wednesday at 10:30 AM,. This is a recurring change that we will continue for future weeks.

#### Wednesday, January 26th
For today's meeting with Hao, we started running some of the DFT scripts that he had sent me the previous week except on the cluster. This was partly to prepare the first set of training data that I had and to gain some additional experience with the cluster. We discussed what exact approach we would use for the initial training and what training scheme he had been experimenting with previously.

For the initial training set, we would use the previous bash scripts to generate a range of different 1-atom primitive cells under triaxial strain and shear. Afterwards, we would add 2-atom configurations under triaxial strain. For today, we focused on getting some jobs submitted to Slurm to get familiarized with the system. The was performed using the submit scripts although we spent much of the session getting familiarized with Slurm operation. I have my finding in the General Notes section.

#### Friday, January 28th - Saturday, January 29th
On these two days, I performed the baseline calculations to for finind

# General Notes

### The MTP interatomic model
<p style ="font-size:smaller">
Please note that the following are personal notes taken from [REF] that explain the MTP method for my personal reference and future understanding.
</p>

As an atomistic potential, the MTP method describes the energy of a system as a function of the configuration of its atoms. The MTP potential does this by considering the sum of the energies associated with each of the atoms within the system.

The energy of each atom can be defined as the weighted summations through a set of basis functions:

$$
V(n_i) = \sum_{\alpha} \xi_\alpha   \beta_\alpha
$$

The weightings, $\xi_\alpha$, are trainable parameters in the machine learning of the algorithm. $\beta_\alpha$ are the members of the basis function to the level specified as a model hyperparameter.

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

We ultimately choose the basis set of the MTP model based on a maximum level $\textrm{lev}_{\max}$ which serves as one of the model's hyperparameters. The basis set consists of all combinations of moment tensor descriptors that use the above operands to contract down to a single scalar value such that the formed expression has a level no more than the maximum value. 

Accordingly, the number of trainable parameters is heavily dependent on $\textrm{lev}_{\max}$ which scales exponentially.Now, consider the radial and angular components as characterized by the $\nu$ and $\mu$ of the particular moment tensor descriptor.

#### Radial Component of the Moment Tensor Descriptor
The radial component, $f_\mu (r_{ij},z_i,z_j)$, is described as the summation of the product of the members of the radial basis set, $Q^{(\beta)(r_{ij})}$, and the corresponding trainable radial parameters $c^{\beta} _ {\mu,z_i,z_j}$.

$$f_\mu (r_{ij},z_i,z_j) = \sum ^ {N_o} _ {\beta = 1} c^{(\beta)} _ {\mu,z_i,z_j}  Q^{(\beta)}(r_{ij})$$

The number of members of the radial basis set, $N_o$ is chosen as a model hyperparameter. The basis set is conditionally evaluated based on the chosen cutoff radius and the minimum distance between atoms in the system, using Chebyshev polynomials on the interval $[R_{min}, R_{cut}]$.

$$Q^{(\beta)}(r_{ij})=  \begin{cases}
    \phi ^{(\beta)}(|r_{ij}|) (R_{cut} - |r_{ij}|)^2& |r_{ij}| < R_{cut} \\
    0 & |r_{ij}| \geq R_{cut} 
\end{cases}$$

Where $\phi^(n)$ represents the $n$th Chebyshev polynomial. This generates a Chebyshev polynomial sequence that smoothly decays to zero at the cutoff radius.

#### Angular Component of the Moment Tensor Descriptors
The angular component is a series of $\nu$ outer products performed on the position vector between the originating atom and its $j$th neighbour. The value of $\nu$ is dependent on the exact moment tensor descriptor. This angular component works to capture the angular information between two atoms and results in a tensor whose rank is equivalent to $\nu$.

#### MTP Model Overview
Overall,  the MTP potential provides a framework atop which radial and angular components are considered. Additionally, the tensor operations performed on the descriptors maintain the model's invariance to permutations, rotations, reflection, and translation. MTP has two hyperparameters which affect its accuracy and computational cost: $N_o$ and $\textrm{lev}_{\max}$ which determine the number of basis functions expressions and the number of Chebyshev polynomials evaluated in the radial basis set. 

The trainable parameters of the model are weightings with respect to the moment tensor basis sets and the radial basis sets, $\xi_\alpha$ and $c^{\beta} _ {\mu,z_i,z_j}$ respectively. Collectively, the vector of trainable parameters will be further expressed as $x$.



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

### Narval Cluster
This all starts by connecting to Narval through my newly-minted Compute Canada account and SSH. Then, I follow the prompts, entering my password to gain access.

```
ssh -Y zjm@narval.computecanada.ca
```
Hao initially recommend me the MobaXterm terminal for automatic reconnection and SFTP (file transfer). However, I am currently running a Linux-based personal machine, and am using Tabby Terminal to perform the same.

Upon SSH'ing into Narval, I'm greeted with the home directory in one of the login nodes (narval3 in this case).

![Narval Login Node](notebook_images/narval_home.png)

This is a Linux-based terminal environment with no GUI. I have some experience with similar environments having run Linux natively for CFD purposes.  I immediately started by setting up a git repository for the project. This is to hold version-controlled scripts, output files, and the research notebook. Moreover, I can easily perform edits on my local machine and push them to Narval without SCP or SFTP. Graphical applications such are Ovito are also unavailable on Narval. The Github link is available below.

https://github.com/RichardZJM/K-MTP-training

Narval utilizes the job scheduler Slurm Workload manager for intensive computations. Only tasks smaller than 10 CPU minutes and 4 RAM are permissible on the login nodes. Slurm is essentially a priority queue for Narval's nodes. Users submit job requests for system resources in job requests. Priority can be allocated based on the relative importance allocated to the project and the principal researcher. A job request resembles the below.

```sh
#!/bin/bash                                                 
#SBATCH --account=def-belandl1                  // PI's account
#SBATCH --ntasks=1                              // Number of CPUs for job
#SBATCH --time=0-2:00 # time (DD-HH:MM)         // Estimated job duration
#SBATCH --mem-per-cpu=9G                        // Memory per CPU requirement

//Code to run on the Narval compute nodes with allocated resources
// ...
```
Additional commands for the batch can be found here. https://slurm.schedmd.com/sbatch.html

A job is defined in the format above in a text file with the .qsub extension. They are submitted to the Slurm system using the sbatch command.

```sh
sbatch <file>
```
Listed below are some additional Slurm commands which are generally useful.

```sh
squeue                  # Shows the current jobs in queue and running
squeue -u zjm           # Shows the curret jobs associated with user
scancel <job number>    # Cancels the specified job
scancel -u zjm          # Cancels all jobs associated with user
```

With this knowledge, I began by preparing the environment to run DFT, MD, and prepare MTP potentials. Quantum Espresso was already present in the Narval environment and simply needed to be loaded into the active node using the below commands. MPI is needed for parallel processing.

```sh
module load    StdEnv/2020  gcc/9.3.0  openmpi/4.0.3
module load    quantumespresso/6.6
```

Additional information about modules can also be obtained with the below commands.

```sh
module spider                           # Gives information on all available modules
module spider quantumespresso           # Same but for a specific module
module spider quantumespresso/6.6       # Same but for specific version
```
The installation of MTP and its interface with LAMMPS was accomplished with the instructions on the project's Gitlab, available below.

https://gitlab.com/ashapeev/interface-lammps-mlip-2

This mostly included cloning from the repository, running a few installation scripts, and verifying the installation.




### Slurm Job Manager
When working with high-performance computing (HPC) clusters, I often use Slurm, an open-source job scheduler and resource manager. Slurm allows me to allocate resources on the cluster, submit and schedule jobs, and monitor their progress. 

To submit a job to Slurm, I create a job script, which is a text file that specifies the resources requireand any other relevant information. and any other relevant information. Here is an example of a job script. the `#SBATCH` tags allow a user to specify the properties the job should be run with. This includes things like 

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --time=10:00
#SBATCH --ntasks=1

echo "Hello, world!"
```

In this script, I specify the job name, the output and error files, the maximum run time of the job, and the number of tasks to be run. I then run a simple command to print "Hello, world!" to the console.

To submit this job to Slurm, I use the `sbatch` command:

```bash
sbatch jobscript.sh
```

This command will submit the job to the Slurm scheduler and assign it resources based on the requirements specified in the job script.

Here are some of the most common Slurm flags I use in my job scripts:

- `--job-name`: specifies the name of the job
- `--output`: specifies the output file for the job
- `--partition`: specifies the partition to use
- `--wait`: specifies whether to resume terminal execution until job completion
- `--qos`: specifies the QOS of the job
- `--mail-user`: specifies the email address to send notifications to
- `--mail-type`: specifies the types of notifications to send. Multiple types can be specified with a comma-separated list. The available types are `BEGIN`, `END`, `FAIL`, `REQUEUE`, and `ALL`
- `--time`: specifies the maximum run time of the job (in minutes or hours:minutes)
- `--ntasks`: specifies the number of tasks to be run
- `--nodes`: specifies the number of nodes to be used
- `--cpus-per-task`: specifies the number of CPUs to be used per task
- `--mem`: specifies the amount of memory to be used per node (in megabytes or gigabytes)


Once a job has been submitted to Slurm, I can monitor its status using the `squeue` command for the user `username`.

```bash
squeue -u myusername
sq       #equivalent shorthand
watch -n 1 sq     #Runs the sq command every second for a constant monitoring
```

This command will display a list of all jobs currently running or waiting in the queue for the user `username`. I can use the `jobid` or my `username` to cancel. 

```bash
scancel jobid
scancel -u username
```

Slurm provides many other features and options that can be customized for specific job requirements. But by using these basic commands and flags, I have most of the functionality needed to run the calculations I need.

### Quantum Espresso
For DFT calculations I'm using Quantum Espresso, a powerful software suite that's widely used in electronic structure calculations and materials modeling. One of its most important modules is the PWF (Plane-Wave Basis Functions), which uses Density Functional Theory to perform highly accurate calculations of electronic structures, interatomic forces and energies. Using periodic boundary conditons, QE PWF can be used to model bulk materials and liquids.

#### Plane Wave Function
Here is an example of one of the PWF (Plane-Wave Function) input file that I use in many of my DFT runs for training set:

```txt
&control
    disk_io = 'none',
    prefix = 'K_expansion-3',
    calculation ='scf',
    outdir = './out',
    pseudo_dir = '/home/zjm'
    tstress = .true.
    tprnfor = .true.
 /
 &system
    ibrav=0,
    nat=1,
    ntyp=1,
    ecutwfc=60,
    occupations='smearing',
    smearing = 'gaussian',
    degauss = 0.01,

 /
 &electrons
    mixing_mode='plain',
    diagonalization='david',
/
 &ions
    ion_dynamics = 'bfgs'
 /
 &cell
 /
CELL_PARAMETERS
4.11046 4.11051 4.11044
-4.11045 4.11047 4.11045
-4.11044 -4.11048 4.11046
ATOMIC_SPECIES
K  39.0983 K.pbe-mt_fhi.UPF
ATOMIC_POSITIONS angstrom
K  0   0   0
K_POINTS automatic
8 8 8 0 0 0

```

Let's break down each section of the PWF input file:

| Section           | Description                                                                                                                                                                                                                                                                                 |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `&control`        | This section sets various control parameters for the calculation, such as the type of calculation to perform (`calculation`), the directory for pseudopotentials (`pseudo_dir`), and the output directory (`outdir`). The `prefix` parameter sets a prefix to use for all file names related to this calculation. `disk_io` is set to `none`, which means no data will be written to disk during the calculation. `tstress` and `tprnfor` are both set to `.true.`, which means the stress tensor and forces will be printed to the output file. |
| `&system`         | This section describes the physical system being studied, including the crystal structure (`ibrav`), number of atoms (`nat`), number of atom types (`ntyp`), energy cutoff for the plane-wave basis set (`ecutwfc`), and how the electronic occupations are treated (`occupations`, `smearing`, `degauss`). In this example, there is only one atom type (`ntyp=1`) and one atom (`nat=1`) of the element potassium (`K`). |
| `&electrons`      | This section sets parameters for the electronic self-consistent field calculation, such as the mixing mode (`mixing_mode`) and diagonalization method (`diagonalization`). In this example, the `mixing_mode` is set to `'plain'` and the `diagonalization` method is set to `'david'`. |
| `&ions`           | This section sets parameters for the ionic relaxation calculation, such as the type of dynamics (`ion_dynamics`). In this example, the Broyden-Fletcher-Goldfarb-Shanno (BFGS) method is used for ionic relaxation. |
| `&cell`           | This section sets parameters for the cell optimization calculation, such as the type of dynamics (`cell_dynamics`) and the optimization method (`press`). In this example, the cell dynamics are not explicitly set, which means the cell will not be optimized. |
| `CELL_PARAMETERS` | This section sets the cell parameters of the crystal structure in Angstroms. The three rows represent the cell vectors, with each row containing three numbers that correspond to the x, y, and z components of the vector. |
| `ATOMIC_SPECIES`  | This section lists the atomic species present in the system, along with their mass and pseudopotential file. In this example, there is only one atomic species, potassium (`K`), and the corresponding pseudopotential file is `/home/zjm/K.pbe-mt_fhi.UPF`. |
| `ATOMIC_POSITIONS` | This section lists the positions of the atoms in the crystal, in Angstroms. In this example, there is only one potassium atom located at the origin (`0 0 0`). |
| `K_POINTS`        | This section sets the k-point sampling of the Brillouin zone. In this example, an automatic mesh of 8x8x8 k-points is used (`{automatic}`). The last three numbers represent shifts in the k-point grid (in this case, no shift is applied). |

Generally, a large part of my work revolved around the automatic parameteriazation of the values within the QE input file and coordiating QE runs with the active learning loop.

#### AI MD
Later, I perform an *ab initio* MD of liquid potassium to generate a benchmark for my MTP calculations. Here is that input file which is provided by Hao. Most of it is similar, although notably there is are considerations for the number of bands and the calculation mode is set to perform a molecular dyanmics run. Additonally, Van der Waals correction is used with teh Grimme-d2 method.

```txt
&control
    restart_mode = 'restart'
    prefix = 'p1',
    calculation ='vc-md',
    outdir = './out',
    Pseudo_dir = '/global/home/hpc5146/',
    tstress = .true.
    tprnfor = .true.
    nstep=3500,
    etot_conv_thr = 1.D-5, 
    forc_conv_thr = 0.02 

 /
&system
   ibrav=0,
   nat=54,
   ntyp=1,
   nbnd=616,
   ecutwfc=50,
   degauss=0.01,
smearing = 'gaussian',
   vdw_corr='grimme-d2',
/
&electrons
mixing_mode='plain',
diagonalization='david',
electron_maxstep=5000,
/
 &ions
ion_temperature= 'andersen'
tempw = 600.0
pot_extrapolation='second-order'
wfc_extrapolation='second-order'
 /
 &cell
    cell_factor=6.0,
 /

CELL_PARAMETERS {angstrom}

15.4837740544,  0,  0
0,     15.4837740544, 0
0,  0,  15.4837740544

ATOMIC_SPECIES
K 39.098  K.pbe-mt_fhi.UPF
ATOMIC_POSITIONS {angstrom}
K 2.6406 3.19658 3.07742
...
K 12.2449 13.7772 12.8085
K_POINTS gamma
```

### LAMMPS
After configuring the necessary MLIP interface, I could start using LAMMPS which is a powerful MD simulation software. LAMMPS uses interatomic to compute the interactions between particles and can simulate systems with millions of particles over long timescales. It also supports various parallel computing on the cluster and can be easily customized with additional features and plugins. 

For most of my active learning, I perform MD parallel calculations in the NVT ensemble to aid the MTP in training under target stresses, temperatures, and phases.

```txt
units            metal
dimension        3
boundary         p p p


atom_style       atomic
lattice          bcc 5.263461147208
region           whole block 0 3 0 3 0 3 units lattice
create_box       1  whole
create_atoms     1 region whole
mass             1 39.0983

pair_style mlip /global/home/hpc5146/Projects/K-MTP-training/phase3/mtpProperties/mlip.ini
pair_coeff * *

neighbor	0.5 bin
neigh_modify    every 1 delay 5 check yes

timestep	0.001

fix		1 all nve
fix		2 all langevin 300 300 0.1 826234 zero yes

thermo_style    custom step temp 
thermo 1000


run             100000
reset_timestep  0
```
Here is the coresponding breakdown of each of the important commands used inside the LAMMPS input. 

| Section | Explanation |
| --- | --- |
| `units` | Sets the units of measurement for the simulation to "metal" |
| `dimension` | Sets the number of dimensions to 3 |
| `boundary` | Sets the boundary conditions for the simulation to periodic in all directions (`p p p`) |
| `atom_style` | Specifies that the atoms in the simulation are treated as point particles (`atomic`) |
| `lattice` | Defines the type of lattice structure to be used in the simulation (`bcc`) and sets the lattice constant (`5.263461147208`) in units specified by `units`. |
| `region` | Defines a rectangular region (`block`) within the simulation cell, with lower and upper bounds specified in lattice units (`0 3 0 3 0 3`), and assigns it a name (`whole`) |
| `create_box` | Creates a simulation box with one type of atom (`1`) and includes the `whole` region defined previously |
| `create_atoms` | Places atoms of the type defined in `create_box` within the `whole` region |
| `mass` | Sets the mass of the atoms of type `1` to `39.0983` in units specified by `units` |
| `pair_style` | Specifies the interatomic potential to be used in the simulation (`mlip`) and the path to the `.ini` file containing parameters for the potential |
| `pair_coeff` | Sets the parameters for the interatomic potential for all pairs of atom types in the simulation (`* *`) |
| `neighbor` | Sets the skin radius for the neighbor list generation to `0.5` in units specified by `units` and uses a binning algorithm |
| `neigh_modify` | Consider modifying neighbor list every `1` timestep and with `5` timesteps minimum spacingS|
| `timestep` | Sets the timestep size to `0.001` in units specified by `units` |
| `fix` | Applies a computational "fix" to all atoms (`all`) of the simulation with ID `1` and specifies the integration algorithm (`nve`) and thermostat (`langevin`) parameters |
| `thermo_style` | Specifies the format for output of thermodynamic data (`custom`) and the variables to output (`step` and `temp`) |
| `thermo` | Sets the frequency of thermodynamic output to every `1000` timesteps |
| `run` | Runs the simulation for `100000` timesteps |
| `reset_timestep` | Resets the timestep counter to `0` |

Much for QE inputs, much of my work revolved around scripting the generation of the input files based on the phase of the training scheme.

### MLIP
The MLIP package is the practical implementation of the MTP that I used for this project. Here, I note the most important points of the MLIP's practical usage.

#### MLIP commands



### Preparing the first DFT calculations

With the environment configured, I began to assemble the initial DFT training dataset. The first step is to utilize Quantum Expresso to determine the lattice parameter of BCC potassium from this value. This represents a realistic baseline from which the initial dataset can be built around. We start by determining DFT parameters which produce well-converged results, including the plane wave cutoff energy and the number of uniform k-points. From previous experience with convergence testing with Potassium in Quantum Espresso, I settled on the following parameters for all future Potassium DFT simulations.

| Parameter                | Value |
| :----------------------- | ----- |
| Plane Wave Cutoff Energy | 60 Ry |
| K-Point Count            | 8     |
| Pseudopotential          | K.pbe-mt-fhi.UPF 

Pseudopotential Source: https://github.com/buck54321/pyspresso/tree/master/pseudo

Referencing the experimental value of BCC potassium's lattice parameter, I locally performed ten DFT simulations whose lattice parameters surrounded the experimental value. This provided me with corresponding system energies which were then minimized. They were fed into Quantum Espresso ev.x's 2nd order Birch fitting function to predict a lattice parameter of **9.67166 Bohr.** Generally, when working on extending DFT calculations, it is important to use consistent simulation parameters. However, for this case, where the goal is to generate a baseline, the experimental lattice parameter would have probably worked. 

In any case, using the reference lattice parameter, I moved to perform bulk simulation on Narval. I received several scripts from Hao from which I adapted my first set of jobs for periodic potassium cells under shear and hydrostatic expansion/compression. These are all 1-atom simulation cells used to form a simpler initial training set. The following are my notes on the important points of the scripts' function. 

The create script is perhaps the most important one, responsible for the generation of the Quantum Espresso instructions which Slurm will schedule. The full script is available on Github, although here are the important points (some lines are omitted).

```sh
basefile = "K_e0bcc.txt";           # File name of the baseline lattice vectors
matl="K";etype="expansion_bcc";nat=1;   # Values for naming conventions

mkdir ../${matl}_${etype}_runs          # Generate an uncle directory to hold runs 

for e in `seq 0 26`; do         # Create runs with the specified offsets

a=$(echo "1+0.05*$e" | bc -l);          # For Expansion vary the length of 
b=$(echo "1+0.05*$e" | bc -l);          # the lattice vectors by 5% per degree of offset
c=$(echo "1+0.05*$e" | bc -l);

cat > top << EOF            # Generate a QE file
&control
    disk_io = 'none',
    prefix = '${matl}_expansion$e',        
    calculation ='scf',             # Self-consistet field calculation
    outdir = './out',
    pseudo_dir = '/home/zjm'            # Directory of pseudopotential
    tstress = .true.
    tprnfor = .true.
 /
 &system
    ibrav=0,            # Type of lattice = lattice vector specified
    nat=$nat,           # Number of atom in cell
    ntyp=1,             # Number of Species
    ecutwfc=60,         # Plane wave cutoff energy (Ry)
    occupations='smearing',     # Next three are smearing parameters
    smearing = 'gaussian',
    degauss = 0.01,

 /
 &electrons
    mixing_mode='plain',
    diagonalization='david',
/
 &ions
    ion_dynamics = 'bfgs'
 /
 &cell
 /
CELL_PARAMETERS
EOF

# Appends the contents of the baseline file and scales the lattice vectors
sed -n '3,5p' $basefile > cell
awk -v a=$a -v b=$b -v c=$c '{print a*$1,b*$2,c*$3}' cell > newcell

# Similar but with the pseudo and kp files (these are constants)
cat top newcell  pseudo kp > ${matl}_${etype}${e}.relax.in

# Makes a new directory to hold the new files and perfoms clean up 
mkdir ../${matl}_${etype}_runs/${matl}_${etype}${e}
mv ${matl}_${etype}${e}.relax.in ../${matl}_${etype}_runs/${matl}_${etype}${e}/
rm top cell newcell #clean-up
```

Overall, the scripts act very much like the previous Python scripts (shutil) I had used for automating QE runs in MECH 868. There is essentially a master file that retrieves values from auxiliary files and fuses them together to form members of a bulk run.

Those auxiliary files are kp, pseudo, and the baseline file.

The baseline file (K_e0bcc.txt), contains the lattice vector as determined by the previous energy minimization calculations. A slight offset is introduced to prevent symmetry although it may not be strictly necessary.  It also includes the atom positions, but for shear and expansion/contraction in 1 atom BCC, any position is valid, so the origin is chosen.

```txt
1     
CELL_PARAMETERS {bohr} 
   4.83583   4.83589   4.835813             # Vector 1
  -4.83582   4.83585   4.8358231            # Vector 2
  -4.83581  -4.83586   4.83583111           # Vector 3
Atom Positions {Angstrom}
K       0.000000000   0.000000000   0.000000000
```
The kp file simply specifics the number of automatic k-points.

```txt
K_POINTS automatic
8 8 8 0 0 0
```

The pseudo file is used to define the specific information including the atomic weight and the pseudopotential to use. 

```txt
ATOMIC_SPECIES
K  39.0983 K.pbe-mt_fhi.UPF         # Potassium, atomic weight = 39.0983
ATOMIC_POSITIONS angstrom
K  0   0   0
```

Overall, after using all these files, we end up with an uncle directory which contains sub-directories which each contain a QE input file for one of the specified expansion/contraction levels.

A very similar process is used to generate the input files for the shear calculation. The difference is the create files script. Instead of modifying the scale of all of the lattice vectors, only the v1 vector is modified to change the shape of the cell. 

```sh

# ...

for e in `seq 1 50`; do

a=$(echo "0.1*$e" | bc -l);      # Only the a parameter is variable
b=$(echo "1+0*$e" | bc -l);
c=$(echo "1+0*$e" | bc -l);

# ...
# The last two parameters of the v1 lattice vector are modified,
# Results in the shear deformation of the cell.
awk -v a=$a -v b=$b -v c=$c '{print $1,a+$2,$3+a}' cell > newcell   

# ...
```
Once the QE job files are generated, the submit bulk run scripts are used to pass the jobs to the Slurm manager. For these single-atom simulations, a single core for twenty minutes should be more than sufficient although the resource requirements will increase for more complex configurations.

```sh
#!/bin/bash
matl="K";etype="expansion_bcc";     # Constants for job name / type

for e in `seq 0 26`; do      # Which jobs to submit

# Generate a Slurm job file
cat > runscript << EOF

#!/bin/bash
#SBATCH --account=def-belandl1
#SBATCH --ntasks=1                     # 1 core is fine
#SBATCH --time=0-2:00 # time (DD-HH:MM)            # 2 hours is probably excessive
#SBATCH --mem-per-cpu=9G            # Probably don't need this much RAM

# Load Quantum Espresso on the Compute Node(s)

module load    StdEnv/2020  gcc/9.3.0  openmpi/4.0.3
module load    quantumespresso/6.6

# Navigate to the QE input
cd /home/zjm/scratch/K-MTP-training/initial_dft_dataset_sim_files/${matl}_${etype}_runs/${matl}_${etype}${e}

# Run with parallel processing on 1 CPU
mpirun -np 1 pw.x < ${matl}_${etype}${e}.relax.in > ../../output/${matl}_${etype}${e}.relax.out

EOF

cp runscript ../${matl}_${etype}_runs/${matl}_${etype}${e}/job_${matl}_${etype}${e}.qsub

# Submit the Slurm job to Slurm
sbatch ../${matl}_${etype}_runs/${matl}_${etype}${e}/job_${matl}_${etype}${e}.qsub
done
rm runscript #clean-up
```

The submission script for shears is the same script with variations on the for loop and the job name constants.

Overall, this gives a framework through which we can easily create a training dataset of DFT data. While this iteration is limited to 1 atom, it should be fairly trivial to modify it for more atoms. I have prior experience doing so although it was done with Python, a language I'm a bit more familiar with.

The rest of the week was mostly spent running simulations and Narval to better familiarize with the system and prepare various samples for the initial training of the MTP.

## Week 3

Week 3 started by reconfiguring and troubleshooting the environment on the Narval setup to start running the MTP.  In the previous weeks, the MLIP interface had been installed improperly although it hadn't been detected due to the verification script mostly focused on confirming that LAMMPs itself had been installed correctly. As a personal note, in the future when I may need to reinstall the MTP interface package, the package must be cloned from the repository in its directory (ie. not in the same directory as the MLIP package). The library packages are created in the lib folder of the MLIP package and must be copied into the interface package manually. Additionally, the install script for the interface takes several minutes to run and produces detailed logs. This is important to verify as failure to install the interface may pass the validation script for LAMMPS. 

After, resolving issues with the interface, I began to assemble the first training set. Using the shell script-based process obtained in the previous week, I was able to generate quantum mechanical datasets for specified ranges of strains and shears relative to the unstrained baseline cell. The question then became one of which method would provide the best response in the trained potential. Although later retraining with more complex configurations would be necessary to capture the behaviour of potassium, a strong starting point could potentially accelerate the learning process.

However, I first started with a relatively arbitrary distribution of strains and shears to understand the process a bit better.  The process is documented below.

### Format of the MTP File

First, DFT the results of the DFT calculation are compiled into a single output folder for easier manipulaiton. At this stage, the user needs to select the hyperparameters of the potential that they wish to train ($\text{lev}_{\max}$). As previously explained, the level of the potential and the Chebyshev polynomial in the radial basis sets, are two of the most important hyperparameters. The structure of an MTP and the current values of its trainable parameters are dictated in a text file. In the passive training process, the parameters are updated based on the optimization of the energy, force, and stress errors with respect to the training set. The MLIP package includes several untrained potentials which act as a starting point. 

Using a $\text{lev}_{max}$ of 8 and 8 members in the radial basis, the potential file resembles the following:

```sh
MTP
version = 1.1.0
potential_name = MTP1m
species_count = 1       #Number of species (just K for me)
potential_tag =
radial_basis_type = RBChebyshev
        min_dist = 2       # Minimum cutoff distance (angstroms)
        max_dist = 5       # Maximal cutoff distance (angstrongs)
        radial_basis_size = 8       #Number of Chebyshev Polynomials in basis set
        radial_funcs_count = 2
alpha_moments_count = 18         # Number of basis functions (combinations of moment tensor descriptors)
alpha_index_basic_count = 11

# The rest are just the values of the trainable parameters
alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 2, 0, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {0, 0, 2, 0}, {0, 0, 1, 1}, {0, 0, 0, 2}, {1, 0, 0, 0}}
alpha_index_times_count = 14
alpha_index_times = {{0, 0, 1, 11}, {1, 1, 1, 12}, {2, 2, 1, 12}, {3, 3, 1, 12}, {4, 4, 1, 13}, {5, 5, 2, 13}, {6, 6, 2, 13}, {7, 7, 1, 13}, {8, 8, 2, 13}, {9, 9, 1, 13}, {0, 10, 1, 14}, {0, 11, 1, 15}, {0, 12, 1, 16}, {0, 15, 1, 17}}
alpha_scalar_moments = 9
alpha_moment_mapping = {0, 10, 11, 12, 13, 14, 15, 16, 17}
```

This setup yields 26 trainable parameters.

### Format of Atomic Configurations

Next, some processing must be applied to the QE outputs to translate the training information into a configuration file which is readable by the MLIP package. This is because, while there is an interface with LAMMPs, the MLIP package makes no assumptions on which quantum chemistry program is used to generate the training sets. Specifically, the training data needs to be assembled with the atomic positions and the resultant energy, force, and stress. The form is as follows, where a list of these configurations is assembled in the text file. 

```sh
BEGIN_CFG
 Size          # Number of atoms
    1
 Supercell        # Parameters of the lattice vector (for periodic boundaries)
         2.226339      2.226339      2.226339
        -2.226339      2.226339      2.226339
        -2.226339     -2.226339      2.226339
# List of atom information
 AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz
             1    0       0.000000      0.000000      0.000000     0.000000    0.000000    0.000000         
 Energy
        -13.805967180600         # Energy DFT values
 PlusStress:  xx          yy          zz          yz          xz          xy
         0.99491     0.99491     0.99491    0.00000     0.00000    0.00000        #Stress DFT values
 Feature   EFS_by       Qe
 Feature   mindist      3.856132       
END_CFG
```
To convert from the form of a QE output to the configuration format python script is used to parse through the QE output. It searches through each one, accounting for configuration with more than 1 atom automatically. The current version, QE-OUTPUT.py, was previously developed by Hao as there is little difference between the functionality he needed and what I need for this step. One point of modification involved the Atom Data id tag. There was initially an issue where the type tag was enumerated from 1 instead of 0, which caused issues with the MTP training which was expecting a single species with type 0.

Afterwards, the training of the MTP can be initiated using the mlp binary file within the mlip package. The command is as follows:

```sh
/home/zjm/mlip-2/bin/mlp train 08.mtp mlip_input.cfg --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=pot.mtp
```

Since this is a long and highly specific command which I will often use for the rest of the project, I decided to create an MTP command reference for easy access. The above command and MTP future will be stored and explained there instead. It should be attached to this package or is available here: https://github.com/RichardZJM/K-MTP-training/blob/master/mtpCommands.md.

This week, I focused on training 

# References
https://iopscience.iop.org/article/10.1088/2632-2153/abc9fe


/global/home/hpc5146/mlip-2/bin/mlp train 08.mtp train.cfg --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=pot.mtp


Week4-5

1st generate two-atom configurations (hydrostatic expansion or compression, within ~5% strain, temperatures (100K)). later we add data for high temperatures (400K, 200K)

Using python scripts

2nd run MD simulations with active learning mode. (check each configuration to see whether it is risky) Nothing we can do

Setup a new state.als file using a  new command:
/global/home/hpc5146/mlip-2/bin/mlp calc-grade pot.mtp train.cfg train.cfg out.cfg --als-filename=state.als
This file path us tbe specified in the MTP MD config ini file

Create multiple MD files.
Change the input data to different strains (hydrostatic compression and expansion aro 5%) at two different temps above melt and two below
This introduces different interactions t differnt ranges but its not too unrealistic
Run these simulations the active learning enabled
The risky configurations will be save in a preselected.cfg file (autogenerated for each run)



3rd From step 2, we have a preselected.cfg file, in which every risky configuration is included.Nothing we can do

Don't do anything (just intermeidate  step)

4th, run a command to check whether all the configurations in preselected.cfg are necessary. Nothing we can do

/global/home/hpc5146/mlip-2/bin/mlp select-add /global/home/hpc5146/Projects/K-MTP-training/phase2/mdLearning/pot.mtp train.cfg ../activeLearningDFT/preselected.cfg diff.cfg --als-filename=state.als

We run a check.sh on the folder.
This visits all the sister directory which hold the MD runs
It generates a diff.cfg file which has the representative configs for each preselected run.

(Try combined the preselected instead of creating seperate diff.cfgs ???)


5th create DFT input files and run DFT. 
Run DFT on each of the configurations generated by the diff.cfg

6th add DFT result into our training data set and retrain the potential.
Run the passive training on the new set of union of the current set and the new results from the diff.cfg dft results/


aim for average energy per atom of 0.01-0.02
aim for average force of  0.1-0.2


Week 1:
My project surrounds the training of machine learning interatomic potentials using the Moment Tensor Potential (MTP) model for potassium metal. I started the week by meeting Hao Sun, a postdoctoral researcher who had done previous work with MTP training and setting up weekly meetings for his guidance. I spent this first week understanding the model which uses linear regression on basis sets consisting of moment tensor descriptors. 

Week 2:
This week, I mostly worked on setting the groundwork on 


Certainly! Here are some common Slurm flags for email notifications:



To use these flags, simply add them to your job script or include them in your `sbatch` command. Here's an example:

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --mail-user=myemail@example.com
#SBATCH --mail-type=ALL
#SBATCH --mail-subject="Job Results"

echo "Hello, world!"
```

With these flags, Slurm will send an email notification to `myemail@example.com` for all job events (`BEGIN`, `END`, `FAIL`, and `REQUEUE`) with the subject line "Job Results". You can customize the email notification settings to fit your specific needs.