# MECH 461 Research Notebook
## Introduction
This is the research notebook for the dataset generation of moment tensor potentials (MTP) for potassium, training of different potential, and the subsequent validation of those potentials in molecular dynamics simulations. As I developed this notebook, I there was a clear distinction between some of the notes I was taking. One category is the day-by-day breakdown of my activities which may be useful in keeping records and tracing my steps although the contents aren't generally worth references. The other category are general notes which I may find myself using throughout the project. These include this such as code references and framework outlines. In the interest of brevity, while the general notes are developed on a day-by-day basis, the updates in the weekly breakdown merely make reference to the general notes instead of making redundant or in-depth explanations. 

Additionally, I do not work on this project everyday and the weekly breakdown reflects this although I try to make mention of extended absences from progress. My project is purely computational and thus, a lot of my work is sparse and is well distributed throughout the semester.

Please refer to my personal website and the project GitHub for the source code and further information.

https://github.com/RichardZJM/K-MTP-training
https://richardzjm.com/projects

## Terminology
A brief overview of some commonly used acronyms in the notebook. 
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
- [Weekly Breakdown](#weekly-breakdown)
  - [Week 1](#week-1)
  - [Week 2](#week-2)
  - [Week 3](#week-3)
  - [Week 4](#week-4)
  - [Week 5](#week-5)
  - [Week 6](#week-6)
  - [Reading Week and Week 7](#reading-week-and-week-7)
  - [Week 8](#week-8)
- [General Notes](#general-notes)
    - [The MTP interatomic model](#the-mtp-interatomic-model)
    - [HPC Clusters](#hpc-clusters)
    - [Slurm Job Manager](#slurm-job-manager)
    - [Quantum Espresso](#quantum-espresso)
    - [LAMMPS](#lammps)
    - [MLIP](#mlip)
    - [Practical Active Learning Procedure](#practical-active-learning-procedure)
    - [Python Scripting Key Techniques](#python-scripting-key-techniques)
    - [Preparing the first DFT calculations](#preparing-the-first-dft-calculations)
  - [Week 3](#week-3-1)
    - [Format of the MTP File](#format-of-the-mtp-file)
    - [Format of Atomic Configurations](#format-of-atomic-configurations)
- [References](#references)


# Weekly Breakdown

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


My extended notes on the MTP architecture can be found in the [General Notes](#the-mtp-interatomic-model) section.

## Week 2
#### Monday, January 16th
This week was mostly focused on getting the software environment set up on the Narval, on the cluster operated through the DRAC. Today, I started by setting up a meeting with Hao for 11:30 AM the following day, This would be a recurring meeting to check up on my progress each week.

#### Tuesday, January 17th
During the meeting with Hao, we started by connecting to Narval for the first time through SSH. This was done through the below command.

```bash
ssh -Y zjm@narval.computecanada.ca
```

Additional notes on the usage of the Narval cluster is available in the [General Notes](#hpc-clusters)

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
####  Monday, January 23th
Today, I started making further preparations for the later DFT calculations. On my local machine, I started by performing convergence testing on the parameters for the DFT calculations. This essentially involves testing different ranges of k-points and plane-wave cutoff energies and finding the lowest possible resolution that provides a reliable prediction. For a representative cell size, I used a Python script to generate these calculations, The general framework for this was developed by myself in a previous term and involves a generation script and a template. The automation idea is to write a QE input file with the highest possible level of completion that serves as the template. We leave a marker in the template and use regular expressions to replace the marker in copies of the template that we generate with the Python script. The OS package can then be used to initiate QE runs on the generated input files. Further detail is available in the General Notes.

Using the process, I used a reference cell size of BCC potassium metal at a lattice parameter of 10 Ry. I performed convergence testing with respect to k-points in a uniform distribution from 1-12, and found that 8 yielded strong convergence. The same was performed for the plane-wave cutoff energy, finding that 60 Ry worked well.  I performed the calculations for the k-points again to confirm there was no significant dependency, and upon finding the same result, settled on the below parameters for the rest of the DFT calculations.

| Parameter                | Value |
| :----------------------- | ----- |
| Plane Wave Cutoff Energy | 60 Ry |
| K-Point Count            | 8     |
| Basis Set | QE PWF Basis Sets|
| Pseudopotential          | K.pbe-mt-fhi.UPF (Packaged with QE)|

####  Tuesday, January 24th
This week the meeting with Hao got pushed to Wednesday at 10:30 AM. This is a recurring change that we will continue for future weeks.

#### Wednesday, January 25th
For today's meeting with Hao, we started running some of the DFT scripts that he had sent me the previous week except on the cluster. This was partly to prepare the first set of training data that I had and to gain some additional experience with the cluster. We discussed what exact approach we would use for the initial training and what training scheme he had been experimenting with previously.

For the initial training set, we would use the previous bash scripts to generate a range of different 1-atom primitive cells under triaxial strain and shear. Later, we would add 2-atom configurations under triaxial strain. For today, we focused on getting some jobs submitted to Slurm to get familiarized with the system. The was performed using the submit scripts although we spent much of the session getting familiarized with Slurm operation. I have my notes regarding Slurm in the [General Notes](#slurm-job-manager) section.

#### Friday, January 27th - Saturday, January 28th
Over these past two days, I performed the calculations for finding the base lattice parameter of the solid BCC potassium. For the aim of creating general-purpose MTPs which spanned both the liquid and solid phases, I wanted to target cases that would be close to realistic potassium conditions in terms of strain and temperature. Minor deviations from the base lattice parameter would serve this purpose. As we are not fitting to empirical data, but rather DFT, we need a DFT base line. Thus, I selected a range of lattice parameters around the emperical base lattice parameters: 8.8-10.5 Bohr, and performed BCC calculations in increments of 0.1 Bohr. This yielded the following data for the elastic curve of BCC potassium.

| Lattice Parameter [Bohr] | Potential Energy [Ry]|
|-------|--------|
| 8.8   | -1.02310361 |
| 8.9   | -1.02445713 |
| 9.0   | -1.02558806 |
| 9.1   | -1.02651523 |
| 9.2   | -1.02725474 |
| 9.3   | -1.02782213 |
| 9.4   | -1.02823133 |
| 9.5   | -1.02849605 |
| 9.6   | -1.02862670 |
| 9.7   | -1.02863538 |
| 9.8   | -1.02853133 |
| 9.9   | -1.02832395 |
| 10.0  | -1.02802176 |
| 10.1  | -1.02763309 |
| 10.2  | -1.02716451 |
| 10.3  | -1.02662317 |
| 10.4  | -1.02601536 |
| 10.5  | -1.02534666 |

Then, using QE's 2nd order Birch fit to the elastic curve, we achieve the following results.

| Property | Value |
|-------|--------|
| Base Lattice Parameter   | 5.11 Å|
| Bulk Modulus  | 3.9 |
| Equilibrium Volume  | 66.72  Å$^3$|

## Week 4
#### Monday, January 30th
Having prepared all the parameters needed for the lattice parameters the last week, I started experimenting with the passive training of the MTP relative to the collections of training data that I could generate. For this, I needed to convert the information that I had store in a bunch of output QE files into a form usable by the MTP, namely into a the configuration file format which is detailed further in the [General Notes](#configuration-files) section.

To convert, a Python script, `QE_OUTPUT.py`, provided by Hao is used to parse through the QE outputs. It searches through each one, accounting for configuration with more than 1 atom automatically. One point of modification from Hao's original script involved the Atom Data id tag.  However, I had issues running the training command and I decided to put it aside until the next meeting with Hao on Wednesday.

#### Wednesday, February 1st
Today, we tried to train some MTPs although we were limited by the installation. In the previous weeks, the MLIP interface had been installed improperly although it hadn't been detected due to the verification script mostly focused on confirming that LAMMPS itself had been installed correctly. As a personal note, in the future when I may need to reinstall the MTP interface package, the package must be cloned from the repository in its directory (ie. not in the same directory as the MLIP package). The library packages are created in the lib folder of the MLIP package and must be copied into the interface package manually. Additionally, the install script for the interface takes several minutes to run and produces detailed logs. This is important to verify as failure to install the interface may pass the validation script for LAMMPS. 

After, that we manage to train some different initial training sets with the`train` command. More notes on the MLIP commands is available in the [General Notes](#mlip-commands) section. There was initially an issue where the type tag was enumerated from 1 instead of 0, which caused issues with the MTP training which was expecting a single species with type 0. This ended up being a problem with the `QE_OUTPUT.py` script that Hao has provided me. His script was designed for his NaCl MTP that he was working on which had two types, which caused issues in the generation of the configuration files.

The `train` command also produces training errors when it finished training. Hao mentioned to me that there are two training errors which he pays closer attention to and offered reference values which are generally good targets.

|Training  Error|Target|
|-|-|
|Average Energy Error Per Atom|0.01 eV/Atom or less|
|Average Force Error Per Atom|0.1 eV/Å Atom or less|

#### Thursday, February 2nd - Friday, February 3rd
I was more occupied with the proposal report for my capstone project and didn't focus much on the research project as much except for review some of the literature.

#### Sunday, February 5rd - Monday, February 6th
I experimented with some different training sets for the different initial datasets. As Hao recommended, I started with a MTP level 08. Using the scripts, I played around with different initial training dataset. The script offers options that affect the range of the triaxial strains and shears that could be represented. 

I started by playing with strains, generating a training set with values from 0.85 to 1.15 in increments of 0.02. Which gave training errors well below the targeted ranges. Playing with the number of the training configurations in a given range of strains tended to have a minimal impact on the error targets. I find this quite reasonable since in cases of a small number of configurations, there are sufficient trainable parameters to have strong representation although the error doesn't quite reach zero. The MTP also seems to make good representations in cases where there are a lot of training sets. Playing with the strain range, I found that the prediction was also good in cases until the range reached around 0.65-1.35, which is also much larger than a reasonable range in any case.

The 1-atom shear configurations weren't strongly impacted by the number of training set either, although the errors increased quite a bit when the length-to-width ratio of the cell surpassed 3 which isn't very representative of a realistic bulk material either. 

## Week 5
#### Wednesday, February 8th
Today Hao, and I had a bit of a shorter session due to other commitments and we started focusing on the requirements need for rest of the project. A big part of the project would focus on performing active learning runs with the MLIP package, so it would an important part of it. I breakdown the majority of the LAMMPS input file for the run in the [General Notes](#lammps). 

After that, Hao gave me some more scripts that he had used for 2-atom configurations under triaxial strain which I could use to start expanding my dataset. We ran out of time shortly after although we agreed to meet a bit earlier the next week since it would be the last session before the reading week.

#### Friday, February 10th
Today, I started unpacking and working with the scripts that Hao had provided me with the previous session. It was pretty much like the previous scripts, and it would be simple enough to run although I was having issues maintaining a connection to the Narval login node. I would frequently lose connection which would prevent me from accessing my work for extended periods of time throughout the day. This had actually happened occasionally from some of the previous days as well although the frequency was increasing. 

Although we had started with Narval since the beginning of the project, Hao had previously mentioned that he was using CAC instead, especially since the job queue time tended to be lower too.  I searched around for a fix to the Narval issue, and ended up migrating to the CAC cluster. Most of the work today consisted of updating my Git setup and performing the migration.

#### Saturday, February 11th
Previously, my coding workflow and script modification process consisted of working on files locally and pushing with GIt onto the cluster. After that point, I SSH onto the cluster using the Linux Terminal and run the scripts and calculations using the Slurm job scheduler. The editor that I use locally is Visual Studio (VS) Code, an extensible editor developed by Microsoft. While working on the migration yesterday, I stumbled across the VS code SSH extension which is allows client to remote into a cluster with all of the VS Code functionality such as graphical file exploration, extensions, and Intellisense. It essentially, offers a near-local development environment on the cluster and saves a ton of development time.  The tutorial is available below. 

[VS Code SSH extension](https://code.visualstudio.com/docs/remote/ssh)

I spent an hour today setting it up and working out the kinks, especially setting up the SSH keys to enable automatic authorization from my local machine. The main part involved setting up the remote SSH key, this involves generating a SSH on the local machine using

```bash
ssh-keygen
```

After following the prompt, a public and private key is generated in the specified file location. In the VS code SSH config, file the preferred authentication type is specified and the location of the private key is given. 
```bash
Host login.cac.queensu.ca
  HostName login.cac.queensu.ca
  ForwardX11 yes
  User hpc5146
  PreferredAuthentications publickey
  IdentityFile /home/richa/.ssh/id_rsa
```
On the user's home directory in the cluster, the public key is added to the `known_hosts` file inside the `.ssh` folder. This allows the VS Code connection to establish without using a password. Additionally, I set the Remote SSH timeout to 60 seconds since I was having issues with receiving a response with the default 15 seconds. 

I also made a small post on the Nuclear Materials Group team channel to share my findings which could be quite useful for others.

## Week 6
#### Monday, February 13th
Having agreed to meet earlier this week due to the forthcoming reading week, Hao and I met at 11:30 AM today. Having gotten me familiarized with the last of the important individual steps the prior week, we worked on finalizing the final framework of the project. I was already familiar with the concepts behind the MTP architecture, the active learning loop although it was important to get a practical level overview of each of the steps so that I could have a concrete guide and knowledge of the techniques to digest over the reading week.

Hao outlined each of the steps that he followed and I asked questions on points that I found to be unclear. My current understanding of the process to create trained MTPs is available in the [General Notes](#practical-active-learning-procedure). This includes some modifications to the process that I think would lead to a cleaner implementation and more efficient process.

#### Tuesday, February 14th
Midterm Week. 😔 Today, it was Computer Architecture.

#### Thursday, February 16th - Friday, February 17th
Today, I was making plans for the reading week. Most I was considering how to automate the active learning process which Hao had introduced to me and that I had modified. The key problem was how I would integrate the sequential execution of bash scripts. I had one idea which would work which was to simply call a bash script with a bash script. This second script could also call another bash script...

```bash
sh script.sh
```

However, I had a few qualms with such an approach.
- Persistent data is not preserved
- Nested bash script calls are a code maintenance nightmare
- Conditional logic is painful
- The execution of the `sbatch` command terminate when the job is submitted, not when the job is completed.

I did have some workaround to some of the problems. For instance, the persistent data could perhaps be stored in a text file which each script could reference upon startup. The job scheduling issue could perhaps be worked around by having each file make a modification to a directory (ie. add a file), and when each job is finishing, I could have it count the number of files to determine whether to run a subsequent script. However, overall it's a bit of a pain to implement and use.

Here, I made the decision to switch all of my project frameworks over to Python, and language I am far more familiar with and a language which is far more powerful. Here, I made a shift from learning and experimenting towards development and automation and accordingly made a new folder on Git, titling future development under Phase 2.

I started by automating the initial generation scripts that had originally been provided to me by Hao. I show the script for 1-atom triaxial strain below and omit the rest since the framework is very similar. 

``` python
performRun = False           # First system argument, generates and performs run if specified
try:
     if sys.argv[1] == "run": performRun = True  
except:
    pass

dftRunTemplateLocation = "../../../1AtomDFTExpansion/templateExpansionDFTRun.in"          #location of dft run, data input, and job templates 
jobTemplateLocation = "../../../1AtomDFTExpansion/templateExpansionDFTRunSubmit.qsub"

baseline = 4.83583;                 # lattice parameter /2 of K  (DFT) calculated (bohr)
strains = np.arange(0.65, 1.36, 0.05)               #strains we wish to consider

#Make and prepare a new directory to hold all runs if needed 
os.chdir("../")
os.system("mkdir runs")
os.chdir("./runs")
os.system(" mkdir dftExpansion")
os.chdir("./dftExpansion")

# os.system("pwd")

for strain in strains:
    # Generate the necessary folder and file names
    folderName = "expSt" + str(round(strain,2))
    inputName = "expSt" + str(round(strain,2)) + ".in"
    jobName = "expSt" + str(round(strain,2)) + ".qsub"
    outputName = "expSt" + str(round(strain,2)) + ".out"
    
    # Generate a new directory for each dft run and navigate to it
    os.system("mkdir "+folderName)
    os.chdir(folderName)
    
    # Copy the templates for the LAMMPS input and data files
    shutil.copyfile(dftRunTemplateLocation, inputName)
    shutil.copyfile(jobTemplateLocation, jobName)
    
    # Make modifications to the QE input using regex substitutions
    with open (inputName, 'r+' ) as f:
        content = f.read()
        contentNew = re.sub("\$aaa", str(strain * baseline), content)      #substitute lattice vector marker with the lattice vector
        f.seek(0)
        f.write(contentNew)
        f.truncate()
        
    # Make modifications to the job file using regex substitutions
    with open (jobName, 'r+' ) as f:
        content = f.read()
        contentNew = re.sub("\$jjj", jobName, content)      #substitute job name marker with job name
        contentNew = re.sub("\$in", inputName, contentNew)      #substitute input name marker with input name
        contentNew = re.sub("\$out", outputName, contentNew)      #substitute output name marker with output name
        f.seek(0)
        f.write(contentNew)
        f.truncate()
    
    if performRun: os.system("sbatch " + jobName)
    
    os.chdir("../")
```

Overall, the main idea is very simple. Much like the previous bash scripts, I utilize a template for both the job submission file and the QE input files. One minor difference is that instead of having the bulk of the files written as literals in the code as was the case with Hao's code, I opt to transfer everything to the template file to create a clear distinction between the text files and the logic. Using a for loop, I iterate through each of the required strains, copy the templates into a fresh directory, and modify the required fields using RegEx substitutions. One ease of use function that I added was a system argument for troubleshooting that didn't submit the runs. Overall, this new framework clears up a lot of the issues that I was having with the bash implementation although I still don't have a perfect solution to the issue of job scheduling. However, later in the reading week, I'll hopefully have more time to find a good solution to the problem.

#### Saturday, February 18th - Saturday February 19th
Reading week has started. No progress for these last two days.

## Reading Week and Week 7
These two weeks are a complete write-off. I went back to Ottawa for the reading week. My family had arrange plans for the start of the reading week, and on the Thursday, I got my wisdom tooth surgery. I had hoped to recover for a few days and return to school and the research project although there were complications and I didn't make progress.

## Week 8
#### Monday, March 27th - Tuesday, March 28th
Getting back into the work on the project, I did additional research into the Slurm job scheduler and the interaction with Python. In doing, so I found the following post on Stack Overflow on how to pause Python until a Slurm job is finished.

[Await Slurm Run Completion](https://stackoverflow.com/questions/46427148/how-to-hold-up-a-script-until-a-slurm-job-start-with-srun-is-completely-finish)

This by itself is not that useful, but when we pair it with the Python subprocesses library, we can run Slurm Jobs in parallel and pause execution until the subprocesses have completed. 

[Awaiting Python Subprocess Completion](https://stackoverflow.com/questions/15107714/wait-process-until-all-subprocess-finish)

A further explanation and code sample is available in the [General Notes](#running-parallel-slurm-jobs-and-pausing-execution-until-all-runs-are-complete)

Having solved the last piece of the automation puzzle, I knew that I could theoretical build a fully automated script that would coordinate the complete active learning of MTPs according to a bottom-up training approach. However, I am concerned about getting the setup working on the cluster as it would rely on a large amount of small jobs. I am also not sure if there might be issues with the amount of computation is would take to fully an MTP and I was hesitant to spend to many of the school resources. 

Either way, I shift the GitHub naming to phase 3. I will not be including the full script here due to the probable length although, I will list all the techniques I use in [General Notes](#python-scripting-key-techniques). I will make sure to comment it and have it available on the GitHub.

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



#### Training
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

### HPC Clusters
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
These are the most common commands used for the MLIP along with a quick blurb. For ease of use I created and alias by adding the following alias command to the user `.bashrc`.
```sh
alias mlp="\path\to\mlp\binary"
```
If this is not done, commands can still be run by directly referring to the binary instead of the alias.

##### Train
Use for the passive training of a potential, `pot.mtp` relative to a set of training configurations `train.cfg`. Results are stored in `out.mtp`. Weighting and optimization parameters are generally fine.

```sh
mlp train pot.mtp train.cfg --energy-weight=1 --force-weight=0.01 --stress-weight=0.001 --max-iter=10000 --bfgs-conv-tol=0.000001 --trained-pot-name=out.mtp
```

##### Calculate Energy Forces and Stresses
Use, `pot.mtp` to obtain predictions of the configurations in `configs.cfg`. Results are stored in `out.cfg`.
```sh
mlp calc-efs pot.mtp configs.cfg out.cfg
```

##### Calculate Minimum Distances
Appends the minimum distance between atoms in all configurations of a `config.cfg` file. Print the global mindist.
```sh
mlp mindist configs.cfg
```

##### Selected Configurations for Further Training
Selects configurations from a set of preselected configurations,`preselected.cfg`, for an `pot.mtp` and existing training set, `train.cfg`, and returns them in `diff.cfg`. Uses an active learning state `als.als` and returns a sparsified subset of in `selected.cfg`
```sh
mlp select-add pot.mtp train.cfg preselected.cfg diff.cfg --als-filename=als.als --selected-filename=selected.cfg
```

#### Configuration Files
Use to define the information of training configurations for the MLIP model to process. Alot of the work revolves around converting to and from this format to perform DFT calculations. Specifically, the configruations data needs to be assembled with the cell parameters and atomic positions,  The resultant energy, force, and stress are needed for some command like the `train` command although it isn't need when forming predictions. The form is as follows, where a list of these configurations is assembled in the text file. 

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

#### MTP Potential Files
These files store the actual the hyperparameters of a given MTP as well as the trainable paraemeters. Most of the modification of these files are performed by the system during training although I do edit the cutoff radius sometimes. The level is define by the arrangement of the alpha lists and for a given MTP level, there are pregenerated files that ship with the MLIP package

```sh
MTP
version = 1.1.0
potential_name = MTP1m
species_count = 1       #Number of species (just K for me)
potential_tag =
radial_basis_type = RBChebyshev
        min_dist = 2       # Minimum cutoff distance (angstroms)
        max_dist = 5       # Maximal cutoff distance (angstronms)
        radial_basis_size = 8       #Number of Chebyshev Polynomials in basis set
        radial_funcs_count = 2
alpha_moments_count = 18        
alpha_index_basic_count = 11

# The rest are just the definition of the potential for a given level and  trainable parameters
alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 2, 0, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {0, 0, 2, 0}, {0, 0, 1, 1}, {0, 0, 0, 2}, {1, 0, 0, 0}}
alpha_index_times_count = 14
alpha_index_times = {{0, 0, 1, 11}, {1, 1, 1, 12}, {2, 2, 1, 12}, {3, 3, 1, 12}, {4, 4, 1, 13}, {5, 5, 2, 13}, {6, 6, 2, 13}, {7, 7, 1, 13}, {8, 8, 2, 13}, {9, 9, 1, 13}, {0, 10, 1, 14}, {0, 11, 1, 15}, {0, 12, 1, 16}, {0, 15, 1, 17}}
alpha_scalar_moments = 9
alpha_moment_mapping = {0, 10, 11, 12, 13, 14, 15, 16, 17}
```

#### MLIP INI Files
These are essentially config files which are pointed to by LAMMPS runs to determine which MTP to use and what course of action to take during a MTP MD run.

```ini
mlip-type mtp
mtp-filename pot.mtp
calculate-efs TRUE
select TRUE    #selection mode is activated
select:threshold  2.1      #Grade> 2.1 are preselected
select:threshold-break 10.0      #Grade > 10 terminate the rum 
select:save-selected  preselected.cfg     #Where to store preselected configurations
select:load-state  state.als     #active learning state is loaded from this file
```

### Practical Active Learning Procedure
This the general procedure that I follow to perform the active learning of an MTP to target an MTP that performs with in general solid-liquid prediction tasks. This includes a high-level overview although I also include practical-level notes which detail important steps and commands.

1. Generate the initial datasets
2. Run parallel MD simulations that are representative of the target representation regime
3. Compile the preselected configurations 
4. Filter out the new configurations
5. Create corresponding QE inputs and perform the DFT calculations 
6. Retrain the MTP with the expanded training dataset
7. Repeat steps 2-6, until there are no more preselected configurations
8. Expand the scale of the MD simulations, repeat step 2-7 until there is a sufficiently rich representation

#### 1. Generate the Initial Datasets
Using bash or an alternative scripting framework, first generate a range of 1-atom primitive cell configurations under a range of shears and triaxial strains. The ranges should be representative of the target configurations and the number of configurations shouldn't be to large for fear that the initial dataset makes up to large a proportion of the final training set. Around two dozen is probably okay. When training from the initial training set, a new active learning state needs to be generated using the calc-grade MLIP command. I then run Hao's script to convert all the datasets into the MLIP configuration format. Then, I train a fresh potential with the latest training configurations. 

#### 2. Run Parallel MD Runs
We then perform LAMMPS MD runs with active learning. I do many of these runs in parallel with the aim to increase the number of selected configurations in a single iteration and reduce the amount of time for full training. Additionally, I design the initial conditions to explore more of the relevant interactions of potassium. Looking to target both liquid and solid configurations I've selected four temperatures relative to the empirical melting point of potassium. This includes 2 solid temperatures and two liquid temperatures with two temperatures close to the melting point and two well within the phase. For each of these temperatures six strains between 0.95 and 1.05 are generated, yielding 24 configurations in parallel. Practically, this involves using a scripting language to generate a folder, a job submission, and an input file for each of the MD jobs. 

#### 3. Compile the Preselected Configurations
The MLIP packages generates a list of preselected configurations it encounters. While it is possible to specify which file location to store these preselected configurations, it is finicky to have multiple MD runs write to the same preselected file each time. Thus, I specify each MD run to store it's preselected configuration in the simulation. However, scripting is need afterwards to parse through the directories of the MD simulations and append each of the preselected files into the master preselected file. In the case there are no preselected files, we end the current stage of the active learning and generate new MD runs with a larger scale.

#### 3. Trim Unnecessary Configurations 
Using the master preselected file, I need to parse through the file and eliminate the configurations which are to similar which reduces the overall computational burden when compared to evaluating all the preselected configurations. This generates the new selected configurations which are available in the in `diff.cfg` file. Practically I can simply run the MLIP command select-add.

#### 5. Perform DFT calculations
In this step, I generate the needed QE inputs which correspond to the queried configurations in the `diff.cfg` files. I perform this using scripting which must parse the `diff.cfg`, extract the files and modify the templates files with the new cell size, atom count, and atom positions. Additionally, since the k-point convergence is performed is with respect to the base lattice-parameter, I must scale the k-point count based on the size of the cell to reduce the computational burden and maintain a consistent resolution. 

#### 6. Retrain
With the updated DFT outputs, I need to rerun Hao's Python script to conglomerate the QE outputs in the configuration file format. I retrain the potential based on the output.

#### 7. Repeat steps 2-6, until there are no more preselected configurations
#### 8. Expand the scale of the MD simulations, repeat step 2-7 until there is a sufficiently rich representation

### Python Scripting Key Techniques
This section contains an overview of the important techniques that I are used to construct the final automation protocol. For the sake of brevity, I do not provide a step-by-step breakdown of the final Python script (600 lines of code). The most important point for future script development are noted here, and the final Python script is fully commented and is available in the GitHub. Moreover, the automation is really just an implementation of the practical active learning procedure explained earlier using the below techniques. 

#### Retrieving User Arguments
This is useful to have the script perform different arguments based on the bash call
```python
configFile = "config.json"           # Variable to store results
try:              #Error handling in the case no user arguments are made
     if sys.argv[1] != None: configFile = sys.argv[1];            #index 0 is not used. String is returned
except:
    pass
```

#### Loading JSON Configuration file
This is useful to have more script customizability. Allows easy import of characteristics into a Python dict without requiring a significant amount of data parsing. 
```python
params = {}       # Empty dict to store results
try:
    f = open(configFile)
    params = json.load(f)        # Open and load JSIN
except:
    raise Exception(configFile + " not found or not in JSON format.")      # Error handling to prompt user of error
```

#### Absolute Folder Paths
To ensure the system works regardless of where the folder is stored and regardless of where the script is called from in the command line, we use absolute file path.
```python
rootFolder = os.path.dirname(os.path.realpath(__file__))              # Get the root folder of the Python script
DFToutputFolder = rootFolder + "/outputDFT"     # Absolute file path to the DFT output folder
```

#### Logging
I use a custom logging file to report to the user live and have useful progress report to text file.
```python
def printAndLog(message):
    now = datetime.now()         # Include the date and time in the log message
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    datedMessage = dt_string + "   " + message        # Dated string
    print(datedMessage)       # Print to console
    with open(logFile, "a") as myfile:  myfile.write(datedMessage + "\n")        # log to file

```

#### Creating Directories As Needed
Some times folders are deleted from existing runs. this creates a directory if needed and prevent errors from being thrown.
```python
if not os.path.exists(slurmRunFolder): os.mkdir(slurmRunFolder)      
```

#### Example of Job Folder, Input, and Submission Creation
This is the bread and butter of the setup. This is how the generation of input files is handled. It used a template approach when a template of the relevant input type (eg. QE input, LAMMPS input) are complete to the fullest extent possible, omitting the parameters which need to be controlled by the script.

```python
for strain in DFT1AtomStrains:
   # Important absolute path names
    folderName = DFT1AtomStrainFolder + "/1AtomDFTstrain" + str(round(strain,2))
    inputName = folderName + "/1AtomDFTstrain" + str(round(strain,2)) + ".in"
    jobName = folderName + "/1AtomDFTstrain" + str(round(strain,2)) + ".qsub"
    outputName = DFToutputFolder + "/1AtomDFTstrain" + str(round(strain,2)) + ".out"   
    
    if not os.path.exists(folderName): os.mkdir(folderName)    # Create a folder to contain the run
    
    shutil.copyfile(template1AtomStrainDFT, inputName)         # Copy the QE input template to the run folder
    shutil.copyfile(templateDFTJob, jobName)          # Copy the job submission file to the run folder 
    
       
    with open (inputName, 'r+' ) as f:     # Open the copied QE template. Make modifications to the QE input using regex substitutions
        content = f.read()
        contentNew = re.sub("\$aaa", str(round(strain * params["baseLatticeParameter"] /2,5)), content)      #substitute lattice vector marker with the lattice vector
        contentNew = re.sub("\$pseudo_dir", params["pseudopotentialDirectory"], contentNew)      
        contentNew = re.sub("\$pseudo", params["pseudopotential"], contentNew)  
        contentNew = re.sub("\$out", folderName, contentNew)  
        f.seek(0)
        f.write(contentNew)
        f.truncate()
    
    with open (jobName, 'r+' ) as f: # Open the copied job submission template. Make modifications using regex substitutions
        content = f.read()
        contentNew = re.sub("\$job", "Strain" + str(strain), content) 
        contentNew = re.sub("\$outfile", folderName + "/out.run",contentNew) 
        contentNew = re.sub("\$account", params["slurmParam"]["account"], contentNew) 
        contentNew = re.sub("\$partition", params["slurmParam"]["partition"], contentNew) 
        contentNew = re.sub("\$qos", params["slurmParam"]["qos"], contentNew) 
        contentNew = re.sub("\$cpus", params["dftJobParam"]["cpus"], contentNew) 
        contentNew = re.sub("\$time", params["dftJobParam"]["time"], contentNew) 
        contentNew = re.sub("\$in", inputName, contentNew)      
        contentNew = re.sub("\$out", outputName, contentNew)
        f.seek(0)
        f.write(contentNew)
        f.truncate()
```

#### Running Parallel Slurm Jobs and Pausing Execution Until All Runs Are Complete
The principal is fairly simple but really effective. In each of the job submission files that I want to await completion, I set a wait flag. This causes the `sbatch` command to freeze bash execution until the job is complete. In a normal loop, this would cause the system to revert to a serial execution. However, with the Python `subprocess` library, we can invoke a subroutine and monitor its progress. Thus, all we need to do is run subroutines of `sbatch` commands that queue a job with a wait flag. We pause execution until the Python scripts detects that all subprocesses have completed.

```python
subprocesses = []     # Empty list to hold subprocesses
# Append a new subprocess which run a bash command. 
subprocesses.append(subprocess.Popen(["sbatch",  jobName]))       #Bash command is a list of string that compose the command
# We can append multiple subprocesses
exitCodes = [p.wait() for p in subprocesses]        # Wait for all the initial generation to finish
# We can also read through the exit codes to see whether are unexpected results
```

#### Reading Specific Lines From a Text File
This is used everywhere to read the outputs from certain software and reformat them into formats usable by other software.

```python
with open(trainOutput, "r") as txtfile:      #Reads in a file
         lines = txtfile.readlines()      #Converts the lines of the file into a list of string
         for i,line in enumerate(lines):        #Enumerate through the list of string 
               if(line == "Energy per atom:\n"):            #Search for the line of interest
                  avgEnergyError = lines[i+3][31:-1]           # Perform operations such as store it into a variable
```

#### Walking Through All Files In A Directory
This is useful for looking through a directory for all the outputs of a bunch of DFT or MD Runs. The example below is for the compiling preselected runs.

```python
with open(preselectedConfigs,'wb') as master:
        #Walk through the tree of directories in MD Runs
        #All child directories are run files which have no further children
            for directory, subdir, files in os.walk(mdFolder):        
                if directory == mdFolder: continue;       # There is no preselected config in the parent directory of the runs so skip
                
                childPreselectedConfigName = directory + "/preselected.cfg"   
                try: 
                         #Copy the preselected files to the master preselected 
                    with open(childPreselectedConfigName,'rb') as child:
                        shutil.copyfileobj(child, master)
                    os.remove(childPreselectedConfigName)
                except:
                    completedRuns += 1
```


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