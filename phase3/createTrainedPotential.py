import shutil
import re
import os
import numpy as np
import sys
import json
import subprocess

# This script generates the a fully trained mtp file from a starting point
# It is designed to be automated on an HPC enviroment which runs Slurm workload manager

# The process is as follows:
# First, generate a intial training set on which forms a basis on which we can apply active learning
# This inital training set is based on 1 atom expansion and 1 atom shear. 
# Additionally, two atom configurations with arbitrary atom positions are used to proviide a more complicated training set.
# Afterwards, we run active learning molecular dynamics on liquid and solid cases under different strains and temperatures to get new cases to learn from
# We start this with 1 atom configurations, then two atom configurations, and so on, increasing until we reach a full set of runs with no more required training

# As QE and LAMMPS runs must be submitted through the coumpute nodes, we need to record when the last of a QE run is completed. 
# This done by adding a completion file to a directory. The end of each run will count the number of completed file to verify whether all runs have completed.

# A configuration file can be specified from the user to get model hyperparameters

configFile = "config.json"           # First system argument, generates and performs run if specified
try:
     if sys.argv[1] != None: configFile = sys.argv[1];
except:
    pass

params = {}
try:
    f = open(configFile)
    params = json.load(f)
except:
    raise Exception(configFile + "not found or not in JSON format.")

dryRun = False           # Second argument. Performs dry run.
try:
     if sys.argv[2] == "dry": 
         dryRun = True;
         print("Performing Dry Run")
except:
    pass

rootFolder = os.getcwd()    
# If there are already inital DFT results, simply pass and continue with the learning step
DFToutputFolder = rootFolder + "/outputDFT"
if not os.path.exists(DFToutputFolder): os.mkdir(DFToutputFolder)
else: 
    pass
    # RUN NEXT PART OF THE SCRIPT
    # quit()

# Try to look for missing values in the dict or issues in the formating.

initialGenerationFolder = rootFolder + "/initialGenerationDFT"   # First we generate a folder to hold all the inital DFT runs
if not os.path.exists(initialGenerationFolder): os.mkdir(initialGenerationFolder)

DFT1AtomStrainFolder = initialGenerationFolder + "/1AtomDFTStrain"             #Same for all the different types of DFT runs
DFT1AtomShearFolder = initialGenerationFolder + "/1AtomDFTShear"
DFT2AtomStrainFolder = initialGenerationFolder + "/2AtomDFTStrain"

if not os.path.exists(DFT1AtomStrainFolder): os.mkdir(DFT1AtomStrainFolder)
if not os.path.exists(DFT1AtomShearFolder): os.mkdir(DFT1AtomShearFolder)
if not os.path.exists(DFT2AtomStrainFolder): os.mkdir(DFT2AtomStrainFolder)

# Now. we generate specifications of each run we will be making

DFT1AtomStrains = np.arange(params["1AtomDFTStrainRange"][0],params["1AtomDFTStrainRange"][1],params["1AtomDFTStrainStep"])
DFT1AtomShears = np.arange(params["1AtomDFTShearRange"][0],params["1AtomDFTShearRange"][1],params["1AtomDFTShearStep"])
DFT2AtomStrains = np.arange(params["2AtomDFTStrainRange"][0],params["2AtomDFTStrainRange"][1],params["2AtomDFTStrainStep"])

templatesFolder = rootFolder + "/templates"
template1AtomStrainDFT = templatesFolder + "/1AtomStrainDFT.in"
template1AtomShearDFT = templatesFolder + "/1AtomShearDFT.in"
template2AtomStrainDFT = templatesFolder + "/2AtomStrainDFT.in"
templateDFTJob = templatesFolder + "/jobInitialDFT.qsub"

scriptsFolder = rootFolder + "/pythonScripts"               
followUpScript = scriptsFolder + "/runMDCases.py"

subprocesses = []


# Generate and submit the 1Atom Strain runs
for strain in DFT1AtomStrains:
    folderName = DFT1AtomStrainFolder + "/1AtomDFTstrain" + str(round(strain,2))
    inputName = folderName + "/1AtomDFTstrain" + str(round(strain,2)) + ".in"
    jobName = folderName + "/1AtomDFTstrain" + str(round(strain,2)) + ".qsub"
    outputName = DFToutputFolder + "/1AtomDFTstrain" + str(round(strain,2)) + ".out"   
    
    if not os.path.exists(folderName): os.mkdir(folderName)
    
    shutil.copyfile(template1AtomStrainDFT, inputName)
    shutil.copyfile(templateDFTJob, jobName)
    
        # Make modifications to the QE input using regex substitutions
    with open (inputName, 'r+' ) as f:
        content = f.read()
        contentNew = re.sub("\$aaa", str(strain * params["baseLatticeParameter"] /2), content)      #substitute lattice vector marker with the lattice vector
        contentNew = re.sub("\$pseudo_dir", params["pseudopotentialDirectory"], contentNew)      
        contentNew = re.sub("\$pseudo", params["pseudopotential"], contentNew)  
        contentNew = re.sub("\$out", folderName, contentNew)  
        f.seek(0)
        f.write(contentNew)
        f.truncate()
    
    with open (jobName, 'r+' ) as f:
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
        contentNew = re.sub("\$next", followUpScript, contentNew)
        contentNew = re.sub("\$root", rootFolder, contentNew)
        f.seek(0)
        f.write(contentNew)
        f.truncate()
        
if (not dryRun): subprocesses.append(subprocess.Popen(["sbatch",  jobName]))  

for shear in DFT1AtomShears:
    folderName = DFT1AtomShearFolder + "/1AtomDFTshear" + str(round(shear,2))
    inputName = folderName + "/1AtomDFTshear" + str(round(shear,2)) + ".in"
    jobName = folderName + "/1AtomDFTshear" + str(round(shear,2)) + ".qsub"
    outputName = DFToutputFolder + "/1AtomDFTshear" + str(round(shear,2)) + ".out"   
    
    if not os.path.exists(folderName): os.mkdir(folderName)
    
    shutil.copyfile(template1AtomShearDFT, inputName)
    shutil.copyfile(templateDFTJob, jobName)
    
        # Make modifications to the QE input using regex substitutions
    with open (inputName, 'r+' ) as f:
        content = f.read()
        contentNew = re.sub("\$aaa", str(shear * params["baseLatticeParameter"] /2), content)      #substitute lattice vector marker with the lattice vector
        contentNew = re.sub("\$bbb", str(params["baseLatticeParameter"] /2), contentNew)      #substitute lattice vector marker with the lattice vector
        contentNew = re.sub("\$pseudo_dir", params["pseudopotentialDirectory"], contentNew)      
        contentNew = re.sub("\$pseudo", params["pseudopotential"], contentNew)  
        contentNew = re.sub("\$out", folderName, contentNew)  
        f.seek(0)
        f.write(contentNew)
        f.truncate()
    
    with open (jobName, 'r+' ) as f:
        content = f.read()
        contentNew = re.sub("\$job", "Shear" + str(strain), content) 
        contentNew = re.sub("\$outfile", folderName + "/out.run",contentNew) 
        contentNew = re.sub("\$account", params["slurmParam"]["account"], contentNew) 
        contentNew = re.sub("\$partition", params["slurmParam"]["partition"], contentNew) 
        contentNew = re.sub("\$qos", params["slurmParam"]["qos"], contentNew) 
        contentNew = re.sub("\$cpus", params["dftJobParam"]["cpus"], contentNew) 
        contentNew = re.sub("\$time", params["dftJobParam"]["time"], contentNew) 
        contentNew = re.sub("\$in", inputName, contentNew)      
        contentNew = re.sub("\$out", outputName, contentNew)
        contentNew = re.sub("\$next", followUpScript, contentNew)
        contentNew = re.sub("\$root", rootFolder, contentNew)
        f.seek(0)
        f.write(contentNew)
        f.truncate()
    
        if (not dryRun): subprocesses.append(subprocess.Popen(["sbatch",  jobName]))  
    
for strain in DFT2AtomStrains:
        folderName = DFT2AtomStrainFolder + "/2AtomDFTstrain" + str(round(strain,2))
        inputName = folderName + "/2AtomDFTstrain" + str(round(strain,2)) + ".in"
        jobName = folderName + "/2AtomDFTstrain" + str(round(strain,2)) + ".qsub"
        outputName = DFToutputFolder + "/2AtomDFTstrain" + str(round(strain,2)) + ".out"   
        
        if not os.path.exists(folderName): os.mkdir(folderName)
        
        shutil.copyfile(template2AtomStrainDFT, inputName)
        shutil.copyfile(templateDFTJob, jobName)
        
            # Make modifications to the QE input using regex substitutions
        with open (inputName, 'r+' ) as f:
            content = f.read()
            contentNew = re.sub("\$aaa", str(strain * params["baseLatticeParameter"] /2), content)      #substitute lattice vector marker with the lattice vector
            contentNew = re.sub("\$pseudo_dir", params["pseudopotentialDirectory"], contentNew)      
            contentNew = re.sub("\$pseudo", params["pseudopotential"], contentNew)  
            contentNew = re.sub("\$out", folderName, contentNew)  
            f.seek(0)
            f.write(contentNew)
            f.truncate()
        
        with open (jobName, 'r+' ) as f:
            content = f.read()
            contentNew = re.sub("\$job", "2Strain" + str(strain), content) 
            contentNew = re.sub("\$outfile", folderName + "/out.run",contentNew) 
            contentNew = re.sub("\$account", params["slurmParam"]["account"], contentNew) 
            contentNew = re.sub("\$partition", params["slurmParam"]["partition"], contentNew) 
            contentNew = re.sub("\$qos", params["slurmParam"]["qos"], contentNew) 
            contentNew = re.sub("\$cpus", params["dftJobParam"]["cpus"], contentNew) 
            contentNew = re.sub("\$time", params["dftJobParam"]["time"], contentNew) 
            contentNew = re.sub("\$in", inputName, contentNew)      
            contentNew = re.sub("\$out", outputName, contentNew)
            contentNew = re.sub("\$next", followUpScript, contentNew)
            contentNew = re.sub("\$root", rootFolder, contentNew)
            f.seek(0)
            f.write(contentNew)
            f.truncate()
        
        if (not dryRun): subprocesses.append(subprocess.Popen(["sbatch",  jobName]))  

print(subprocesses)

exitCodes = [p.wait() for p in subprocesses]
print(exitCodes)
# except Exception as e:
#     print("An error has occur during the generation of the initial files. Verify the formating of your JSON config file.")
#     raise Exception(e)

