import shutil
import re
import os
import sys
import numpy as np

# This program generates a series of 1 atom DFT expansion  runs files which based on a set of user-defined strains and temperatures
# This is to generate an initial training set
# This is done by copying and modifying template files

performRun = False           # First system argument, generates and performs run if specified
try:
     if sys.argv[1] == "run": performRun = True  
except:
    pass

dftRunTemplateLocation = "../../../1AtomDFTExpansion/templateExpansionDFTRun.in"          #location of dft run, data input, and job templates 
jobTemplateLocation = "../../../1AtomDFTExpansion/templateExpansionDFTRunSubmit.qsub"

baseline = 4.83583;                 # lattice parameter /2 of K  (DFT) calculated (bohr)
strains = np.arange(0.85, 1.16, 0.02)               #strains we wish to consider

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







# 4.1574

# 4.52
