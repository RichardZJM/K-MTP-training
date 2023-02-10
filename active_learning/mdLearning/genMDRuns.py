import shutil
import re
import os
import sys
import numpy as np

# This program generates a series of 2 atom MD runs files which based on a set of user-defined strains and temperatures
# The purpose to to enable active-learning and generate a new set of preselected configurations to train off of
# This is done by copying and modifying template files

performRun = False           # First system argument, generates and performs run if specified
try:
     if sys.argv[1] == "run": performRun = True  
except:
    pass

mdRunTemplateLocation = "../../mdLearning/templateMDRun.in"          #location of md run, data input, and job templates 
dataTemplateLocation = "../../mdLearning/templateData.dat"
jobTemplateLocation = "../../mdLearning/templateMDRunSubmit.qsub"

baseline = 5.118022063;                 # lattice parameter of K  (DFT) calculated
strains = np.arange(0.95, 1.06, 0.02)               #strains we wish to consider
temperatures = [100, 300 , 400, 600 ]           # Temperature

#Make and prepare a new directory to hold all runs if needed 
os.chdir("../")
os.system(" mkdir MDRuns")
os.chdir("./MDRuns")

for strain in strains:
    for temperature in temperatures:
        
        # Generate the necessary folder and file names
        folderName = "mdT" + str (temperature) + "S" + str(strain)
        inputName = "T" + str (temperature) + "S" + str(strain) + ".lmp"
        dataName = "T" + str (temperature) + "S" + str(strain) + ".dat"
        jobName = "T" + str (temperature) + "S" + str(strain) + ".qsub"
        outputName = "T" + str (temperature) + "S" + str(strain) + ".out"
        
        # Generate a new directory for each MD Run and navigate to it
        os.system("mkdir "+folderName)
        os.chdir(folderName)
        
        # Copy the templates for the LAMMPS input and data files
        shutil.copyfile(mdRunTemplateLocation, inputName)
        shutil.copyfile(dataTemplateLocation, dataName)
        shutil.copyfile(jobTemplateLocation, jobName)
        
        # Make modifications to the LAMMPS input using regex substitutions
        with open (inputName, 'r+' ) as f:
            content = f.read()
            contentNew = re.sub("\$ttt", str(temperature), content)      #substitute temperature marker with the temperature
            contentNew = re.sub("\$ddd", dataName, contentNew)       #subsitute data file name marker with the data file name
            f.seek(0)
            f.write(contentNew)
            f.truncate()
            
        # Make modifications to the data file using regex substitutions
        with open (dataName, 'r+' ) as f:
            content = f.read()
            contentNew = re.sub("\$aaa", str(baseline*strain), content)      #substitute lattice parameter marker with the strained cell dimensions
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
