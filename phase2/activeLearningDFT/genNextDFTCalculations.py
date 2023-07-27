import shutil
import re
import os
import numpy as np
import sys
from datetime import datetime
from random import random

#This script complies all the individual preselected config scripts in the md runs parent folder and complies them all
#This allows the MLP diff script to prune the similar new configurations an elimiante similar datasets prior to further ab initio calculation

performRun = False           # First system argument, generates and performs run if specified
try:
     if sys.argv[1] == "run": performRun = True  
except:
    pass


# We first combine the preselected configurations from each MD Run into a single master folder
masterConfigFileLocation = "./preselected.cfg"
mdRunsLocation = "../runs/MDRuns/"

#Clear the existing configurations
os.remove(masterConfigFileLocation)

with open(masterConfigFileLocation,'wb') as master:
    #Walk through the tree of directories in MD Runs
    #All child directories are run files which have no further children
    
    for directory, subdir, files in os.walk(mdRunsLocation):        
        if directory == mdRunsLocation: continue;       # There is no preselected config in the parent directory of the runs
         
        try: 
            childPreselectedConfigName = directory + "/preselected.cfg"         #Copy the preselected files to the master preselected 
            with open(childPreselectedConfigName,'rb') as child:
                shutil.copyfileobj(child, master)
        except:
            print(directory + " has failed to provide a preselected.cfg")

# We can then run the mlp script to isolate out the sufficiently different scripts 
os.chdir("../mdLearning")
os.system("/global/home/hpc5146/mlip-2/bin/mlp select-add /global/home/hpc5146/Projects/K-MTP-training/phase2/mdLearning/pot.mtp train.cfg ../activeLearningDFT/preselected.cfg ../activeLearningDFT/diff.cfg --als-filename=state.als")
os.chdir("../activeLearningDFT")


# Next, we can open the newly isolated configurations and extract the necessary information for the next training sets. 
# The following is borrows heavily from file.py, the exisiting script which was provided by Hao. I have adapted the scripts and added comments

#!!!!!!THIS DOES NOT CURRENTLY WORK FOR MULTIPLE TYPES OF ATOMS!!!!!!!!

# Let's prepare a series of arrays which have the respective properties of the ith atom, in the ith postion.
# We can then parse the diff.cfg file and store the needed information. See diff.cfg for format.

superCellVectorsList = []
numAtomsList = [] 
posAtomsList = []
with open("diff.cfg", 'r') as txtfile:
    fileLines = txtfile.readlines()
    index = np.where(np.array(fileLines) == "BEGIN_CFG\n")[0]   #Seach for indiicides which match the beginning of a configuration
    for i in index:
        
        configNumAtoms = int(fileLines[i+2].split()[0])
        numAtomsList.append(configNumAtoms)     #Read numAtoms
        
        v1 = np.array(fileLines[i+4].split(),dtype=float)       #Read supercell
        v2 = np.array(fileLines[i+5].split(),dtype=float)
        v3 = np.array(fileLines[i+6].split(),dtype=float)
        superCellVectorsList.append([v1,v2,v3])         
        
        configAtomicPositions = np.zeros((configNumAtoms,3))           #Temporary arrary to hold atomic positions
        for j in range(configNumAtoms):
            configAtomicPositions[j] = np.array(fileLines[i+8+j].split(),dtype=float)[2:5]
        posAtomsList.append(configAtomicPositions)


# We can generate the input and job submission files in the usual way now
dftRunTemplateLocation = "../../../activeLearningDFT/templateDiffDFTRun.in"          #location of dft run, data input, and job templates 
jobTemplateLocation = "../../../activeLearningDFT/templateDiffDFTRunSubmit.qsub"

#Make and prepare a new directory to hold all runs if needed 
os.chdir("../")
os.system("mkdir runs")
os.chdir("./runs")
os.system(" mkdir diffDFT")
os.chdir("./diffDFT")

for i in range(len(superCellVectorsList)):
    # Generate the necessary folder and file names (use a fairly unique identifier from the sum of position vectors)
    # identifier = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + random()
    identifier = posAtomsList[i][-1][0] + posAtomsList[i][-1][1] + posAtomsList[i][-1][2]
    folderName = "diffDFTRun" + str(identifier)
    inputName = "diffDFTRun" + str(identifier) + ".in"
    jobName = "diffDFTRun" + str(identifier)+ ".qsub"
    outputName = "diffDFTRun" + str(identifier) + ".out"
    
    numAtoms = numAtomsList[i]          # Extract the config info into variables for easier future usage
    superCell = superCellVectorsList[i]
    atomPositions = posAtomsList[i]
    
    # Generate a new directory for each dft run and navigate to it
    os.system("mkdir "+folderName)
    os.chdir(folderName)
    
    # Copy the templates for the QE input and data files
    shutil.copyfile(dftRunTemplateLocation, inputName)
    shutil.copyfile(jobTemplateLocation, jobName)
    
    # Make modifications to the QE input using regex substitutions
    with open (inputName, 'r+' ) as f:
        content = f.read()
        contentNew = re.sub("\$nnn", str(numAtoms), content)      #substitute nat marker with the number of atoms
        contentNew = re.sub("\$v1", str(superCell[0])[1:-1], contentNew)          #Same with supercell vectors.
        contentNew = re.sub("\$v2", str(superCell[1])[1:-1], contentNew)
        contentNew = re.sub("\$v3", str(superCell[2])[1:-1], contentNew)
        
        # Generate a series of string representing the list of atoms and positions
        atomPositionsString = []        
        for a in np.arange(numAtoms):
            atomPositionsString.append(' K %f %f %f 0 0 0  \n' % (posAtomsList[i][a][0], posAtomsList[i][a][1], posAtomsList[i][a][2]))         
        atomPositions = ' '.join(atomPositionsString)    
        contentNew = re.sub("\$aaa", atomPositions, contentNew)         #Subsitiute it in for the marker
        
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

print("Processed " + str(len(numAtomsList))  + " configurations." )

# for i in range(len(superCellVectorsList)):
#     with open(r'./command.in', 'r') as f:      
#         lines = f.readlines()
#     f.close()
#     nat = ('nat = %d\n' % numAtomsList[i])
#     atomPositionsString = []
#     for a in np.arange(numAtomsList[i]):
#         atomPositionsString.append(' Na %f %f %f 0 0 0  \n' % (posAtomsList[i][a][0], posAtomsList[i][a][1], posAtomsList[i][a][2]))
#     atom = ' '.join(atomPositionsString)    
#     lines[13] = nat
#     #lines[35] = " "
#     lines[38] = atom    
#     cellx = ' %f \t %f \t %f\n' % (superCellVectorsList[i][0][0], superCellVectorsList[i][0][1], superCellVectorsList[i][0][2])
#     celly = ' %f \t %f \t %f\n' % (superCellVectorsList[i][1][0], superCellVectorsList[i][1][1], superCellVectorsList[i][1][2])
#     cellz = ' %f \t %f \t %f\n' % (superCellVectorsList[i][2][0], superCellVectorsList[i][2][1], superCellVectorsList[i][2][2])
#     lines[32] = cellx
#     lines[33] = celly
#     lines[34] = cellz    
#     if not os.path.exists('./DFT_calc/%d' %i):
#         os.mkdir('./DFT_calc/%d' %i)   
#         #os.system("cp ../../run.sh ./DFT_calc/%d" %i)
#         #os.system("sbatch run.sh")
#     fp = open("./DFT_calc/%d/command.in" %i, 'w')  

#     configAtomicPositions = ' '.join(lines)
#     fp.write(configAtomicPositions)
#     fp.close()
