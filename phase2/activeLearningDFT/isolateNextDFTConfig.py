import shutil
import re
import os
import sys

#This script complies all the individual preselected config scripts in the md runs parent folder and complies them all
#This allows the MLP diff script to prune the similar new configurations an elimiante similar datasets prior to further ab initio calculation

masterConfigFileLocation = "./preselected.cfg"
mdRunsLocation = "../runs/MDRuns/"

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
            
os.chdir("../mdLearning")
os.system("/global/home/hpc5146/mlip-2/bin/mlp select-add /global/home/hpc5146/Projects/K-MTP-training/phase2/mdLearning/pot.mtp train.cfg ../activeLearningDFT/preselected.cfg ../activeLearningDFT/diff.cfg --als-filename=state.als")
os.chdir("../activeLearningDFT")