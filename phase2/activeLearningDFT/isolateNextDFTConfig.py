import shutil
import re
import os
import sys

#This script complies all the individual preselected config scripts in the md runs parent folder and complies them all
#This allows the MLP diff script to prune the similar new configurations an elimiante similar datasets prior to further ab initio calculation

masterConfigFileLocation = "./preselected.cfg"
mdRunsLocation = "../runs/MDRuns/"



with open(masterConfigFileLocation,'wb') as master:
    for childDirectories, descendantDirectories, files in os.walk(masterConfigFileLocation):
        print(childDirectories)
        for runDirectory in childDirectories:
            childPreselectedConfigName = mdRunsLocation + runDirectory + "preselected.cfg"
            with open(childPreselectedConfigName,'rb') as child:
                shutil.copyfileobj(child, master)