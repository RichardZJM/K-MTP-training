import shutil
import re
import os
import numpy as np
import sys
import json

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

# Try loop look for missing values in the dict or issues in the formating.
try:
    rootFolder = os.getcwd()                    # First we generate a folder to hold all the inital DFT runs
    initalGenerationFolder = rootFolder + "/initalGenerationDFT"
    if not os.path.exists(initalGenerationFolder): os.mkdir(initalGenerationFolder)
    
    DFT1AtomExpansionFolder = initalGenerationFolder + "/1AtomDFTExpansion"
    DFT1AtomShearFolder = initalGenerationFolder + "/1AtomDFTShear"
    DFT2AtomExpansionFolder = initalGenerationFolder + "/2AtomDFTExpansion"
    
    if not os.path.exists(DFT1AtomExpansionFolder): os.mkdir(DFT1AtomExpansionFolder)
    if not os.path.exists(DFT1AtomShearFolder): os.mkdir(DFT1AtomShearFolder)
    if not os.path.exists(DFT2AtomExpansionFolder): os.mkdir(DFT2AtomExpansionFolder)
    
    
except Exception as e:
    print("An error has occur during the generation of the initial files. Verify the formating of your JSON config file.")
    raise Exception(e)

