import shutil
import re
import os
import numpy as np
import sys
import json
import subprocess
import random
from datetime import datetime

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

#region Folder Setup
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
    raise Exception(configFile + " not found or not in JSON format.")

dryRun = False           # Second argument. Performs dry run.
try:
     if sys.argv[2] == "dry": 
         dryRun = True;
         print("Performing Dry Run")
except:
    pass

rootFolder = os.path.dirname(os.path.realpath(__file__))              # Get useful folder locations
DFToutputFolder = rootFolder + "/outputDFT"
templatesFolder = rootFolder + "/templates"
scriptsFolder = rootFolder + "/pythonScripts"   
mtpFolder = rootFolder + "/mtpProperties"
slurmRunFolder = rootFolder + "/slurmRunOutputs"
mdFolder = rootFolder + "/mdLearningRuns"
diffDFTFolder = rootFolder + "/diffDFT"
initialGenerationFolder = rootFolder + "/initialGenerationDFT" 
os.chdir(rootFolder)

logFile = rootFolder + "/createTrainedPotential.log"

def printAndLog(message):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    datedMessage = dt_string + "   " + message
    print(datedMessage)
    with open(logFile, "a") as myfile:  myfile.write(datedMessage + "\n")

printAndLog("=============================================================")
printAndLog("Starting New Potential Training !!!")
printAndLog("=============================================================")

if not os.path.exists(slurmRunFolder): os.mkdir(slurmRunFolder)
if not os.path.exists(mdFolder): os.mkdir(mdFolder)
if not os.path.exists(diffDFTFolder): os.mkdir(diffDFTFolder)
if not os.path.exists(initialGenerationFolder): os.mkdir(initialGenerationFolder)
if not os.path.exists(DFToutputFolder): os.mkdir(DFToutputFolder)
else: 
    pass
    # RUN NEXT PART OF THE SCRIPT
    # exit(1)
#endregion

#region Inital Generation
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


template1AtomStrainDFT = templatesFolder + "/1AtomStrainDFT.in"
template1AtomShearDFT = templatesFolder + "/1AtomShearDFT.in"
template2AtomStrainDFT = templatesFolder + "/2AtomStrainDFT.in"
templateDFTJob = templatesFolder + "/dftRun.qsub"

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
        contentNew = re.sub("\$aaa", str(round(strain * params["baseLatticeParameter"] /2,5)), content)      #substitute lattice vector marker with the lattice vector
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
        contentNew = re.sub("\$aaa", str(round(shear * params["baseLatticeParameter"] /2,5)), content)      #substitute lattice vector marker with the lattice vector
        contentNew = re.sub("\$bbb", str(params["baseLatticeParameter"] /2), contentNew)      #substitute lattice vector marker with the lattice vector
        contentNew = re.sub("\$pseudo_dir", params["pseudopotentialDirectory"], contentNew)      
        contentNew = re.sub("\$pseudo", params["pseudopotential"], contentNew)  
        contentNew = re.sub("\$out", folderName, contentNew)  
        f.seek(0)
        f.write(contentNew)
        f.truncate()
    
    with open (jobName, 'r+' ) as f:
        content = f.read()
        contentNew = re.sub("\$job", "Shear" + str(shear), content) 
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
            contentNew = re.sub("\$aaa", str(round(strain * params["baseLatticeParameter"] ,5)), content)      #substitute lattice vector marker with the lattice vector
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
            f.seek(0)
            f.write(contentNew)
            f.truncate()
        
        if (not dryRun): subprocesses.append(subprocess.Popen(["sbatch",  jobName]))  

exitCodes = [p.wait() for p in subprocesses]        # Wait for all the initial generation to finish
subprocesses = []
failure = bool(sum(exitCodes))
if failure:
    printAndLog( str(sum([1 for x in exitCodes if x != 0])) + " of the inital DFT runs has been unsuccessful. Exiting now...")
    exit(1)  

printAndLog("Initial generation of DFT training dataset has completed.")
#endregion    

#region Active Learning

#region Active Learning Setup
# Get some useful file locations for the active learning
mtpFile = mtpFolder + "/pot.almtp"
trainingConfigs = mtpFolder + "/train.cfg"
preselectedConfigs = mtpFolder + "/preselected.cfg"
selectedConfigs = mtpFolder + "/selected.cfg"
diffConfigs = mtpFolder + "/diff.cfg"
outConfigs = mtpFolder + "/out.cfg"

#Prepare MD Runs folders
mdRunTemplate = templatesFolder + "/mdRun.in"
jobTemplate = templatesFolder + "/mdRun.qsub"

#Load the MDRuns parameters to use
temperatures = params["MDTemperatures"]
strains = np.arange(params["MDStrainRange"][0],params["MDStrainRange"][1],params["MDStrainStep"] )
configs = params["MDLatticeConfigs"]
baseline = params["baseLatticeParameter"]
maxItersPerConfig = params["maxItersPerConfig"]
maxIters = {tuple(configs[i]) : maxItersPerConfig[i] for i in range(len(configs))}
#endregion

printAndLog("Starting the active learning loop")

for config in configs:
    printAndLog("Beginning active learning of " + str(config) + " atoms.") 
    configName = str([int(x) for x in config]).replace(" ", "").replace(",", "")[1:-1]
    
    #region Generate MDRuns Folders
    if (os.path.exists(mdFolder)): shutil.rmtree(mdFolder)
    os.mkdir(mdFolder)
        
    for strain in strains:
        for temperature in temperatures:
            # Generate the necessary folder and file names
            folderName = mdFolder +  "/N" + configName + "T" + str (temperature) + "S" + str(strain)
            inputName =   folderName +  "/N" + configName + "T" + str (temperature) + "S" + str(strain) + ".in"
            dataName =  folderName +  "/N" + configName + "T" + str (temperature) + "S" + str(strain) + ".dat"
            jobName =  folderName +  "/N" + configName + "T" + str (temperature) + "S" + str(strain) + ".qsub"
            outputName =  folderName +  "/N" + configName + "T" + str (temperature) + "S" + str(strain)+ ".out"
            
            # Generate a new directory for each MD Run 
            if not os.path.exists(folderName): os.mkdir(folderName)
        
            # Copy the templates for the LAMMPS input and data files
            shutil.copyfile(mdRunTemplate, inputName)
            shutil.copyfile(jobTemplate, jobName)
            
            # Make modifications to the LAMMPS input using regex substitutions
            with open (inputName, 'r+' ) as f:
                content = f.read()
                contentNew = re.sub("\$ttt", str(temperature), content)  
                contentNew = re.sub("\$base", str(params["baseLatticeParameter"]*0.529177*strain), contentNew)
                contentNew = re.sub("\$111", str(config[0]), contentNew)   
                contentNew = re.sub("\$222", str(config[1]), contentNew)   
                contentNew = re.sub("\$333", str(config[2]), contentNew)   
                contentNew = re.sub("\$pot", mtpFile, contentNew)   
                f.seek(0)
                f.write(contentNew)
                f.truncate()
                
            # Make modifications to the job file using regex substitutions
            with open (jobName, 'r+' ) as f:
                content = f.read()
                contentNew = re.sub("\$job", "N" + configName + "T" + str(temperature) + "S" +str(strain), content) 
                contentNew = re.sub("\$outfile", folderName + "/out.run",contentNew) 
                contentNew = re.sub("\$folder", folderName, contentNew) 
                contentNew = re.sub("\$account", params["slurmParam"]["account"], contentNew) 
                contentNew = re.sub("\$partition", params["slurmParam"]["partition"], contentNew) 
                contentNew = re.sub("\$qos", params["slurmParam"]["qos"], contentNew) 
                contentNew = re.sub("\$cpus", params["mdJobParam"]["cpus"], contentNew) 
                contentNew = re.sub("\$time", params["mdJobParam"]["time"], contentNew) 
                contentNew = re.sub("\$lmpmpi", params["lmpMPIFile"], contentNew) 
                contentNew = re.sub("\$in", inputName, contentNew)      
                contentNew = re.sub("\$out", outputName, contentNew)
                f.seek(0)
                f.write(contentNew)
                f.truncate()
    printAndLog("Generated MD runs.")
    #endregion
    
    
    for i in range(maxIters[tuple(config)]):
        printAndLog(str(config) + " atoms, iteration: " + str(i+1) + " of up to " + str(maxIters[tuple(config)]))
        
        #region Extraction of DFT Results and Training
        extractionScript = scriptsFolder + "/extractConfigFromDFT.py"

        # Extract the outputs from the individual files and assemble a training config file
        os.chdir(DFToutputFolder)
        exitCode = subprocess.Popen(["python", extractionScript]).wait()
        # Copy the newly formed training config to the mtpProperties
        os.system("mv train.cfg " + trainingConfigs)
        os.chdir(rootFolder)

        # Generate and run train job file (job file must be used to avoid clogging login nodes)
        trainJobTemplate = templatesFolder + "/trainMTP.qsub"
        trainJob = mtpFolder + "/trainMTP.qsub"
        trainOutput = slurmRunFolder + "/train.out"
        shutil.copyfile(trainJobTemplate, trainJob)
        with open (trainJob, 'r+' ) as f:
                    content = f.read()
                    contentNew = re.sub("\$account", params["slurmParam"]["account"], content) 
                    contentNew = re.sub("\$partition", params["slurmParam"]["partition"], contentNew) 
                    contentNew = re.sub("\$qos", params["slurmParam"]["qos"], contentNew) 
                    contentNew = re.sub("\$mlp", params["mlpBinary"], contentNew)
                    contentNew = re.sub("\$mtp", mtpFile, contentNew)
                    contentNew = re.sub("\$train", trainingConfigs, contentNew)
                    contentNew = re.sub("\$outfile", slurmRunFolder  + "/train.out", contentNew)
                    f.seek(0)
                    f.write(contentNew)
                    f.truncate()

        exitCode = subprocess.Popen(["sbatch", trainJob]).wait()
        if(exitCode):
            print("The train call has failed. Potential may be unstable. Exiting...")
            exit(1)
        os.remove(trainJob)
        avgEnergyError = "Not Found"
        avgForceError = "Not Found"
        with open(trainOutput, "r") as txtfile:
            lines = txtfile.readlines()
            for i,line in enumerate(lines):
                if(line == "Energy per atom:\n"):
                    avgEnergyError = lines[i+3][31:-1]
                if(line == "Forces:\n"):
                    avgForceError = lines[i+3][31:-1]
        printAndLog("Passive training iteration completed.")
        printAndLog("Average energy per atom error: " + avgEnergyError)
        printAndLog("Average force per atom error: " + avgForceError)
        #endregion
        
        # Run the MD Jobs
        printAndLog("Starting MD Runs")
        for strain in strains:
            for temperature in temperatures:
                folderName = mdFolder +  "/N" + configName + "T" + str (temperature) + "S" + str(strain)
                jobName =  folderName +  "/N" + configName + "T" + str (temperature) + "S" + str(strain) + ".qsub"
                subprocesses.append(subprocess.Popen(["sbatch",  jobName]))  
        
        #region Assemble Preselected and Generate Diff CFG
        exitCodes = [p.wait() for p in subprocesses]        # Wait for all the mdRuns to finish
        os.chdir(rootFolder)
        subprocesses = []
        
        if bool(sum(exitCodes)):
            printAndLog( str(sum([1 for x in exitCodes if x != 0])) + " risky configurations found.")
        else: 
            printAndLog("No risky configurations found.")
        printAndLog("MD Runs Completed")
        
        try: os.remove(preselectedConfigs)
        except: pass
        
        completedRuns = 0
        with open(preselectedConfigs,'wb') as master:
        #Walk through the tree of directories in MD Runs
        #All child directories are run files which have no further children
            for directory, subdir, files in os.walk(mdFolder):        
                if directory == mdFolder: continue;       # There is no preselected config in the parent directory of the runs so skip
                
                childPreselectedConfigName = directory + "/preselected.cfg.0"   
                try: 
                         #Copy the preselected files to the master preselected 
                    with open(childPreselectedConfigName,'rb') as child:
                        shutil.copyfileobj(child, master)
                    os.remove(childPreselectedConfigName)
                except:
                    completedRuns += 1
                
        printAndLog("Runs with no preselected configurations: " + str(completedRuns) + " / " + str(len(temperatures)*len(strains)))
        
        # Generate the diff cfg
        selectAddJobTemplate = templatesFolder + "/selectAdd.qsub"
        selectAddJob = mtpFolder + "/selectAdd.qsub"
        shutil.copyfile(selectAddJobTemplate, selectAddJob)
        with open (selectAddJob, 'r+' ) as f:
                content = f.read()
                contentNew = re.sub("\$account", params["slurmParam"]["account"], content) 
                contentNew = re.sub("\$partition", params["slurmParam"]["partition"], contentNew) 
                contentNew = re.sub("\$qos", params["slurmParam"]["qos"], contentNew) 
                contentNew = re.sub("\$mlp", params["mlpBinary"], contentNew)
                contentNew = re.sub("\$outfile", slurmRunFolder + "/selectAdd.out", contentNew)
                contentNew = re.sub("\$mtp", mtpFile, contentNew)
                contentNew = re.sub("\$train", trainingConfigs, contentNew)
                contentNew = re.sub("\$preselected", preselectedConfigs, contentNew)
                contentNew = re.sub("\$selected", selectedConfigs, contentNew)
                contentNew = re.sub("\$diff", diffConfigs, contentNew)
                f.seek(0)
                f.write(contentNew)
                f.truncate()
        exitCode = subprocess.Popen(["sbatch", selectAddJob]).wait()
        if(exitCode):
            printAndLog("The select add call has failed. Exiting...")
            exit(1)
        os.remove(selectAddJob)
        printAndLog("Diff DFT configurations selected.")
        #endregion
        
        #region Assemble and run diff DFT Runs
        superCellVectorsList = []
        numAtomsList = [] 
        posAtomsList = []
        
        with open(diffConfigs, 'r') as txtfile:
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
        
        printAndLog("There are " + str(len(numAtomsList)) + " new diff DFT configurations.")
        printAndLog("Commencing diff DFT Runs.")
        # We can generate the input and job submission files in the usual way now
        dftRunTemplateLocation = templatesFolder + "/diffDFT.in"          #location of dft run, data input, and job templates 
        jobTemplateLocation = templatesFolder + "/dftRun.qsub"

        # Break if we find no more preselected configurations ahead of the iteration cap 
        if len(superCellVectorsList) == 0:
            print("No preselected configurations found. Moving to next atom count.")
            break
        
        for i in range(len(superCellVectorsList)):
            # Generate the necessary folder and file names (use a fairly unique identifier from the sum of position vectors)
            # identifier = datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + random()
            identifier = posAtomsList[i][-1][0] + posAtomsList[i][-1][1] + posAtomsList[i][-1][2]
            folderName = diffDFTFolder + "/" +str(identifier)
            inputName = folderName + "/diffDFTRun" + str(identifier) + ".in"
            jobName = folderName + "/diffDFTRun" + str(identifier) + ".qsub"
            outputName = DFToutputFolder + "/diffDFTRun" + str(identifier) +".out"
            
            numAtoms = numAtomsList[i]          # Extract the config info into variables for easier future usage
            superCell = superCellVectorsList[i]
            atomPositions = posAtomsList[i]
            #Use the inverse scaling property of k points to scale the computational burden based on the diff dft cell size
            kpoints = [int((params["baseLatticeParameter"]*0.529177)*params["baseKPoints"]//max(x)+ 1) for x in superCell]
            
            if not os.path.exists(folderName): os.mkdir(folderName)
            
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
                contentNew = re.sub("\$k1", str(kpoints[0]), contentNew)          #Same with supercell vectors.
                contentNew = re.sub("\$k2", str(kpoints[1]), contentNew)
                contentNew = re.sub("\$k3", str(kpoints[2]), contentNew)
                contentNew = re.sub("\$pseudo_dir", params["pseudopotentialDirectory"], contentNew)      
                contentNew = re.sub("\$pseudo", params["pseudopotential"], contentNew)  
                contentNew = re.sub("\$out", folderName, contentNew)  
                
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
                contentNew = re.sub("\$job", "diffDFT" + str(identifier), content) 
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
            subprocesses.append(subprocess.Popen(["sbatch",  jobName]))  
        
        exitCodes = [p.wait() for p in subprocesses]        # Wait for all the diffDFT to finish
        subprocesses = []
        
        if bool(sum(exitCodes)):
            printAndLog("One or more of the diff DFT runs has failed. Potential may be unstable be warned. Continuing...")
        else: 
            pass
        printAndLog("Diff DFT calculations complete.")
        # exit(1)
        #endregion
        
#endregion

# except Exception as e:
#     print("An error has occur during the generation of the initial files. Verify the formating of your JSON config file.")
#     raise Exception(e)
