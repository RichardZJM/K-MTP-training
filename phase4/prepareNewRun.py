import os
import sys
import shutil

#Clears the directories and prepares another createTrainedPotential run

loadedMTP = "12"          # Second argument. Performs dry run.
try:
     loadedMTP = sys.argv[1]
except:
    pass

preserveInitial = False         # Second argument. Performs dry run.
try:
     if sys.argv[2] == "save": preserveInitial = True
except:
    pass


rootFolder = os.path.dirname(os.path.realpath(__file__))              # Get useful folder locations
DFToutputFolder = rootFolder + "/outputDFT"
mtpFolder = rootFolder + "/mtpProperties"
slurmRunFolder = rootFolder + "/slurmRunOutputs"
mdFolder = rootFolder + "/mdLearningRuns"
diffDFTFolder = rootFolder + "/diffDFT"
initialGenerationFolder = rootFolder + "/initialGenerationDFT" 

mtpFile = mtpFolder + "/pot.mtp"
mtpJobFile = mtpFile + "/trainMTP.qsub"
trainingConfigs = mtpFolder + "/train.cfg"
preselectedConfigs = mtpFolder + "/preselected.cfg"
selectedConfigs = mtpFolder + "/selected.cfg"
diffConfigs = mtpFolder + "/diff.cfg"
outConfigs = mtpFolder + "/out.cfg"
iniFile = mtpFolder + "/mlip.ini"
alsFile = mtpFolder + "/state.als"
bfgsLog = rootFolder + "/bfgs.log"

# if  os.path.exists(slurmRunFolder): shutil.rmtree(slurmRunFolder)
if  os.path.exists(mdFolder): shutil.rmtree(mdFolder)
if  os.path.exists(diffDFTFolder): shutil.rmtree(diffDFTFolder)
if  os.path.exists(DFToutputFolder): shutil.rmtree(DFToutputFolder)

if  os.path.exists(initialGenerationFolder) and not preserveInitial: shutil.rmtree(initialGenerationFolder)

if  os.path.exists(mtpFile): os.remove(mtpFile)
if  os.path.exists(bfgsLog): os.remove(bfgsLog)
if  os.path.exists(mtpJobFile): os.remove(mtpJobFile)
if  os.path.exists(trainingConfigs): os.remove(trainingConfigs)
if  os.path.exists(preselectedConfigs): os.remove(preselectedConfigs)
if  os.path.exists(selectedConfigs): os.remove(selectedConfigs)
if  os.path.exists(diffConfigs): os.remove(diffConfigs)
if  os.path.exists(outConfigs): os.remove(outConfigs)
if  os.path.exists(iniFile): os.remove(iniFile)
if  os.path.exists(alsFile): os.remove(alsFile)

freshMTP = mtpFolder + "/" + loadedMTP + ".mtp"

shutil.copy(freshMTP, mtpFile)

os.chdir(rootFolder)