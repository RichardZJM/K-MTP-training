import os
import sys
import shutil

#Clears the directories and prepares another createTrainedPotential run

loadedMTP = "12"          # Second argument. Performs dry run.
try:
     loadedMTP = sys.argv[1]
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
trainingConfigs = mtpFolder + "/train.cfg"
preselectedConfigs = mtpFolder + "/preselected.cfg"
selectedConfigs = mtpFolder + "/selected.cfg"
diffConfigs = mtpFolder + "/diff.cfg"
outConfigs = mtpFolder + "/out.cfg"
iniFile = mtpFolder + "/mlip.ini"
alsFile = mtpFolder + "/state.als"

if  os.path.exists(slurmRunFolder): os.remove(slurmRunFolder)
if  os.path.exists(mdFolder): os.remove(mdFolder)
if  os.path.exists(diffDFTFolder): os.remove(diffDFTFolder)
if  os.path.exists(initialGenerationFolder): os.remove(initialGenerationFolder)
if  os.path.exists(DFToutputFolder): os.remove(DFToutputFolder)

if  os.path.exists(mtpFile): os.remove(slurmRunFolder)
if  os.path.exists(trainingConfigs): os.remove(mdFolder)
if  os.path.exists(preselectedConfigs): os.remove(diffDFTFolder)
if  os.path.exists(selectedConfigs): os.remove(initialGenerationFolder)
if  os.path.exists(diffConfigs): os.remove(DFToutputFolder)
if  os.path.exists(outConfigs): os.remove(DFToutputFolder)
if  os.path.exists(iniFile): os.remove(DFToutputFolder)
if  os.path.exists(alsFile): os.remove(DFToutputFolder)

freshMTP = mtpFolder + "/" + loadedMTP + ".mtp"

shutil.copy(freshMTP,mtpFile)

os.chdir(rootFolder)