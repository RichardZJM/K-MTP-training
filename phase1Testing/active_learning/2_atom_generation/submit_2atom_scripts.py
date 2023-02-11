import shutil
import re
import os
import numpy as np

points = np.arange(0.8, 1.2, 0.02)

os.chdir("../")
os.chdir("./2AtomRuns")

for ele in points:
        
    fileName = "2Atom"+str(round(ele,2))+".in"
    folderName = "2Atom"+str(round(ele,2))
    outputName = "2Atom"+str(round(ele,2))+".out"
    submissionName = "2Atom"+str(round(ele,2))+".qsub"
    
    os.chdir("./2Atom"+str(round(ele,2)))
    
    shutil.copyfile("../../2_atom_generation/submit2AtomTemplate.qsub",submissionName)
    
    with open (submissionName, "r+") as f:
        content = f.read()
        
        content_new = re.sub("ininin", fileName, content)
        content_new = re.sub("folderfolder", folderName, content_new)
        content_new = re.sub("outoutout", outputName, content_new)
        
        f.seek(0)
        f.write(content_new)
        f.truncate()
        
        
    os.system("sbatch "+ submissionName);
    os.chdir("../")



# 4.1574

# 4.52
