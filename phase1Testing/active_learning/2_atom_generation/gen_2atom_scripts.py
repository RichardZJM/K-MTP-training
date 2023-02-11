import shutil
import re
import os
import numpy as np



baseline = 4.83583;
points = np.arange(0.8, 1.2, 0.02)

os.chdir("../")
os.system("mkdir 2AtomRuns")
os.chdir("./2AtomRuns")

for ele in points:
        
    fileName = "2Atom"+str(round(ele,2))+".in"
    outputName = "2Atom"+str(round(ele,2))+".out"
    
    os.system("mkdir 2Atom"+str(round(ele,2)))
    os.chdir("./2Atom"+str(round(ele,2)))
    os.system("pwd")
    
    shutil.copyfile("../../2_atom_generation/2AtomTemplate.in",fileName)
    
    
    
    with open (fileName, 'r+' ) as f:
        content = f.read()

        content_new = re.sub("ee", str(baseline*ele), content)
        
        f.seek(0)
        f.write(content_new)
        f.truncate()
        
        # os.system("pw.x < " + fileName + ">" + outputName)
        
    os.chdir("../")



# 4.1574

# 4.52
