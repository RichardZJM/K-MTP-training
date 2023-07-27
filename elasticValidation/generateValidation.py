import os
import numpy as np

pairs = {}


if os.path.exists("outMTP.txt"): os.remove("outMTP.txt")

with open("out.cfg", 'r') as txtfile:
            fileLines = txtfile.readlines()
            index = np.where(np.array(fileLines) == "BEGIN_CFG\n")[0]   #Seach for indiicides which match the beginning of a configuration
            for i in index:
                
                energy = float(fileLines[i+10])   #Read energy
                # print(energy)
                
               
                v1 = np.array(fileLines[i+4].split(),dtype=float)       #Read supercell
                latticeParam  = max(v1) * 1.88973 * 2
                pairs[latticeParam] = str(energy*0.0734985857)
                
                
for lat,eng in sorted(pairs.items(), key = lambda x: x[0]):
    with open ("outMTP.txt","a") as file:
        file.write(str(lat) + "\t" + eng + "\n")