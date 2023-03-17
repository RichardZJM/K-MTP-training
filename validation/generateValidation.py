import os
import numpy as np


distances = np.arange(9.5,10.6,0.1)
counter = 0

if os.path.exists("outMTP.txt"): os.remove("outMTP.txt")

with open("out.cfg", 'r') as txtfile:
            fileLines = txtfile.readlines()
            index = np.where(np.array(fileLines) == "BEGIN_CFG\n")[0]   #Seach for indiicides which match the beginning of a configuration
            for i in index:
                
                energy = float(fileLines[i+10])   #Read energy
                print(energy)
                with open ("outMTP.txt","a") as file:
                    file.write(str(round(distances[counter],1)) + " " + str(energy*0.0734985857) + "\n" )
                counter += 1
