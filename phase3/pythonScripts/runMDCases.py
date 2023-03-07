import shutil
import re
import os
import numpy as np
import sys
import json

rootFolder = ""           # First system argument, accepts the root folder from the 

try:
    if sys.argv[1] == None: quit()
    else: rootFolder = sys.argv[1]
except:
    quit()
    
progressFolder = rootFolder + "/progress"
