import sys
import os
from os import environ
# print environ["PYTHON_MODULE_PATH"]
sys.path.append(environ["PYTHON_MODULE_PATH"])
sys.path.append(environ["SWIG_LIB_INSTALL_DIR"])

import bionetAPITest
sbmlModelPath = os.getcwd() + "/Demos/BionetSolverExamples/sbmlModels/SimpleExample.xml"
print "sbmlModelPath=",sbmlModelPath,len(sbmlModelPath)
bionetAPITest.initializeBionetworkManager( None )
bionetAPITest.loadSBMLModel( "SimpleExample", sbmlModelPath, "SE", 0.5 )