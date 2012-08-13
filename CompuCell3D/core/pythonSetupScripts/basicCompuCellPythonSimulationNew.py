import sys
from os import environ
import string
python_module_path=os.environ["PYTHON_MODULE_PATH"]
appended=sys.path.count(python_module_path)
if not appended:
    sys.path.append(python_module_path)
    
# sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup
if not CompuCellSetup.simulationFileName=="":
    CompuCellSetup.setSimulationXMLFileName(CompuCellSetup.simulationFileName)
sim,simthread = CompuCellSetup.getCoreSimulationObjects()
import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

#Create extra player fields here or add attributes or plugin:
#PUT YOUR CODE HERE
#PUT YOUR CODE HERE
#PUT YOUR CODE HERE


import XMLUtils
steppableList=CompuCellSetup.cc3dXML2ObjConverter.root.getElements("Plugin")
steppableListPy=XMLUtils.CC3DXMLListPy(steppableList)

for element in steppableListPy:
    print "Element",element.name
#     ," name", element.getNumberOfChildren()



CompuCellSetup.initializeSimulationObjects(sim,simthread)


steppableRegistry=CompuCellSetup.getSteppableRegistry()
#Add Python steppables here
#PUT YOUR CODE HERE
#PUT YOUR CODE HERE
#PUT YOUR CODE HERE




# sim.ps.steppableCC3DXMLElementVector()
CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)#main loop - simulation is invoked inside this function

