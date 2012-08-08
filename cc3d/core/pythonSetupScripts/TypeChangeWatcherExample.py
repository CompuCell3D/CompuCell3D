import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])
import SystemUtils
SystemUtils.initializeSystemResources()

import CompuCellSetup
sim,simthread = CompuCellSetup.getCoreSimulationObjects()
import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()

#Create extra player fields here or add attributes

#typeChangeWatcher=CompuCell.TypeChangeWatcherPyWrapper()
#from PyPluginsExamples import TypeChangeWatcherExample
#typeChangeWatcherExample=TypeChangeWatcherExample(typeChangeWatcher)
#typeChangeWatcher.registerPyTypeChangeWatcher(typeChangeWatcherExample)

#sim.getPotts().getTypeTransition().registerTypeChangeWatcher(typeChangeWatcher)



CompuCellSetup.initializeSimulationObjects(sim,simthread)

dim=sim.getPotts().getCellFieldG().getDim()


#Add Python steppables here
from PySteppablesExamples import SteppableRegistry
steppableRegistry=SteppableRegistry()

from PySteppablesExamples import ContactLocalFlexPrinter
clfdPrinter=ContactLocalFlexPrinter()
steppableRegistry.registerSteppable(clfdPrinter)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



#def mainfcn():
   #import sys
   #from os import environ
   #import string
   #sys.path.append(environ["PYTHON_MODULE_PATH"])

   #import SystemUtils
   #SystemUtils.setSwigPaths()
   #SystemUtils.initializeSystemResources()

   #import CompuCell
   #import CompuCell
   #import PlayerPython
   #from PyPluginsExamples import VolumeEnergyFunction
   #from PyPluginsExamples import SurfaceEnergyFunction
   #from PyPluginsExamples import MitosisPy
   ##from PyPluginsExamples import PyWatcher
   ## This function wraps up the plugin initialization code
   #CompuCell.initializePlugins()
   
   ## Create a Simulator.  This returns a Python object that wraps
   ## Simulator.
   #sim = CompuCell.Simulator()
   
   #simthread=PlayerPython.getSimthreadBasePtr();
   #simthread.setSimulator(sim)
   #simulationFileName=simthread.getSimulationFileName()
   #print "simulationFileName=",simulationFileName
   
   ## Add the Python specific extensions
   #reg = sim.getClassRegistry()

   #CompuCell.parseConfig(simulationFileName, sim)
   
   ##setting up Py Energy function holder
   #extraEnergy=CompuCell.EnergyFunctionPyWrapper()
   #extraEnergy.setSimulator(sim)
   #extraEnergy.setPotts(sim.getPotts())
   
   #sim.getPotts().registerEnergyFunction(extraEnergy.getEnergyFunctionPyWrapperPtr())
   
   #typeChangeWatcher=CompuCell.TypeChangeWatcherPyWrapper()
   #from PyPluginsExamples import TypeChangeWatcherExample
   #typeChangeWatcherExample=TypeChangeWatcherExample(typeChangeWatcher)
   #typeChangeWatcher.registerPyTypeChangeWatcher(typeChangeWatcherExample)
   
   #sim.getPotts().getTypeTransition().registerTypeChangeWatcher(typeChangeWatcher)
   
   

   #sim.extraInit() #after all xml steppables and plugins have been loaded we call extraInit to complete initialization
   
   #print "GOT HERE"
   #sys.exit()
   
   #simthread.preStartInit()
   
   
   #dim=sim.getPotts().getCellFieldG().getDim()
   
   
   #sim.start()
   
   
   #simthread.postStartInit()
   
   #screenUpdateFrequency=simthread.getScreenUpdateFrequency()
   
   #from PySteppables import SteppableRegistry
   
   #steppableRegistry=SteppableRegistry()
   
   #from PySteppablesExamples import ContactLocalFlexPrinter
   #clfdPrinter=ContactLocalFlexPrinter()
   #steppableRegistry.registerSteppable(clfdPrinter)
   
   #steppableRegistry.init(sim)
   
   #steppableRegistry.start()

   #for i in range(sim.getNumSteps()):
      #sim.step(i)
      #steppableRegistry.step(i)
      ##volumeEnergy.vt+=1
      #if not i % screenUpdateFrequency:
         #simthread.loopWork(i)
         #simthread.loopWorkPostEvent(i)

   #sim.finish()
   #steppableRegistry.finish(i)
   
#mainfcn()