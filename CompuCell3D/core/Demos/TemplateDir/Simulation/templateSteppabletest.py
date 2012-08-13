def configureSimulation(sim):
    import CompuCellSetup,time
    from XMLUtils import ElementCC3D
    C1 = 1
    C2 = 1
    C3 = 1
    C4 = 1
    J0 = 16

    cc3d=ElementCC3D("CompuCell3D")
    potts=cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions",{"x":50,"y":50,"z":1})
    potts.ElementCC3D("Steps",{},1000)
    potts.ElementCC3D("Temperature",{},10)
    potts.ElementCC3D("NeighborOrder",{},2)

    cellType=cc3d.ElementCC3D("Plugin",{"Name":"CellType"})
    cellType.ElementCC3D("CellType", {"TypeName":"Medium", "TypeId":"0"})
    cellType.ElementCC3D("CellType", {"TypeName":"Condensing", "TypeId":"1"})
    cellType.ElementCC3D("CellType", {"TypeName":"NonCondensing", "TypeId":"2"})



    volume=cc3d.ElementCC3D("Plugin",{"Name":"Volume"})
    volume.ElementCC3D("TargetVolume",{},25)
    volume.ElementCC3D("LambdaVolume",{},2.0)

    contact=cc3d.ElementCC3D("Plugin",{"Name":"Contact"})
    contact.ElementCC3D("Energy", {"Type1":"Medium", "Type2":"Medium"},0)
    contact.ElementCC3D("Energy", {"Type1":"NonCondensing", "Type2":"NonCondensing"},16)
    contact.ElementCC3D("Energy", {"Type1":"Condensing", "Type2":"Condensing"},2)
    contact.ElementCC3D("Energy",{"Type1":"NonCondensing", "Type2":"Condensing"},11)
    contact.ElementCC3D("Energy", {"Type1":"NonCondensing", "Type2":"Medium"},16)
    contact.ElementCC3D("Energy", {"Type1":"Condensing", "Type2":"Medium"},16)

    uniform = cc3d.ElementCC3D("Steppable",{"Type":"UniformInitializer"})                                                            
    region = uniform.ElementCC3D("Region") 
    region.ElementCC3D("BoxMin",{"x":20,  "y":20,  "z":0})                                                                 
    region.ElementCC3D("BoxMax",{"x":25,  "y":25,  "z":1})                                                         
    region.ElementCC3D("Types",{}, "Condensing")                                                                                             
    region.ElementCC3D("Width", {}, 5)                                                                                                  
    region1 = uniform.ElementCC3D("Region") 
    region1.ElementCC3D("BoxMin",{"x":20,  "y":25,  "z":0})                                                                 
    region1.ElementCC3D("BoxMax",{"x":25,  "y":30,  "z":1})                                                         
    region1.ElementCC3D("Types",{}, "NonCondensing")                                                                                             
    region1.ElementCC3D("Width", {}, 5)                                                                                                  
    region1 = uniform.ElementCC3D("Region") 
    region1.ElementCC3D("BoxMin",{"x":25,  "y":25,  "z":0})                                                                 
    region1.ElementCC3D("BoxMax",{"x":30,  "y":30,  "z":1})                                                         
    region1.ElementCC3D("Types",{}, "Condensing")                                                                                             
    region1.ElementCC3D("Width", {}, 5)                                                                                                  
    
    templateSteppable = cc3d.ElementCC3D("Steppable",{"Type":"TemplateSteppable"})    
    templateSteppable.ElementCC3D("PIFName",{},"happy")                                                        
      

    CompuCellSetup.setSimulationXMLDescription(cc3d)
    
import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])
    
import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

configureSimulation(sim)

CompuCellSetup.initializeSimulationObjects(sim,simthread)


from PySteppables import SteppableRegistry
steppableRegistry=SteppableRegistry()

#from molecule_steppable import InfoPrinterSteppable
#infoPrinterSteppable=InfoPrinterSteppable(_simulator=sim,_frequency=10)
#steppableRegistry.registerSteppable(infoPrinterSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

