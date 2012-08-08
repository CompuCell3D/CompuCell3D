def configureSimulation(sim):
    import CompuCellSetup
    from XMLUtils import ElementCC3D

    cc3d=ElementCC3D("CompuCell3D")
    potts=cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions",{"x":100,"y":100,"z":1})
    potts.ElementCC3D("Steps",{},10000)
    potts.ElementCC3D("Temperature",{},5)
    potts.ElementCC3D("NeighborOrder",{},2)
    potts.ElementCC3D("Boundary_x",{},"Periodic")

    cellType=cc3d.ElementCC3D("Plugin",{"Name":"CellType"})
    cellType.ElementCC3D("CellType", {"TypeName":"Medium", "TypeId":"0"})
    cellType.ElementCC3D("CellType", {"TypeName":"Body1", "TypeId":"1"})
    cellType.ElementCC3D("CellType", {"TypeName":"Body2", "TypeId":"2"})
    cellType.ElementCC3D("CellType", {"TypeName":"Body3", "TypeId":"3"})

    volume=cc3d.ElementCC3D("Plugin",{"Name":"Volume"})
    volume.ElementCC3D("TargetVolume",{},25)
    volume.ElementCC3D("LambdaVolume",{},4.0)




    contact=cc3d.ElementCC3D("Plugin",{"Name":"Contact"})
    contact.ElementCC3D("Energy", {"Type1":"Medium", "Type2":"Medium"},0)
    contact.ElementCC3D("Energy", {"Type1":"Body1", "Type2":"Body1"},16)
    contact.ElementCC3D("Energy", {"Type1":"Body1", "Type2":"Medium"},4)
    contact.ElementCC3D("Energy",{"Type1":"Body2", "Type2":"Body2"},16)
    contact.ElementCC3D("Energy", {"Type1":"Body2", "Type2":"Medium"},4)
    contact.ElementCC3D("Energy", {"Type1":"Body3", "Type2":"Body3"},16)
    contact.ElementCC3D("Energy", {"Type1":"Body3", "Type2":"Medium"},4)
    contact.ElementCC3D("Energy", {"Type1":"Body1", "Type2":"Body2"},16)
    contact.ElementCC3D("Energy", {"Type1":"Body1", "Type2":"Body3"},16)
    contact.ElementCC3D("Energy", {"Type1":"Body2", "Type2":"Body3"},16)
    contact.ElementCC3D("neighborOrder" , {} , 2)

    centerOfMass=cc3d.ElementCC3D("Plugin",{"Name":"CenterOfMass"})

    elasticityTracker=cc3d.ElementCC3D("Plugin",{"Name":"ElasticityTracker"})
    elasticityTracker.ElementCC3D("IncludeType",{},"Body1")
    elasticityTracker.ElementCC3D("IncludeType",{},"Body2")
    elasticityTracker.ElementCC3D("IncludeType",{},"Body3")

    elasticityEnergy=cc3d.ElementCC3D("Plugin",{"Name":"ElasticityEnergy"})
    elasticityEnergy.ElementCC3D("Local",{})
    # elasticityEnergy.ElementCC3D("LambdaElasticity",{},200.0)
    # elasticityEnergy.ElementCC3D("TargetLengthElasticity",{},6)

    externalPotential=cc3d.ElementCC3D("Plugin",{"Name":"ExternalPotential"})
    externalPotential.ElementCC3D("Lambda",{"x":-10,"y":0, "z":0})


    pifInitializer=cc3d.ElementCC3D("Steppable",{"Type":"PIFInitializer"})
    pifInitializer.ElementCC3D("PIFName",{},"Simulation/elasticitytest.piff")

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

from elasticitytestSteppables import ElasticityLocalSteppable
elasticitySteppable=ElasticityLocalSteppable(_simulator=sim,_frequency=50)
steppableRegistry.registerSteppable(elasticitySteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)

