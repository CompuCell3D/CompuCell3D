def configureSimulation(sim):
    import CompuCellSetup
    from XMLUtils import ElementCC3D

    cc3d=ElementCC3D("CompuCell3D")
    potts=cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions",{"x":55,"y":55,"z":1})
    potts.ElementCC3D("Steps",{},1000)
    potts.ElementCC3D("Temperature",{},0)
    potts.ElementCC3D("Flip2DimRatio",{},0.0)

    cellType=cc3d.ElementCC3D("Plugin",{"Name":"CellType"})
    cellType.ElementCC3D("CellType", {"TypeName":"Medium", "TypeId":"0"})


    flexDiffSolver=cc3d.ElementCC3D("Steppable",{"Type":"DiffusionSolverFE"})
    diffusionField=flexDiffSolver.ElementCC3D("DiffusionField")
    diffusionData=diffusionField.ElementCC3D("DiffusionData")
    diffusionData.ElementCC3D("FieldName",{},"FGF")
    diffusionData.ElementCC3D("DiffusionConstant",{},0.10)
    diffusionData.ElementCC3D("DecayConstant",{},0.0)
    diffusionData.ElementCC3D("ConcentrationFileName",{},"Simulation/diffusion_2D.pulse.txt")
    


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

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
