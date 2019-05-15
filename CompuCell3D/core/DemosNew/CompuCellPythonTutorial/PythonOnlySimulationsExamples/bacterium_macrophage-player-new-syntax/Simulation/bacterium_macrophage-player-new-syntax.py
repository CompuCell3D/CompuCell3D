from cc3d import CompuCellSetup


def configure_simulation():
    from cc3d.core.XMLUtils import ElementCC3D

    cc3d = ElementCC3D("CompuCell3D")
    potts = cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions", {"x": 100, "y": 100, "z": 1})
    potts.ElementCC3D("Steps", {}, 10000)
    potts.ElementCC3D("Temperature", {}, 15)
    potts.ElementCC3D("NeighborOrder", {}, 2)

    cellType = cc3d.ElementCC3D("Plugin", {"Name": "CellType"})
    cellType.ElementCC3D("CellType", {"TypeName": "Medium", "TypeId": "0"})
    cellType.ElementCC3D("CellType", {"TypeName": "Bacterium", "TypeId": "1"})
    cellType.ElementCC3D("CellType", {"TypeName": "Macrophage", "TypeId": "2"})
    cellType.ElementCC3D("CellType", {"TypeName": "Wall", "TypeId": "3", "Freeze": ""})

    volume = cc3d.ElementCC3D("Plugin", {"Name": "Volume"})
    volume.ElementCC3D("TargetVolume", {}, 25)
    volume.ElementCC3D("LambdaVolume", {}, 15.0)

    surface = cc3d.ElementCC3D("Plugin", {"Name": "Surface"})
    surface.ElementCC3D("TargetSurface", {}, 20)
    surface.ElementCC3D("LambdaSurface", {}, 4.0)

    contact = cc3d.ElementCC3D("Plugin", {"Name": "Contact"})
    contact.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "Medium"}, 0)
    contact.ElementCC3D("Energy", {"Type1": "Macrophage", "Type2": "Macrophage"}, 15)
    contact.ElementCC3D("Energy", {"Type1": "Macrophage", "Type2": "Medium"}, 8)
    contact.ElementCC3D("Energy", {"Type1": "Bacterium", "Type2": "Bacterium"}, 15)
    contact.ElementCC3D("Energy", {"Type1": "Bacterium", "Type2": "Macrophage"}, 15)
    contact.ElementCC3D("Energy", {"Type1": "Bacterium", "Type2": "Medium"}, 8)
    contact.ElementCC3D("Energy", {"Type1": "Wall", "Type2": "Wall"}, 0)
    contact.ElementCC3D("Energy", {"Type1": "Wall", "Type2": "Medium"}, 0)
    contact.ElementCC3D("Energy", {"Type1": "Wall", "Type2": "Bacterium"}, 50)
    contact.ElementCC3D("Energy", {"Type1": "Wall", "Type2": "Macrophage"}, 50)

    chemotaxis = cc3d.ElementCC3D("Plugin", {"Name": "Chemotaxis"})
    chemicalField = chemotaxis.ElementCC3D("ChemicalField", {"Source": "DiffusionSolverFE", "Name": "ATTR"})
    chemicalField.ElementCC3D("ChemotaxisByType", {"Type": "Macrophage", "Lambda": 200})

    flexDiffSolver = cc3d.ElementCC3D("Steppable", {"Type": "DiffusionSolverFE"})
    diffusionField = flexDiffSolver.ElementCC3D("DiffusionField")
    diffusionData = diffusionField.ElementCC3D("DiffusionData")
    diffusionData.ElementCC3D("FieldName", {}, "ATTR")
    diffusionData.ElementCC3D("DiffusionConstant", {}, 0.10)
    diffusionData.ElementCC3D("DecayConstant", {}, 0.0)
    diffusionData.ElementCC3D("DoNotDiffuseTo", {}, "Wall")
    secretionData = diffusionField.ElementCC3D("SecretionData")
    secretionData.ElementCC3D("Secretion", {"Type": "Bacterium"}, 200)

    pifInitializer = cc3d.ElementCC3D("Steppable", {"Type": "PIFInitializer"})
    pifInitializer.ElementCC3D("PIFName", {}, "Simulation/bacterium_macrophage_2D_wall.piff")

    CompuCellSetup.setSimulationXMLDescription(cc3d)


configure_simulation()

CompuCellSetup.run()
