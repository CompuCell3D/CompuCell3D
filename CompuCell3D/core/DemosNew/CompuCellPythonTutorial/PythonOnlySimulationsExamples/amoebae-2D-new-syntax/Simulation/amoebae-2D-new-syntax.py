from cc3d import CompuCellSetup


def configure_simulation():
    from cc3d.core.XMLUtils import ElementCC3D

    cc3d = ElementCC3D("CompuCell3D")
    potts = cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions", {"x": 55, "y": 55, "z": 1})
    potts.ElementCC3D("Steps", {}, 1000)
    potts.ElementCC3D("Temperature", {}, 15)
    potts.ElementCC3D("Boundary_y", {}, "Periodic")

    cellType = cc3d.ElementCC3D("Plugin", {"Name": "CellType"})
    cellType.ElementCC3D("CellType", {"TypeName": "Medium", "TypeId": "0"})
    cellType.ElementCC3D("CellType", {"TypeName": "Amoeba", "TypeId": "1"})
    cellType.ElementCC3D("CellType", {"TypeName": "Bacteria", "TypeId": "2"})

    volume = cc3d.ElementCC3D("Plugin", {"Name": "Volume"})
    volume.ElementCC3D("TargetVolume", {}, 25)
    volume.ElementCC3D("LambdaVolume", {}, 15.0)

    surface = cc3d.ElementCC3D("Plugin", {"Name": "Surface"})
    surface.ElementCC3D("TargetSurface", {}, 25)
    surface.ElementCC3D("LambdaSurface", {}, 2.0)

    contact = cc3d.ElementCC3D("Plugin", {"Name": "Contact"})
    contact.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "Medium"}, 0)
    contact.ElementCC3D("Energy", {"Type1": "Amoeba", "Type2": "Amoeba"}, 15)
    contact.ElementCC3D("Energy", {"Type1": "Amoeba", "Type2": "Medium"}, 8)
    contact.ElementCC3D("Energy", {"Type1": "Bacteria", "Type2": "Bacteria"}, 15)
    contact.ElementCC3D("Energy", {"Type1": "Bacteria", "Type2": "Amoeba"}, 15)
    contact.ElementCC3D("Energy", {"Type1": "Bacteria", "Type2": "Medium"}, 8)
    contact.ElementCC3D("NeighborOrder", {}, 2)

    chemotaxis = cc3d.ElementCC3D("Plugin", {"Name": "Chemotaxis"})
    chemicalField = chemotaxis.ElementCC3D("ChemicalField", {"Source": "DiffusionSolverFE", "Name": "FGF"})
    chemicalField.ElementCC3D("ChemotaxisByType", {"Type": "Amoeba", "Lambda": 3})
    chemicalField.ElementCC3D("ChemotaxisByType", {"Type": "Bacteria", "Lambda": 2})

    flexDiffSolver = cc3d.ElementCC3D("Steppable", {"Type": "DiffusionSolverFE"})
    diffusionField = flexDiffSolver.ElementCC3D("DiffusionField")
    diffusionData = diffusionField.ElementCC3D("DiffusionData")
    diffusionData.ElementCC3D("FieldName", {}, "FGF")
    diffusionData.ElementCC3D("DiffusionConstant", {}, 0.0)
    diffusionData.ElementCC3D("DecayConstant", {}, 0.0)
    diffusionData.ElementCC3D("ConcentrationFileName", {}, "Simulation/amoebaConcentrationField_2D.txt")

    pifInitializer = cc3d.ElementCC3D("Steppable", {"Type": "PIFInitializer"})
    pifInitializer.ElementCC3D("PIFName", {}, "Simulation/amoebae_2D.piff")

    CompuCellSetup.setSimulationXMLDescription(cc3d)


configure_simulation()
CompuCellSetup.run()
