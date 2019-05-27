from cc3d import CompuCellSetup


def configure_simulation():
    from cc3d.core.XMLUtils import ElementCC3D

    dim = 300
    cc3d = ElementCC3D("CompuCell3D")

    metadata = cc3d.ElementCC3D("Metadata")
    metadata.ElementCC3D("NumberOfProcessors", {}, 4)

    potts = cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions", {"x": dim, "y": dim, "z": dim})
    potts.ElementCC3D("Steps", {}, 100)
    potts.ElementCC3D("Temperature", {}, 0)
    potts.ElementCC3D("Flip2DimRatio", {}, 0.0)

    cellType = cc3d.ElementCC3D("Plugin", {"Name": "CellType"})
    cellType.ElementCC3D("CellType", {"TypeName": "Medium", "TypeId": "0"})

    flexDiffSolver = cc3d.ElementCC3D("Steppable", {"Type": "DiffusionSolverFE"})
    diffusionField = flexDiffSolver.ElementCC3D("DiffusionField")
    diffusionData = diffusionField.ElementCC3D("DiffusionData")
    diffusionData.ElementCC3D("FieldName", {}, "FGF")
    diffusionData.ElementCC3D("DiffusionConstant", {}, 0.10)
    diffusionData.ElementCC3D("DecayConstant", {}, 0.0)
    diffusionData.ElementCC3D("ConcentrationFileName", {}, "Simulation/diffusion_3D.pulse.txt")

    CompuCellSetup.setSimulationXMLDescription(cc3d)


configure_simulation()
CompuCellSetup.run()


