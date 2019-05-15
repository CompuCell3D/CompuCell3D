from cc3d import CompuCellSetup


def configure_simulation():
    from cc3d.core.XMLUtils import ElementCC3D

    cc3d = ElementCC3D("CompuCell3D")
    potts = cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions", {"x": 100, "y": 100, "z": 1})
    potts.ElementCC3D("Steps", {}, 1000)
    potts.ElementCC3D("Temperature", {}, 10)
    potts.ElementCC3D("NeighborOrder", {}, 2)

    cellType = cc3d.ElementCC3D("Plugin", {"Name": "CellType"})
    cellType.ElementCC3D("CellType", {"TypeName": "Medium", "TypeId": "0"})
    cellType.ElementCC3D("CellType", {"TypeName": "Condensing", "TypeId": "1"})
    cellType.ElementCC3D("CellType", {"TypeName": "NonCondensing", "TypeId": "2"})

    volume = cc3d.ElementCC3D("Plugin", {"Name": "Volume"})
    volume.ElementCC3D("TargetVolume", {}, 25)
    volume.ElementCC3D("LambdaVolume", {}, 2.0)

    contact = cc3d.ElementCC3D("Plugin", {"Name": "Contact"})
    contact.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "Medium"}, 0)
    contact.ElementCC3D("Energy", {"Type1": "NonCondensing", "Type2": "NonCondensing"}, 16)
    contact.ElementCC3D("Energy", {"Type1": "Condensing", "Type2": "Condensing"}, 2)
    contact.ElementCC3D("Energy", {"Type1": "NonCondensing", "Type2": "Condensing"}, 11)
    contact.ElementCC3D("Energy", {"Type1": "NonCondensing", "Type2": "Medium"}, 16)
    contact.ElementCC3D("Energy", {"Type1": "Condensing", "Type2": "Medium"}, 16)

    blobInitializer = cc3d.ElementCC3D("Steppable", {"Type": "BlobInitializer"})
    blobInitializer.ElementCC3D("Gap", {}, 0)
    blobInitializer.ElementCC3D("Width", {}, 5)
    blobInitializer.ElementCC3D("CellSortInit", {}, "yes")
    blobInitializer.ElementCC3D("Radius", {}, 40)

    CompuCellSetup.setSimulationXMLDescription(cc3d)


configure_simulation()

CompuCellSetup.run()
