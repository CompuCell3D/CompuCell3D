from cc3d import CompuCellSetup
from .DeltaNotchSteppables import DeltaNotchClass
from .DeltaNotchSteppables import ExtraFields


def configure_simulation():
    from cc3d.core.XMLUtils import ElementCC3D

    cc3d = ElementCC3D("CompuCell3D")
    potts = cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions", {"x": 42, "y": 42, "z": 1})
    potts.ElementCC3D("Steps", {}, 10000)
    potts.ElementCC3D("Anneal", {}, 10)
    potts.ElementCC3D("Temperature", {}, 10)
    potts.ElementCC3D("NeighborOrder", {}, 2)
    potts.ElementCC3D("Boundary_x", {}, "Periodic")
    potts.ElementCC3D("Boundary_y", {}, "Periodic")

    cellType = cc3d.ElementCC3D("Plugin", {"Name": "CellType"})
    cellType.ElementCC3D("CellType", {"TypeName": "Medium", "TypeId": "0"})
    cellType.ElementCC3D("CellType", {"TypeName": "TypeA", "TypeId": "1"})

    contact = cc3d.ElementCC3D("Plugin", {"Name": "Contact"})
    contact.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "Medium"}, 0)
    contact.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "TypeA"}, 5)
    contact.ElementCC3D("Energy", {"Type1": "TypeA", "Type2": "TypeA"}, 5)
    contact.ElementCC3D("NeighborOrder", {}, 4)

    vp = cc3d.ElementCC3D("Plugin", {"Name": "Volume"})
    vp.ElementCC3D("TargetVolume", {}, 49)
    vp.ElementCC3D("LambdaVolume", {}, 5)

    ntp = cc3d.ElementCC3D("Plugin", {"Name": "NeighborTracker"})

    uipd = cc3d.ElementCC3D("Steppable", {"Type": "UniformInitializer"})
    region = uipd.ElementCC3D("Region")
    region.ElementCC3D("BoxMin", {"x": 0, "y": 0, "z": 0})
    region.ElementCC3D("BoxMax", {"x": 42, "y": 42, "z": 1})
    region.ElementCC3D("Types", {}, "TypeA")
    region.ElementCC3D("Width", {}, 7)

    CompuCellSetup.setSimulationXMLDescription(cc3d)


CompuCellSetup.register_steppable(steppable=DeltaNotchClass(frequency=1))
CompuCellSetup.register_steppable(steppable=ExtraFields(frequency=1))

configure_simulation()


CompuCellSetup.run()


