from .elasticitytestSteppables import ElasticityLocalSteppable
from cc3d import CompuCellSetup


def configure_simulation():
    from cc3d.core.XMLUtils import ElementCC3D

    cc3d = ElementCC3D("CompuCell3D")
    potts = cc3d.ElementCC3D("Potts")
    potts.ElementCC3D("Dimensions", {"x": 100, "y": 100, "z": 1})
    potts.ElementCC3D("Steps", {}, 10000)
    potts.ElementCC3D("Temperature", {}, 5)
    potts.ElementCC3D("NeighborOrder", {}, 2)
    potts.ElementCC3D("Boundary_x", {}, "Periodic")

    cell_type = cc3d.ElementCC3D("Plugin", {"Name": "CellType"})
    cell_type.ElementCC3D("CellType", {"TypeName": "Medium", "TypeId": "0"})
    cell_type.ElementCC3D("CellType", {"TypeName": "Body1", "TypeId": "1"})
    cell_type.ElementCC3D("CellType", {"TypeName": "Body2", "TypeId": "2"})
    cell_type.ElementCC3D("CellType", {"TypeName": "Body3", "TypeId": "3"})

    volume = cc3d.ElementCC3D("Plugin", {"Name": "Volume"})
    volume.ElementCC3D("TargetVolume", {}, 25)
    volume.ElementCC3D("LambdaVolume", {}, 4.0)

    contact = cc3d.ElementCC3D("Plugin", {"Name": "Contact"})
    contact.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "Medium"}, 0)
    contact.ElementCC3D("Energy", {"Type1": "Body1", "Type2": "Body1"}, 16)
    contact.ElementCC3D("Energy", {"Type1": "Body1", "Type2": "Medium"}, 4)
    contact.ElementCC3D("Energy", {"Type1": "Body2", "Type2": "Body2"}, 16)
    contact.ElementCC3D("Energy", {"Type1": "Body2", "Type2": "Medium"}, 4)
    contact.ElementCC3D("Energy", {"Type1": "Body3", "Type2": "Body3"}, 16)
    contact.ElementCC3D("Energy", {"Type1": "Body3", "Type2": "Medium"}, 4)
    contact.ElementCC3D("Energy", {"Type1": "Body1", "Type2": "Body2"}, 16)
    contact.ElementCC3D("Energy", {"Type1": "Body1", "Type2": "Body3"}, 16)
    contact.ElementCC3D("Energy", {"Type1": "Body2", "Type2": "Body3"}, 16)
    contact.ElementCC3D("neighborOrder", {}, 2)

    center_of_mass = cc3d.ElementCC3D("Plugin", {"Name": "CenterOfMass"})

    elasticity_tracker = cc3d.ElementCC3D("Plugin", {"Name": "ElasticityTracker"})
    elasticity_tracker.ElementCC3D("IncludeType", {}, "Body1")
    elasticity_tracker.ElementCC3D("IncludeType", {}, "Body2")
    elasticity_tracker.ElementCC3D("IncludeType", {}, "Body3")

    elasticity_energy = cc3d.ElementCC3D("Plugin", {"Name": "ElasticityEnergy"})
    elasticity_energy.ElementCC3D("Local", {})
    # elasticity_energy.ElementCC3D("LambdaElasticity",{},200.0)
    # elasticity_energy.ElementCC3D("TargetLengthElasticity",{},6)

    external_potential = cc3d.ElementCC3D("Plugin", {"Name": "ExternalPotential"})
    external_potential.ElementCC3D("Lambda", {"x": -10, "y": 0, "z": 0})

    pif_initializer = cc3d.ElementCC3D("Steppable", {"Type": "PIFInitializer"})
    pif_initializer.ElementCC3D("PIFName", {}, "Simulation/elasticitytest.piff")

    CompuCellSetup.setSimulationXMLDescription(cc3d)


configure_simulation()

CompuCellSetup.register_steppable(steppable=ElasticityLocalSteppable(frequency=50))


CompuCellSetup.run()

