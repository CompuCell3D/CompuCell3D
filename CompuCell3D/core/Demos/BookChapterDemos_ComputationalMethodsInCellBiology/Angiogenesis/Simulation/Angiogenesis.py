from .AngiogenesisSteppables import AngiogenesisStetppable
from cc3d import CompuCellSetup


def configure_simulation():
    from cc3d.core.XMLUtils import ElementCC3D

    CompuCell3DElmnt = ElementCC3D("CompuCell3D", {"version": "3.6.0"})

    # Basic properties of CPM (GGH) algorithm
    potts_el = CompuCell3DElmnt.ElementCC3D("Potts")
    potts_el.ElementCC3D("Dimensions", {"x": "50", "y": "50", "z": "50"})
    potts_el.ElementCC3D("Steps", {}, "10000")
    potts_el.ElementCC3D("Temperature", {}, "20.0")
    potts_el.ElementCC3D("NeighborOrder", {}, "3")
    potts_el.ElementCC3D("Boundary_x", {}, "Periodic")
    potts_el.ElementCC3D("Boundary_y", {}, "Periodic")
    potts_el.ElementCC3D("Boundary_z", {}, "Periodic")

    # Listing all cell types in the simulation
    cell_type_el = CompuCell3DElmnt.ElementCC3D("Plugin", {"Name": "CellType"})
    cell_type_el.ElementCC3D("CellType", {"TypeId": "0", "TypeName": "Medium"})
    cell_type_el.ElementCC3D("CellType", {"TypeId": "1", "TypeName": "Endothelial"})

    # Constraint on cell volume. Each cell type has different constraint.
    # For more flexible specification of the constraint (done in Python) please use VolumeLocalFlex plugin
    vol_plug_elem = CompuCell3DElmnt.ElementCC3D("Plugin", {"Name": "Volume"})
    vol_plug_elem.ElementCC3D("VolumeEnergyParameters",
                              {"CellType": "Endothelial", "LambdaVolume": "20.0", "TargetVolume": "74"})

    # Specification of adhesion energies
    contact_plug_el = CompuCell3DElmnt.ElementCC3D("Plugin", {"Name": "Contact"})
    contact_plug_el.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "Medium"}, "0")
    contact_plug_el.ElementCC3D("Energy", {"Type1": "Medium", "Type2": "Endothelial"}, "12")
    contact_plug_el.ElementCC3D("Energy", {"Type1": "Endothelial", "Type2": "Endothelial"}, "5")
    contact_plug_el.ElementCC3D("NeighborOrder", {}, "4")

    # Specification of chemotaxis properties of select cell types.
    chem_plug_el = CompuCell3DElmnt.ElementCC3D("Plugin", {"Name": "Chemotaxis"})
    chemical_field_el = chem_plug_el.ElementCC3D("ChemicalField",
                                                   {"Name": "VEGF", "Source": "FlexibleDiffusionSolverFE"})
    chemical_field_el.ElementCC3D("ChemotaxisByType",
                                   {"ChemotactTowards": "Medium", "Lambda": "6000.0", "Type": "Endothelial"})

    # Specification of secretion properties of select cell types.
    # Specification of secretion properties of individual cells can be done in Python
    secr_plug_el = CompuCell3DElmnt.ElementCC3D("Plugin", {"Name": "Secretion"})
    field_el = secr_plug_el.ElementCC3D("Field", {"Name": "VEGF"})
    field_el.ElementCC3D("Secretion", {"Type": "Endothelial"}, "0.013")

    # Module allowing multiple calls of the PDE solver. By default number of extra calls is set to 0.
    # Change these settings to desired values after consulting CC3D manual on how to work with large diffusion constants
    # (>0.16 in 3D with DeltaX=1.0 and DeltaT=1.0)
    pde_solv_call_plug_el = CompuCell3DElmnt.ElementCC3D("Plugin", {"Name": "PDESolverCaller"})
    pde_solv_call_plug_el.ElementCC3D("CallPDE", {"ExtraTimesPerMC": "0", "PDESolverName": "FlexibleDiffusionSolverFE"})

    # Specification of PDE solvers
    flex_diff_solv_el = CompuCell3DElmnt.ElementCC3D("Steppable", {"Type": "FlexibleDiffusionSolverFE"})
    diff_field_elmnt = flex_diff_solv_el.ElementCC3D("DiffusionField")
    diff_data_el = diff_field_elmnt.ElementCC3D("DiffusionData")
    diff_data_el.ElementCC3D("FieldName", {}, "VEGF")
    diff_data_el.ElementCC3D("DiffusionConstant", {}, "0.16")
    diff_data_el.ElementCC3D("DecayConstant", {}, "0.016")

    # Additional options are:
    # DiffusionDataElmnt.ElementCC3D("DoNotDiffuseTo",{},"LIST YOUR CELL TYPES HERE")
    diff_data_el.ElementCC3D("DoNotDecayIn", {}, "Endothelial")
    # DiffusionDataElmnt.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with
    # path Simulation/NAME_OF_THE_FILE.txt")
    # To run solver for large diffusion constants you typically call solver multiple times - ExtraTimesPerMCS
    # to specify additional calls to the solver in each MCS
    # IMPORTANT: make sure not to mix this setting with the PDESolverCaller module! See manual for more information
    # DiffusionDataElmnt.ElementCC3D("ExtraTimesPerMCS",{},"0")
    diff_data_el.ElementCC3D("DeltaX", {}, "1.0")
    diff_data_el.ElementCC3D("DeltaT", {}, "1.0")

    # Initial layout of cells in the form of rectangular slab
    uni_init_el = CompuCell3DElmnt.ElementCC3D("Steppable", {"Type": "UniformInitializer"})
    region_el = uni_init_el.ElementCC3D("Region")
    region_el.ElementCC3D("BoxMin", {"x": "15", "y": "15", "z": "15"})
    # region_el.ElementCC3D("BoxMax", {"x": "19", "y": "19", "z": "19"})
    region_el.ElementCC3D("BoxMax", {"x": "35", "y": "35", "z": "35"})
    region_el.ElementCC3D("Gap", {}, "0")
    region_el.ElementCC3D("Width", {}, "4")
    region_el.ElementCC3D("Types", {}, "Endothelial")

    CompuCellSetup.set_simulation_xml_description(CompuCell3DElmnt)

configure_simulation()

CompuCellSetup.register_steppable(steppable=AngiogenesisStetppable(frequency=100))

CompuCellSetup.run()

