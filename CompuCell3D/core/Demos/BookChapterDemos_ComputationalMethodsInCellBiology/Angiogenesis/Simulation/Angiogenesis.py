
def configureSimulation(sim):
    import CompuCellSetup
    from XMLUtils import ElementCC3D
    
    CompuCell3DElmnt=ElementCC3D("CompuCell3D",{"version":"3.6.0"})
    
    # Basic properties of CPM (GGH) algorithm
    PottsElmnt=CompuCell3DElmnt.ElementCC3D("Potts")
    PottsElmnt.ElementCC3D("Dimensions",{"x":"50","y":"50","z":"50"})
    PottsElmnt.ElementCC3D("Steps",{},"10000")
    PottsElmnt.ElementCC3D("Temperature",{},"20.0")
    PottsElmnt.ElementCC3D("NeighborOrder",{},"3")
    PottsElmnt.ElementCC3D("Boundary_x",{},"Periodic")
    PottsElmnt.ElementCC3D("Boundary_y",{},"Periodic")
    PottsElmnt.ElementCC3D("Boundary_z",{},"Periodic")
    
    # Listing all cell types in the simulation
    PluginElmnt=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"CellType"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"0","TypeName":"Medium"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"1","TypeName":"Endothelial"})
    
    # Constraint on cell volume. Each cell type has different constraint.
    # For more flexible specification of the constraint (done in Python) please use VolumeLocalFlex plugin
    PluginElmnt_1=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Volume"})
    PluginElmnt_1.ElementCC3D("VolumeEnergyParameters",{"CellType":"Endothelial","LambdaVolume":"20.0","TargetVolume":"74"})
    
    # Specification of adhesion energies
    PluginElmnt_2=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Contact"})
    PluginElmnt_2.ElementCC3D("Energy",{"Type1":"Medium","Type2":"Medium"},"0")
    PluginElmnt_2.ElementCC3D("Energy",{"Type1":"Medium","Type2":"Endothelial"},"12")
    PluginElmnt_2.ElementCC3D("Energy",{"Type1":"Endothelial","Type2":"Endothelial"},"5")
    PluginElmnt_2.ElementCC3D("NeighborOrder",{},"4")
    
    # Specification of chemotaxis properties of select cell types.
    PluginElmnt_3=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Chemotaxis"})
    ChemicalFieldElmnt=PluginElmnt_3.ElementCC3D("ChemicalField",{"Name":"VEGF","Source":"FlexibleDiffusionSolverFE"})
    ChemicalFieldElmnt.ElementCC3D("ChemotaxisByType",{"ChemotactTowards":"Medium","Lambda":"6000.0","Type":"Endothelial"})
    
    # Specification of secretion properties of select cell types.
    # Specification of secretion properties of individual cells can be done in Python
    PluginElmnt_4=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Secretion"})
    FieldElmnt=PluginElmnt_4.ElementCC3D("Field",{"Name":"VEGF"})
    FieldElmnt.ElementCC3D("Secretion",{"Type":"Endothelial"},"0.013")
    
    # Module allowing multiple calls of the PDE solver. By default number of extra calls is set to 0. 
    # Change these settings to desired values after consulting CC3D manual on how to work with large diffusion constants (>0.16 in 3D with DeltaX=1.0 and DeltaT=1.0)
    PluginElmnt_5=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"PDESolverCaller"})
    PluginElmnt_5.ElementCC3D("CallPDE",{"ExtraTimesPerMC":"0","PDESolverName":"FlexibleDiffusionSolverFE"})
    
    # Specification of PDE solvers
    SteppableElmnt=CompuCell3DElmnt.ElementCC3D("Steppable",{"Type":"FlexibleDiffusionSolverFE"})
    DiffusionFieldElmnt=SteppableElmnt.ElementCC3D("DiffusionField")
    DiffusionDataElmnt=DiffusionFieldElmnt.ElementCC3D("DiffusionData")
    DiffusionDataElmnt.ElementCC3D("FieldName",{},"VEGF")
    DiffusionDataElmnt.ElementCC3D("DiffusionConstant",{},"0.16")
    DiffusionDataElmnt.ElementCC3D("DecayConstant",{},"0.016")
    # Additional options are:
    #DiffusionDataElmnt.ElementCC3D("DoNotDiffuseTo",{},"LIST YOUR CELL TYPES HERE")
    DiffusionDataElmnt.ElementCC3D("DoNotDecayIn",{},"Endothelial")
    # DiffusionDataElmnt.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
    # To run solver for large diffusion constants you typically call solver multiple times - ExtraTimesPerMCS to specify additional calls to the solver in each MCS 
    # IMPORTANT: make sure not to mix this setting with the PDESolverCaller module! See manual for more information
    # DiffusionDataElmnt.ElementCC3D("ExtraTimesPerMCS",{},"0")
    DiffusionDataElmnt.ElementCC3D("DeltaX",{},"1.0")
    DiffusionDataElmnt.ElementCC3D("DeltaT",{},"1.0")
    
    # Initial layout of cells in the form of rectangular slab
    SteppableElmnt_1=CompuCell3DElmnt.ElementCC3D("Steppable",{"Type":"UniformInitializer"})
    RegionElmnt=SteppableElmnt_1.ElementCC3D("Region")
    RegionElmnt.ElementCC3D("BoxMin",{"x":"15","y":"15","z":"15"})
    RegionElmnt.ElementCC3D("BoxMax",{"x":"35","y":"35","z":"35"})
    RegionElmnt.ElementCC3D("Gap",{},"0")
    RegionElmnt.ElementCC3D("Width",{},"4")
    RegionElmnt.ElementCC3D("Types",{},"Endothelial")

            
    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)
            
import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup


sim,simthread = CompuCellSetup.getCoreSimulationObjects()
        
configureSimulation(sim)            
                
# add extra attributes here
            
CompuCellSetup.initializeSimulationObjects(sim,simthread)
# Definitions of additional Python-managed fields go here
        
#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()
        
from AngiogenesisSteppables import AngiogenesisStetppable
steppableInstance=AngiogenesisStetppable(sim,_frequency=100)
steppableRegistry.registerSteppable(steppableInstance)
        
CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
        
        