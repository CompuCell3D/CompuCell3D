
import os
import sys

import CompuCell
from PySteppables import *
from PySteppablesExamples import MitosisSteppableBase

import bionetAPI

class MultiscaleSimulationSteppable( SteppableBasePy ):
    def __init__( self, _simulator, _frequency = 1 ):
        SteppableBasePy.__init__( self,_simulator, _frequency )
        
        bionetAPI.initializeBionetworkManager( self.simulator )
    
    def start(self):
        
        # #################### Create SBML models ######################
        ## Create a bionetwork SBML model named "DeltaNotchModel"
        sbmlModelName = "DeltaNotchModel"
        sbmlModelKey = "DN"
        sbmlModelPath =  "Simulation/MinimalDeltaNotch.sbml"
        timeStepOfIntegration = 0.05
        bionetAPI.loadSBMLModel( sbmlModelName, sbmlModelPath, sbmlModelKey, timeStepOfIntegration )
        
        ## Create a bionetwork SBML model named "CadherinCatenin"
        sbmlModelPath =  "Simulation/CadherinCatenin_RamisConde2008.sbml"
        bionetAPI.loadSBMLModel( "CadherinCatenin", sbmlModelPath, "CC", 0.05 )
        
        ## Create a bionetwork SBML model named "BloodLiverPK"
        sbmlModelPath =  "Simulation/PK_BloodLiver.sbml"
        bionetAPI.loadSBMLModel( "BloodLiverPK", sbmlModelPath, "BLPK", 0.05 )
        
        ## Create a bionetwork SBML model named "SimpleExample"
        sbmlModelPath =  "Simulation/SimpleExample.sbml"
        bionetAPI.loadSBMLModel( "SimpleExample", sbmlModelPath, "SE", 0.05 )
        
        
        # ################ Add SBML models to bionetwork template libraries ##################
        
        ## Add SBML model to CellTypeA
        bionetAPI.addSBMLModelToTemplateLibrary( "DeltaNotchModel", "CellTypeA" )
        bionetAPI.addSBMLModelToTemplateLibrary( "BloodLiverPK", "CellTypeA" )
        
        ## Add SBML models to CellTypeB
        bionetAPI.addSBMLModelToTemplateLibrary( "DeltaNotchModel", "CellTypeB" )
        bionetAPI.addSBMLModelToTemplateLibrary( "CadherinCatenin", "CellTypeB" )
        
        ## Add SBML models to CellTypeC
        bionetAPI.addSBMLModelToTemplateLibrary( "CadherinCatenin", "CellTypeC" )
        
        ## Add SBML models to CellTypeD
        bionetAPI.addSBMLModelToTemplateLibrary( "BloodLiverPK", "CellTypeD" )
        bionetAPI.addSBMLModelToTemplateLibrary( "SimpleExample", "CellTypeD" )
        
        # ####### Set initial conditions for SBML properties #########
        
        ## Set global (for all bionetwork template libraries) SBML initial conditions
        bionetAPI.setBionetworkInitialCondition( "Global", "DN_di", 0.4 )
        bionetAPI.setBionetworkInitialCondition( "Global", "CC_beta", 24 )
        
        bionetAPI.setBionetworkInitialCondition( "Global", "Ab", 1.1 )
        
        ## Set SBML initial conditions for CellTypeA
        bionetAPI.setBionetworkInitialCondition( "CellTypeA", "DN_ni", 0.8 )
        
        ## Set SBML initial conditions for CellTypeB
        bionetAPI.setBionetworkInitialCondition( "CellTypeB", "CC_Ebeta", 33.333 )
        bionetAPI.setBionetworkInitialCondition( "CellTypeB", "BLPK_A1", 0.44444 )
        
        ## Set SBML initial conditions for CellTypeC
        bionetAPI.setBionetworkInitialCondition( "CellTypeC", "CC_Ebeta", 4.32111 )
        bionetAPI.setBionetworkInitialCondition( "CellTypeC", "SE_S1", 0.66666666 )
        bionetAPI.setBionetworkInitialCondition( "CellTypeC", "CLK_Nan1", 0.55555 )
        
        ## Set SBML initial conditions for CellTypeD
        bionetAPI.setBionetworkInitialCondition( "CellTypeD", "CC_Emem", 12.34567 )
        bionetAPI.setBionetworkInitialCondition( "CellTypeD", "CLK_Nan1", 0.77777 )
        bionetAPI.setBionetworkInitialCondition( "CellTypeD", "DN_ni", 0.88888888 )
        
        ## Setting SBML initial conditions for non-existent cell type
        bionetAPI.setBionetworkInitialCondition( "NonExistentCellType", "SE_S1", 0.62 )
        
        
        # ######## Create bionetworks and initialize their states ##########
        bionetAPI.initializeBionetworks()
        
        
        for cell in self.cellList:
            dictionaryAttrib = CompuCell.getPyAttrib( cell )
            dictionaryAttrib["InitialVolume"] = cell.volume
            dictionaryAttrib["DivideVolume"] = 280.
            cell.targetVolume = 100.0
            cell.lambdaVolume = 2.0
        
        import CompuCellSetup
        self.cellTypeMap = CompuCellSetup.ExtractTypeNamesAndIds()
        del(self.cellTypeMap[0])
        
    def step(self, mcs):
        
        # ######### Update all bionetwork integrator(s) ###########
        bionetAPI.timestepBionetworks()
        
        print "\n"; print bionetAPI.getBionetworkValue( "dii", "CellTypeB" )
        print "\n"; print bionetAPI.getBionetworkValue( "di", "CellTypeB" )
        print "\n"; print bionetAPI.getBionetworkValue( "DN_di", "CellTypeB" )
        print "\n"; print bionetAPI.getBionetworkValue( "DeltaNotchModel_di", "CellTypeB" )
        
        bionetAPI.setBionetworkValue( "SE_k1", 0.1, "CellTypeD" )
        bionetAPI.setBionetworkValue( "SE_k2", 0.3, "CellTypeD" )
        
        cellType = "CellTypeD"; propertyName = "SE_S1"
        S1 = bionetAPI.getBionetworkValue( propertyName, cellType )
        print "Value of %s in %s: %s" % ( propertyName, cellType, S1 )
        
        
        if( mcs >= 20 ):
            
            if( mcs == 40 ):
                if self.cellTypeMap[bionetAPI.getCC3DCellByID(1).type] == "CellTypeB":
                    bionetAPI.setBionetworkValue( "TemplateLibrary", "CellTypeA", 1 )
                elif self.cellTypeMap[bionetAPI.getCC3DCellByID(1).type] == "CellTypeA":
                    bionetAPI.setBionetworkValue( "TemplateLibrary", "CellTypeB", "CellTypeA" )
            
            if( mcs < 120 ):
                cell1_di = bionetAPI.getBionetworkValue( "di", 1 )
                
                if( cell1_di > 0.0 ):  newPropertyValue = 4.0 / cell1_di
                else:  newPropertyValue = 10.0
                
                for cell in self.cellList:
                    if self.cellTypeMap[cell.type] == "CellTypeB":
                        cell.targetVolume = newPropertyValue


class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, _simulator, _frequency=1):
        MitosisSteppableBase.__init__(self, _simulator, _frequency)
    
    def step(self,mcs):
        cells_to_divide=[]
        for cell in self.cellList:
            dictionaryAttrib = CompuCell.getPyAttrib( cell )
            if cell.volume > dictionaryAttrib["DivideVolume"]:
                cells_to_divide.append(cell)
                
        for cell in cells_to_divide:
            self.divideCellRandomOrientation(cell)

    def updateAttributes(self):
        childCell = self.mitosisSteppable.childCell
        parentCell = self.mitosisSteppable.parentCell
        
        dictionaryAttrib = CompuCell.getPyAttrib( childCell )
        dictionaryAttrib["InitialVolume"] = childCell.volume
        dictionaryAttrib["DivideVolume"] = 280.
        
        print "Child cell ID: %s" % childCell.id
        print "Parent cell ID: %s" % parentCell.id
        
        parentCell.targetVolume = parentCell.volume;
        childCell.targetVolume = childCell.volume;
        
        parentCell.targetSurface = parentCell.surface;
        childCell.targetSurface = childCell.surface;
        
        childCell.type = parentCell.type;
        childCell.lambdaVolume = parentCell.lambdaVolume;
        childCell.lambdaSurface = parentCell.lambdaSurface;
        
        bionetAPI.copyBionetworkFromParent( parentCell, childCell )






