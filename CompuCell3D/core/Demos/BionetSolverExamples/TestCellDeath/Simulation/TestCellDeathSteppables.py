
import os
import sys

import CompuCell
from PySteppables import *
from PySteppablesExamples import MitosisSteppableBase

import bionetAPI

class TestCellDeathSteppable( SteppableBasePy ):
    def __init__( self, _simulator, _frequency = 1 ):
        SteppableBasePy.__init__( self,_simulator, _frequency )
        self.simulator = _simulator
        self.inventory = self.simulator.getPotts().getCellInventory()
        self.cellList = CellList(self.inventory)
        
        bionetAPI.initializeBionetworkManager( self.simulator )
    
    def start(self):
        
        # #################### Create SBML models ######################
        sbmlModelName = "DeltaNotchModel"
        sbmlModelKey = "DN"
        sbmlModelPath =  "Simulation/MinimalDeltaNotch.sbml"
        timeStepOfIntegration = 0.1
        bionetAPI.loadSBMLModel( sbmlModelName, sbmlModelPath, sbmlModelKey, timeStepOfIntegration )
        
        # ################ Add SBML models to cell types ##################
        bionetAPI.addSBMLModelToTemplateLibrary( "DeltaNotchModel", "LowDelta" )
        bionetAPI.addSBMLModelToTemplateLibrary( "DeltaNotchModel", "HighDelta" )
        
        ## Include this for a test
        bionetAPI.addSBMLModelToTemplateLibrary( "DeltaNotchModel", "External" )
        
        # ####### Set initial conditions for SBML properties #########
        bionetAPI.setBionetworkInitialCondition( "LowDelta", "DN_di", 0.2 )
        bionetAPI.setBionetworkInitialCondition( "HighDelta", "DN_di", 0.8 )
        
        for cell in self.cellList:
            cell.targetVolume = 32.0
            cell.lambdaVolume = 1.0
            cell.targetSurface = 32.0
            cell.lambdaSurface = 1.0
        
        # ######## Create bionetworks and initialize their states ##########
        bionetAPI.initializeBionetworks()
        
        # ######## Set cell initial conditions by individual cell ##########
        for cell in self.cellList:
            dictionaryAttrib = CompuCell.getPyAttrib( cell )
            dictionaryAttrib["InitialVolume"] = cell.volume
            dictionaryAttrib["DivideVolume"] = 2.*cell.volume
        
        self.initialNumberOfCells = len(self.cellList)
        print "\n\nNumber of cells: %s\n\n" % self.initialNumberOfCells
        
        import CompuCellSetup
        self.cellTypeMap = CompuCellSetup.ExtractTypeNamesAndIds()
        del(self.cellTypeMap[0])
        
    def step(self, mcs):
        
        # ######### Update bionetwork integrator(s) for all cells ###########
        bionetAPI.timestepBionetworks()
        
        # ######## Implement cell growth by increasing target volume ##########
        for cell in self.cellList:
            if cell.id > self.initialNumberOfCells:
                ## All new cells are assigned a target volume of 0 to force cell death
                cell.targetVolume = 0.
                cell.lambdaVolume = 2.
                cell.targetSurface = 0.
                cell.lambdaSurface = 2.
            else:
                dictionaryAttrib = CompuCell.getPyAttrib( cell )
                cell.targetVolume = cell.volume + 0.15*dictionaryAttrib["InitialVolume"]
                
        # ###### Retrieve delta values and set cell type according to delta concentration ########
        for cell in self.cellList:
            currentDeltaValue = bionetAPI.getBionetworkValue( "DN_di", cell.id )
            if( currentDeltaValue > 0.5 ):
                if( self.cellTypeMap[cell.type] == "LowDelta" ):
                    print "ID of cell to undergo type change %s" % cell.id
                    print "Type of cell to undergo type change %s" % bionetAPI.getBionetworkValue( "TemplateLibrary", cell.id )
                    bionetAPI.setBionetworkValue( "TemplateLibrary", "HighDelta", cell.id )
            else:
                if( self.cellTypeMap[cell.type] == "HighDelta" ):
                    bionetAPI.setBionetworkValue( "TemplateLibrary", "LowDelta", cell.id )
        
        # ####### Set all cell dbari values as a function of neighbor delta values #########
        for cell in self.cellList:
            weightedSumOfNeighborDeltaValues = 0.0
            neighborContactAreas = bionetAPI.getNeighborContactAreas( cell.id )
            neighborDeltaValues = bionetAPI.getNeighborProperty( "DN_di", cell.id )
            
            for neighborID in neighborContactAreas.keys():
                weightedSumOfNeighborDeltaValues += (neighborContactAreas[neighborID] * neighborDeltaValues[neighborID])
                
            bionetAPI.setBionetworkValue( "DN_dbari", weightedSumOfNeighborDeltaValues/cell.surface, cell.id )
        

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
        dictionaryAttrib["DivideVolume"] = 2.*childCell.volume
        
        print "Child cell ID: %s" % childCell.id
        print "Parent cell ID: %s" % parentCell.id
        
        parentCell.targetVolume = parentCell.volume;
        childCell.targetVolume = childCell.volume;
        
        parentCell.targetSurface = parentCell.surface;
        childCell.targetSurface = childCell.surface;
        
        childCell.type = parentCell.type;
        childCell.lambdaVolume = parentCell.lambdaVolume;
        childCell.lambdaSurface = parentCell.lambdaSurface;
        
        #NOTE: This call should be after we set basic properties for childCell.
        #      Otherwise soslib may pull up wrong model and associate it with cell
        bionetAPI.copyBionetworkFromParent( parentCell, childCell )
        






