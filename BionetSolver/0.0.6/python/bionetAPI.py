
"""
Question/Suggestion: when you create bionetwork object, you attach it to a cell and it lives there so any references to such bionetwork are via cell object (CC3D cell).  For non-cell bionetworks bionetworkManager or bionetwork warehouse sould store them 
and have them searchable by names (or by ids?). 

We also store bionetworks templates  - i.e. SBML models which are not attached to any cell and they are not even instances of bionetworkIntegrator - they have to be loaded to cell or instantiated on their own as non-cell bionetwork instance and then they can be integrated.
Bionetwork templates should be searchable by name because they are associated with cell type name, or SBML model name etc. 

Based on above we need two types of storage one for bionetwork templates and one for bionetwork instances( only for non-cellular bionetworks, as cellular bionetworks are attached to cell )

Is this how soslib works now? If so I would celanup the interface (API) you use to access the two storage types - looks like mainly fcn name changes will be reqauired. So this is next step in cleaning the soslib

"""

# Rename CC3DSOSlibPy ??? - how about sosAPILib ? 
from BionetSolverPy import *

import sys
import CompuCell
import CompuCellSetup
from PySteppables import *

# As a rule - classes names should begin with capital letter - variables names shuold use lower case first letter

# I would rename SOSPy to bionetworkManager
global bionetworkManager
global pyAttributeAdder
global dictAdder

bionetworkManager = None
pyAttributeAdder = None
dictAdder = None

#   initializeBionetworkManager( _cc3dSimulator )
def initializeBionetworkManager( _cc3dSimulator ):
    global bionetworkManager
    global pyAttributeAdder
    global dictAdder
    
    if( bionetworkManager == None ):
        pyAttributeAdder, dictAdder = CompuCellSetup.attachDictionaryToCells( _cc3dSimulator )
        bionetworkManager = BionetworkManager( _cc3dSimulator )

def printInitializationWarningMessage():
    print "\n** WARNING: BionetSolver API not initialized **"
    print "-- initializeBionetworkManager( <CC3D simulator object> ) must first be called\n"
    sys.stdout.flush()


# ################### CORE BionetworkManager API FUNCTIONS #####################
#   loadSBMLModel( _sbmlModelName, _sbmlModelPath, _modelKey = "", _timeStepOfIntegration = -1.0 )
def loadSBMLModel( _sbmlModelName, _sbmlModelPath, _modelKey = "", _timeStepOfIntegration = -1.0 ):
    global bionetworkManager
    print "INSIDE bionetAPI loading fcn"
    if( bionetworkManager != None ):
        bionetworkManager.loadSBMLModel( _sbmlModelName, _sbmlModelPath, _modelKey, _timeStepOfIntegration )
    else:
        printInitializationWarningMessage()

#addSBMLModelToWarehouse( _sbmlModelName, _warehouseName )
#OR addSBMLModelToTemplateLibrary( _sbmlModelName, _templateLibraryName )
def addSBMLModelToTemplateLibrary( _sbmlModelName, _templateLibraryName ):
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.addSBMLModelToTemplateLibrary( _sbmlModelName, _templateLibraryName )
    else:
        printInitializationWarningMessage()

#   setBionetworkInitialCondition( _templateLibraryName, _propertyName, _propertyValue )
def setBionetworkInitialCondition( _templateLibraryName, _propertyName, _propertyValue ):
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.setBionetworkInitialCondition( _templateLibraryName, _propertyName, _propertyValue )
    else:
        printInitializationWarningMessage()

#   initializeBionetworks()
def initializeBionetworks():
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.initializeBionetworks()
    else:
        printInitializationWarningMessage()

#   timestepBionetworks()
def timestepBionetworks():
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.timestepBionetworks( -1.0 )
    else:
        printInitializationWarningMessage()

#   getBionetworkValue( _propertyName, _currentID )
def getBionetworkValue( _propertyName, _currentID = "Global"):
    global bionetworkManager
    propertyValue = None
    if( bionetworkManager != None ):
        propertyValue = bionetworkManager.getBionetworkValue( _propertyName, _currentID )
    else:
        printInitializationWarningMessage()
    return propertyValue

#   setBionetworkValue( _propertyName, _newPropertyValue, _currentID )
def setBionetworkValue( _propertyName, _newPropertyValue, _currentID = "Global" ):
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.setBionetworkValue( _propertyName, _newPropertyValue, _currentID )
    else:
        printInitializationWarningMessage()

#   copyBionetworkFromParent( _CC3DParentCell, _CC3DChildCell ) - here I would reverse the order of arguments copyBionetworkFromParent( _CC3DParentCell, _CC3DChildCell ) -  this is more standard
def copyBionetworkFromParent( _CC3DParentCell, _CC3DChildCell ):
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.copyBionetworkFromParent( _CC3DParentCell, _CC3DChildCell )
    else:
        printInitializationWarningMessage()


# ################### OTHER USEFUL BionetworkManager API FUNCTIONS #####################
def getCC3DCellByID( cellID ):
    global bionetworkManager
    cc3dCell = None
    if( bionetworkManager != None ):
        cc3dCell = bionetworkManager.getCC3DCellByID( cellID )
    else:
        printInitializationWarningMessage()
    return cc3dCell

#   getBionetworkSBMLNames( cellID )
def getBionetworkSBMLNames( cellID ):
    global bionetworkManager
    sbmlModelNames = None
    if( bionetworkManager != None ):
        sbmlModelNames = bionetworkManager.getBionetworkSBMLNames( cellID )
    else:
        printInitializationWarningMessage()
    return sbmlModelNames

# this goes to CC3D scripts
def generateLinearGradientChemicalField( 
    fileName, directionOfGradient, startConcentration, stopConcentration, pottsDimensions):
    
    directionToPottsMap = {"x":0, "y":1, "z":2}
    directionToPositionMap = {"x":"xpos", "y":"ypos", "z":"zpos"}
    outputFile = open( fileName, "w" )
    concentrationStep = \
        (stopConcentration - startConcentration)/ \
        pottsDimensions[directionToPottsMap[directionOfGradient]]
    for xpos in range(pottsDimensions[0]):
        for ypos in range(pottsDimensions[1]):
            for zpos in range(pottsDimensions[2]):
                currentPosition = \
                    locals()[directionToPositionMap[directionOfGradient]]
                currentConcentration = \
                    startConcentration + currentPosition*concentrationStep
                outputFile.write( "%s %s %s %s\n" % (xpos, ypos, zpos, currentConcentration) )

#   writeBionetworkStateVarNamesToFile( _currentID, _bionetworkSBML, _outputFileName, _fileMode )
def writeBionetworkStateVarNamesToFile( _currentID, _bionetworkSBML, _outputFileName, _fileMode ):
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.writeBionetworkStateVarNamesToFile( _currentID, _bionetworkSBML, _outputFileName, _fileMode )
    else:
        printInitializationWarningMessage()

#   writeBionetworkStateToFile( _mcs, _currentID, _bionetworkSBML, _outputFileName, _fileMode )
def writeBionetworkStateToFile( _mcs, _currentID, _bionetworkSBML, _outputFileName, _fileMode ):
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.writeBionetworkStateToFile( _mcs, _currentID, _bionetworkSBML, _outputFileName, _fileMode )
    else:
        printInitializationWarningMessage()

#   printBionetworkState( _currentID )
def printBionetworkState( _currentID ):
    global bionetworkManager
    if( bionetworkManager != None ):
        bionetworkManager.printBionetworkState( _currentID )
    else:
        printInitializationWarningMessage()

# no change - integrate with CC3D scripts
def getNeighborContactAreas( _currentCellID ):
    global bionetworkManager
    neighborContactAreas = None
    if( bionetworkManager != None ):
        neighborContactAreas = bionetworkManager.getNeighborContactAreas( _currentCellID )
    else:
        printInitializationWarningMessage()
    return neighborContactAreas

# no change - integrate with CC3D scripts
def getFractionalNeighborContactAreas( _currentCellID ):
    global bionetworkManager
    fracNeighborContactAreas = None
    if( bionetworkManager != None ):
        fracNeighborContactAreas = bionetworkManager.getFractionalNeighborContactAreas( _currentCellID )
    else:
        printInitializationWarningMessage()
    return fracNeighborContactAreas

# no change - integrate with CC3D scripts
def getNeighborProperty( _propertyName, _currentCellID ):
    global bionetworkManager
    neighborProperty = None
    if( bionetworkManager != None ):
        neighborProperty = bionetworkManager.getNeighborProperty( _propertyName, _currentCellID )
    else:
        printInitializationWarningMessage()
    return neighborProperty


# ################### CLASS THAT DEFINES BionetworkManager API FUNCTIONS #####################
#     BionetworkManager( object )
class BionetworkManager( object ):
    def __init__( self, simulator ):
        self.simulator = simulator
        self.inventory = self.simulator.getPotts().getCellInventory()
        self.neighborTracker=None
        if self.simulator.pluginManager.isLoaded("NeighborTracker"):
        ##if 1:
            import CompuCell            
            self.neighborTracker=CompuCell.getNeighborTrackerPlugin()
        if( self.neighborTracker == None ):
            print "**** WARNING: getNeighborTrackerPlugin() returned Null. ****"
        
        # self.neighborTracker = CompuCell.getNeighborTrackerPlugin()
        # if( self.neighborTracker == None ):
            # print "**** WARNING: getNeighborTrackerPlugin() returned Null. ****"
        self.cells = CellList(self.inventory)
        
        ########### GET RID OF ALL OF THIS ###########
        #self.settableCC3DProperties = [
        #    "TargetVolume",
        #    "LambdaVolume",
        #    "TargetSurface",
        #    "LambdaSurface",
        #    "CellType"]
        #
        #self.gettableCC3DProperties = list(self.settableCC3DProperties)
        #self.gettableCC3DProperties.extend(
        #    [ "Volume",
        #      "SurfaceArea" ] )
        ##############################################
        
        self.cellTypeMap = {}
        
        self.nonCellBionetworkInventory = {}
        self.bionetworkTemplateLibraryInventory = {}
        self.bionetworkSBMLInventory = {}
        
        self.utilities = BionetworkUtilManager()
    
    #   attachBionetworkToCell( self, Bionetwork, CC3DCell )
    def attachBionetworkToCell( self, bionetwork, CC3DCell ):
        dictionaryAttrib = CompuCell.getPyAttrib( CC3DCell )
        dictionaryAttrib["Bionetwork"] = bionetwork
    
    #   getBionetworkFromCell( self, CC3DCell )
    def getBionetworkFromCell( self, CC3DCell ):
        bionetwork = None
        if( not (CC3DCell == None) ):
            dictionaryAttrib = CompuCell.getPyAttrib( CC3DCell )
            if( "Bionetwork" in dictionaryAttrib.keys() ):
                bionetwork = dictionaryAttrib["Bionetwork"]
        else:
            print "** NOTIFICATION: Null CC3D Cell **"
            print "-- Will not attempt to retrieve dictionary attribute 'Bionetwork'"
            print "-- Returning Null pointer for bionetwork"
        return bionetwork
    
    # Maciek will send replacement code 
    def getCC3DCellByID(self, cellID):
        #### PUT THE FOLLOWING BACK IN AFTER FIGURING OUT WHY IT'S NOT WORKING ####
        #return self.inventory.getCellById(cellID) # this will work only with 3.5.0
        
        ########### TAKE FOLLOWING OUT ONCE THE ABOVE CODE IS WORKING #############
        CC3DCell = None
        for cell in self.cells:
            if(cell.id == cellID):
                CC3DCell = cell
                break
        if( CC3DCell == None ):
            print "** WARNING: Returning null CC3D cell **"
            print "-- getCC3DCellByID was called, but no cell was found."
        
        return CC3DCell
        ###########################################################################
    
    #   getBionetworkWarehouseNames( self )
    #OR getBionetworkTemplateLibraryNames( self )
    def getBionetworkTemplateLibraryNames( self ):
        return self.bionetworkTemplateLibraryInventory.keys()
    
    # no change - how about getCellIds ?
    def getCellIDs(self):
        cellIDs = []
        for cell in self.cells:
            cellIDs.append(cell.id)
        return cellIDs
    
    #   getBionetworks( self )
    def getBionetworks( self ):
        bionetworkList = []
        for cell in self.cells:  
            bionetworkList.append( self.getBionetworkFromCell(cell) )
        return bionetworkList
    
    #   getBionetworkByID( self, cellID ) - how about getBionetworkByCellID?    
    def getBionetworkByCellID(self, cellID):
        bionetwork = None
        if( cellID in self.getNonCellBionetworkNames() ):
            bionetwork = self.nonCellBionetworkInventory[cellID]
        else:
            for cell in self.cells:
                if(cell.id == cellID):
                    bionetwork = self.getBionetworkFromCell(cell)
                    break
        return bionetwork
    
    #   getNonCellBionetworkNames( self )
    def getNonCellBionetworkNames( self ):
        return self.nonCellBionetworkInventory.keys()
    
    #   getNonCellBionetworks( self )
    def getNonCellBionetworks( self ):
        return self.nonCellBionetworkInventory.values()
    
    # getBionetworksWithIDsByTypeName( self, warehouseName ) -  shouldn't all Bionetworks have ID?
    #OR getBionetworksWithIDsByTemplateName( self, templateLibraryName )
    def getBionetworksWithIDsByTemplateName( self, templateLibraryName ):
        bionetworkInventory = {}
        if( templateLibraryName in self.getNonCellBionetworkNames() ):
            bionetworkInventory[templateLibraryName] = self.nonCellBionetworkInventory[templateLibraryName]
        else:
            for cellID in self.getCellIDs():
                currentBionetwork = self.getBionetworkByCellID(cellID)
                if( currentBionetwork.getTemplateLibraryName() == templateLibraryName ):
                    bionetworkInventory[cellID] = currentBionetwork
        return bionetworkInventory
    
    #   getBionetworkSBMLNames( self, cellID )
    def getBionetworkSBMLNames( self, cellID ):
        modelNames = None
        if( cellID in self.getCellIDs() ):
            modelNames = self.getBionetworkByCellID(cellID).getSBMLModelNames()
        elif( cellID in self.getNonCellBionetworkNames() ):
            modelNames = self.nonCellBionetworkInventory[cellID].getSBMLModelNames()
        else:
            print "cellID %s was not found." % cellID
        return modelNames
    
    # checkIfBionetworkTypeNameInUse( self, warehouseName )
    #OR checkIfBionetworkTemplateLibraryNameInUse( self, templateLibraryName )
    def checkIfBionetworkTemplateLibraryNameInUse( self, templateLibraryName ):
        alreadyInUse = False
        if( templateLibraryName in self.bionetworkTemplateLibraryInventory.keys() ):
            alreadyInUse = True
        return alreadyInUse
    
    # getSBMLModelKeysAsList( self )  (or perhaps no change)
    #OR getBionetworkSBMLKeysAsList( self )
    def getBionetworkSBMLKeysAsList( self ):
        sbmlKeys = []
        for sbmlModel in self.bionetworkSBMLInventory.values():
            currentModelKey = sbmlModel.getModelKey()
            if( not (currentModelKey == "") ):
                sbmlKeys.append( currentModelKey )
        return sbmlKeys
    
    # no change - how about generateCellTypeMap
    def generateCellTypeMap( self ):
        cellTypeMap = CompuCellSetup.ExtractTypeNamesAndIds()
        del(cellTypeMap[0])
        
        for cellTypeID in cellTypeMap.keys():
            print "\nCell type ID: %s\tCell type name: %s\n" % (cellTypeID, cellTypeMap[cellTypeID])
        return cellTypeMap
    
    # no change - how about getCellTypeIndexFromName
    def getCellTypeIndexFromName( self, cellTypeName ):
        cellTypeIndex = None
        if( len(self.cellTypeMap) > 0 ):
            for index in self.cellTypeMap.keys():
                if( self.cellTypeMap[index] == cellTypeName ):
                    cellTypeIndex = index
        return cellTypeIndex
    
    #   loadSBMLModel( self, sbmlModelName, sbmlModelPath, modelKey = "", timeStepOfIntegration = -1.0 )
    def loadSBMLModel( self, sbmlModelName, sbmlModelPath, modelKey = "", timeStepOfIntegration = -1.0 ):
        newSBMLModel = BionetworkSBML( sbmlModelName, sbmlModelPath, timeStepOfIntegration )
        
        if( newSBMLModel.getOdeModel().getOdeModel() == None ):
            print "\n**** ERROR creating a new SBML model ****"
            print "--- SBML file not found or path to SBML file may be incorrect"
            print "--- Path provided: %s" % sbmlModelPath
            print "--- Will *not* create new SBML model object.\n"
            print "\nExiting 'loadSBMLModel' without creating new SBML model object.\n"
            sys.stdout.flush()
        else:
            self.bionetworkSBMLInventory[sbmlModelName] = newSBMLModel
            if( not (modelKey == "") ):
                self.bionetworkSBMLInventory[sbmlModelName].setModelKey( modelKey )
    
    #   createBionetworkTemplateLibrary( self, templateLibraryName )
    def createBionetworkTemplateLibrary( self, templateLibraryName ):
        templateLibraryNameAlreadyUsed = self.checkIfBionetworkTemplateLibraryNameInUse(templateLibraryName)
        if( templateLibraryNameAlreadyUsed ):
            print "\n**** WARNING: Bionetwork template library name '%s' already in use" % templateLibraryName
            print "--- Will not create new bionetwork template library by this name.\n"
            sys.stdout.flush()
        else:
            self.bionetworkTemplateLibraryInventory[templateLibraryName] = BionetworkTemplateLibrary( templateLibraryName )
    
    # addSBMLModelToWarehouse( self, sbmlModelName, warehouseName )
    #OR addSBMLModelToTemplateLibrary( self, sbmlModelName, templateLibraryName )
    def addSBMLModelToTemplateLibrary( self, sbmlModelName, templateLibraryName ):
        validSBML = ( sbmlModelName in self.bionetworkSBMLInventory.keys() )
        if( not validSBML ):
            print "\n**** ERROR: Invalid SBML model name provided. ****"
            print "--- %s is not a valid SBML model.\n" % sbmlModelName
            sys.stdout.flush()
        
        if( len(self.cellTypeMap) == 0 ):
            self.cellTypeMap = self.generateCellTypeMap()
        
        cellTypeInTemplateLibraryInventory = ( templateLibraryName in self.bionetworkTemplateLibraryInventory.keys() )
        if( not cellTypeInTemplateLibraryInventory ):
            print "\n**** NOTIFICATION: Creation of new template library ****"
            print "--- Template library '%s' does not exist yet. Creating new template library...\n" % templateLibraryName
            sys.stdout.flush()
            self.createBionetworkTemplateLibrary( templateLibraryName )
        validCellType = ( templateLibraryName in self.bionetworkTemplateLibraryInventory.keys() )
        
        if( validSBML & validCellType ):
            self.bionetworkTemplateLibraryInventory[templateLibraryName].addSBMLModel(
                self.bionetworkSBMLInventory[sbmlModelName] )
    
    #   setBionetworkInitialCondition( self, templateName, propertyName, propertyValue )
    def setBionetworkInitialCondition( self, templateName, propertyName, propertyValue ):
        #listOfCellTypes = []
        #if( type(templateName) == list ):
        #    listOfCellTypes.extend(templateName)
        #else:
        #    listOfCellTypes.append(templateName)
        listOfTemplateLibraryNames = []
        if( type(templateName) == list ):
            listOfTemplateLibraryNames.extend(templateName)
        else:
            listOfTemplateLibraryNames.append(templateName)
        
        if( len(self.cellTypeMap) == 0 ):
            self.cellTypeMap = self.generateCellTypeMap()
        
        #for typeName in listOfCellTypes:
        for currentTemplateName in listOfTemplateLibraryNames:
            #validNamespace = False
            #if( self.utilities.charFoundInString('_', propertyName) ):
            #    splitString = self.utilities.splitStringAtFirst('_', propertyName)
            #    print "Namespace '%s' was found." % splitString[0]
            #    print "Variable name: %s.\n" % splitString[1]
            #    sys.stdout.flush()
            #    
            #    if( splitString[0] in self.getBionetworkSBMLKeysAsList() ):
            #        validNamespace = True
            #    elif( splitString[0] in self.bionetworkSBMLInventory.keys() ):
            #        validNamespace = True
                
            #if( typeName in self.cellTypeMap.values() ):
            #    if( not (typeName in self.bionetworkTemplateLibraryInventory.keys() ) ):
            #        self.createBionetworkTemplateLibrary(typeName)
            if( currentTemplateName in self.cellTypeMap.values() ):
                if( not (currentTemplateName in self.bionetworkTemplateLibraryInventory.keys() ) ):
                    self.createBionetworkTemplateLibrary(currentTemplateName)
            
            #validCellType = ( typeName in self.bionetworkTemplateLibraryInventory.keys() )
            validTemplateName = ( currentTemplateName in self.bionetworkTemplateLibraryInventory.keys() )
            #if( validCellType ):
            if( validTemplateName ):
                #numSBMLModels = len(self.bionetworkTemplateLibraryInventory[typeName].getModelNames())
                numSBMLModels = len(self.bionetworkTemplateLibraryInventory[currentTemplateName].getModelNames())
                if( numSBMLModels==0 ):
                    print "**** WARNING: SBML initial condition specified for a cell type with no SBML models ****"
                    print "--- Cell type %s has no SBML models associated with it." % currentTemplateName
                    print "--- SBML initial condition '%s = %s' will be ignored unless SBML model is specified." \
                        % (propertyName, propertyValue)
                    sys.stdout.flush()
                #self.bionetworkTemplateLibraryInventory[typeName].setInitialCondition( propertyName, propertyValue )
                self.bionetworkTemplateLibraryInventory[currentTemplateName].setInitialCondition( propertyName, propertyValue )
                #if( validNamespace ):
                #    numSBMLModels = len(self.bionetworkTemplateLibraryInventory[typeName].getModelNames())
                #    if( numSBMLModels==0 ):
                #        print "**** WARNING: SBML initial condition specified for a cell type with no SBML models ****"
                #        print "--- Cell type %s has no SBML models associated with it." % typeName
                #        print "--- SBML initial condition '%s = %s' will be ignored unless SBML model is specified." \
                #            % (propertyName, propertyValue)
                #        sys.stdout.flush()
                #    self.bionetworkTemplateLibraryInventory[typeName].setInitialCondition( propertyName, propertyValue )
                #else:
                #    self.bionetworkTemplateLibraryInventory[typeName].setInitialCondition( propertyName, propertyValue )
            #elif( typeName == "Global" ):
            elif( currentTemplateName == "Global" ):
                #for cType in self.bionetworkTemplateLibraryInventory.values():
                #    cType.setInitialCondition( propertyName, propertyValue )
                for template in self.bionetworkTemplateLibraryInventory.values():
                    template.setInitialCondition( propertyName, propertyValue )
                #if( validNamespace ):
                #    for cType in self.bionetworkTemplateLibraryInventory.values():
                #        cType.setInitialCondition( propertyName, propertyValue )
                #else:
                #    for cType in self.bionetworkTemplateLibraryInventory.values():
                #        #cType.setInitialCondition( "SBML", propertyName, propertyValue )
                #        #cType.setInitialCondition( "CC3D", propertyName, propertyValue )
                #        cType.setInitialCondition( propertyName, propertyValue )
            else:
                print "\n**** ERROR: Invalid template library name provided. ****"
                print "--- %s is not a valid template library. Will not set %s initial condition.\n" \
                    % (currentTemplateName, propertyName)
                sys.stdout.flush()
            
            #if( not validNamespace ):
            #    print "\n**** WARNING: No valid namespace identifier specified. ****"
            #    print "--- If property %s appears in more than one SBML model " % (propertyName) \
            #        + "%s will be set in all cases to the same value.\n" % (propertyName)
            #    sys.stdout.flush()
    
    #   initializeBionetworks( self )
    def initializeBionetworks( self ):
        self.createAndInitializeAllBionetworks()
        
    # this fcn should not exist here at all
    #def setCC3DCellProperty(self, propertyName, propertyValue, CC3DCell):
    #    if( CC3DCell == None ):
    #        print "** NOTIFICATION: Null CC3D Cell **"
    #        print "-- Will not attempt to set CC3D cell property"
    #    else:
    #        if( propertyName == "TargetVolume" ):
    #            CC3DCell.targetVolume = propertyValue
    #        elif( propertyName == "LambdaVolume" ):
    #            CC3DCell.lambdaVolume = propertyValue
    #        elif( propertyName == "TargetSurface" ):
    #            CC3DCell.targetSurface = propertyValue
    #        elif( propertyName == "LambdaSurface" ):
    #            CC3DCell.lambdaSurface = propertyValue
    #        elif( propertyName == "CellType" ):
    #            CC3DCell.type = propertyValue
                
    # this fcn should not exist here at all
    #def getCC3DCellProperty(self, propertyName, CC3DCell):
    #    propertyValue = None
    #    if( CC3DCell == None ):
    #        print "** NOTIFICATION: Null CC3D Cell **"
    #        print "-- Will not attempt to get CC3D cell property"
    #        print "-- Returning Null property value"
    #    else:
    #        if( propertyName == "TargetVolume" ):
    #            propertyValue = CC3DCell.targetVolume
    #        elif( propertyName == "LambdaVolume" ):
    #            propertyValue = CC3DCell.lambdaVolume
    #        elif( propertyName == "TargetSurface" ):
    #            propertyValue = CC3DCell.targetSurface
    #        elif( propertyName == "LambdaSurface" ):
    #            propertyValue = CC3DCell.lambdaSurface
    #        elif( propertyName == "CellType" ):
    #            propertyValue = CC3DCell.type
    #        elif( propertyName == "Volume" ):
    #            propertyValue = CC3DCell.volume
    #        elif( propertyName == "SurfaceArea" ):
    #            propertyValue = CC3DCell.surface
    #    return propertyValue
    
    # this fcn should not exist here at all
    #what is the purpose of this one?
    #def initializeCellProperties(self, CC3DCell, SOSCell):
    #    cellPropertyInitialConditions = \
    #        SOSCell.getCellTypeInstancePtr().getInitialConditions("CC3D")
    #    
    #    for propertyName in cellPropertyInitialConditions.keys():
    #        propertyValue = cellPropertyInitialConditions[propertyName]
    #        if( propertyName == "CellType" ):
    #            self.setPropertyValueForSpecifiedCellID( propertyName, propertyValue, CC3DCell.id)
    #        elif( propertyName in self.settableCC3DProperties ):
    #            self.setCC3DCellProperty(propertyName, propertyValue, CC3DCell)
    #        elif( propertyName == "DivideVolume" ):
    #            SOSCell.setDivideVolume(propertyValue)
    
    #   createAndInitializeAllBionetworks( self ):
    def createAndInitializeAllBionetworks( self ):
        if( len(self.cellTypeMap) == 0 ):
            self.cellTypeMap = self.generateCellTypeMap()
            
        for cellType in self.cellTypeMap.values():
            cellTypeInTemplateLibraryList = (cellType in self.bionetworkTemplateLibraryInventory.keys())
            if( not (cellTypeInTemplateLibraryList) ):
                print "\n**** NOTIFICATION: Creation of new bionetwork template library ****"
                print "--- Bionetwork template library '%s' has not been created. Will create it now...\n" % cellType
                self.createBionetworkTemplateLibrary( cellType )
        
        for cell in self.cells:
            currentTypeName = self.simulator.getPotts().getAutomaton().getTypeName(cell.type)
            if( currentTypeName in self.bionetworkTemplateLibraryInventory.keys() ):
                newBionetwork = Bionetwork(
                    currentTypeName, self.bionetworkTemplateLibraryInventory[currentTypeName] )
                newBionetwork.initializeIntegrators()
                #newBionetwork.setInitialVolume(cell.volume)
                
                #self.initializeCellProperties(cell, newBionetwork)
                self.attachBionetworkToCell( newBionetwork, cell )
        
        for templateLibraryName in self.bionetworkTemplateLibraryInventory.keys():
            isNotInCC3DTypeList = not (templateLibraryName in self.cellTypeMap.values())
            isMedium = (templateLibraryName == "Medium")
            if( isNotInCC3DTypeList | isMedium ):
                print "\n**** NOTIFICATION: Creation of bionetwork without a corresponding CC3D cell ****"
                sys.stdout.flush()
                self.nonCellBionetworkInventory[templateLibraryName] = Bionetwork(
                    templateLibraryName, self.bionetworkTemplateLibraryInventory[templateLibraryName] )
                print "--- Bionetwork '%s' has been created.\n" % templateLibraryName
                sys.stdout.flush()
                self.nonCellBionetworkInventory[templateLibraryName].initializeIntegrators()
    
    #   timestepBionetworks( self, globalTimeStepOfIntegration = -1.0 )
    def timestepBionetworks( self, globalTimeStepOfIntegration = -1.0 ):
        self.updateAllBionetworkStates( globalTimeStepOfIntegration )
    
    #   updateAllBionetworkStates( self, globalTimeStepOfIntegration = -1.0 )
    def updateAllBionetworkStates( self, globalTimeStepOfIntegration = -1.0 ):
        if( globalTimeStepOfIntegration > 0.0 ):
            for bionet in self.getBionetworks():
                #bionet.updateIntracellularStateWithTimeStep( globalTimeStepOfIntegration )
                bionet.updateBionetworkStateWithTimeStep( globalTimeStepOfIntegration )
            for nonCellBionet in self.getNonCellBionetworks():
                #nonCellBionet.updateIntracellularStateWithTimeStep( globalTimeStepOfIntegration )
                nonCellBionet.updateBionetworkStateWithTimeStep( globalTimeStepOfIntegration )
        else:
            for bionet in self.getBionetworks():
                #bionet.updateIntracellularState()
                bionet.updateBionetworkState()
            for nonCellBionet in self.getNonCellBionetworks():
                #nonCellBionet.updateIntracellularState()
                nonCellBionet.updateBionetworkState()
    
    #   writeBionetworkStateVarNamesToFile( self, currentID, bionetworkSBML, outputFileName, fileMode )
    def writeBionetworkStateVarNamesToFile( self, currentID, bionetworkSBML, outputFileName, fileMode ):
        output = open( outputFileName, fileMode )
        if( currentID in self.getCellIDs() ):
            output.write( "\t%s" % self.getBionetworkByCellID(currentID).
                #getIntracellStateVarNamesAsString( bionetworkSBML ) )
                getBionetworkStateVarNamesAsString( bionetworkSBML ) )
        elif( currentID in self.getNonCellBionetworkNames() ):
            output.write( "\t%s" % self.nonCellBionetworkInventory[ currentID ].
                #getIntracellStateVarNamesAsString( bionetworkSBML ) )
                getBionetworkStateVarNamesAsString( bionetworkSBML ) )
        output.close()
    
    #   writeBionetworkStateToFile( self, mcs, currentID, bionetworkSBML, outputFileName, fileMode )
    def writeBionetworkStateToFile( self, mcs, currentID, bionetworkSBML, outputFileName, fileMode ):
        output = open( outputFileName, fileMode )
        if( currentID in self.getCellIDs() ):
            output.write( "%s\t%s" % (mcs, self.getBionetworkByCellID(currentID).
                #getIntracellStateAsString( bionetworkSBML ) ) )
                getBionetworkStateAsString( bionetworkSBML ) ) )
        elif( currentID in self.getNonCellBionetworkNames() ):
            output.write( "%s\t%s" % (mcs, self.nonCellBionetworkInventory[ currentID ].
                #getIntracellStateAsString( bionetworkSBML ) ) )
                getBionetworkStateAsString( bionetworkSBML ) ) )
        output.close()
    
    #   printBionetworkState( self, currentID )
    def printBionetworkState( self, currentID ):
        if( currentID in self.getCellIDs() ):
            #self.getBionetworkByCellID(currentID).printIntracellularState( True )
            self.getBionetworkByCellID(currentID).printBionetworkState( True )
        elif( currentID in self.getNonCellBionetworkNames() ):
            #self.nonCellBionetworkInventory[ currentID ].printIntracellularState( True )
            self.nonCellBionetworkInventory[ currentID ].printBionetworkState( True )
    
    #   getBionetworkValue( self, propertyName, currentID = "Global")
    def getBionetworkValue( self, propertyName, currentID = "Global"):
        returnValue = None
        propertyValue = self.findPropertyValue( propertyName, currentID )
        if( type(propertyValue) == dict ):
            returnValue = propertyValue
        else:
            returnValue = propertyValue[1]
        return returnValue
        
    #   findBionetworkPropertyValue( self, propertyName, cellId)
    def findBionetworkPropertyValue( self, propertyName, cellID):
        splitString = self.utilities.splitStringAtFirst( '_', propertyName )
        propertyValue = [False, 0.0]
        #if( propertyName == "CellType" ):
        if( propertyName == "TemplateLibrary" ):
            propertyValue[0] = True
            #propertyValue[1] = self.getBionetworkByCellID(cellID).getCellTypeName()
            propertyValue[1] = self.getBionetworkByCellID(cellID).getTemplateLibraryName()
        #elif( propertyName == "DivideVolume" ):
        #    propertyValue[0] = True
        #    propertyValue[1] = self.getBionetworkByCellID(cellID).getDivideVolume()
        #elif( propertyName == "InitialVolume" ):
        #    propertyValue[0] = True
        #    propertyValue[1] = self.getBionetworkByCellID(cellID).getInitialVolume()
        #elif( propertyName in self.gettableCC3DProperties ):
        #    propertyValue[0] = True
        #    propertyValue[1] = self.getCC3DCellProperty( propertyName, self.getCC3DCellByID(cellID) )
        #elif( splitString[1] in self.gettableCC3DProperties ):
        #    if( splitString[0] == "GGH" ):
        #        propertyValue[0] = True
        #        propertyValue[1] = self.getCC3DCellProperty( splitString[1], self.getCC3DCellByID(cellID) )
        else:
            propertyValue = self.getBionetworkByCellID(cellID).findPropertyValue( propertyName )
        return propertyValue
    
    def findPropertyValue( self, propertyName, cellID = "Global" ):
        propertyValue = None
        if( cellID == "Global" ):
            propertyValue = {}
            for currentCellID in self.getCellIDs():
                propertyValue[currentCellID] = \
                    self.findBionetworkPropertyValue(propertyName, currentCellID)[1]
        elif( cellID in self.getCellIDs() ):
            propertyValue = self.findBionetworkPropertyValue(propertyName, cellID)
        elif( cellID in self.getBionetworkTemplateLibraryNames() ):
            if( cellID in self.getNonCellBionetworkNames() ):
                propertyValue = self.nonCellBionetworkInventory[ cellID ].findPropertyValue( propertyName )
            else:
                propertyValue = {}
                cellInventory = self.getBionetworksWithIDsByTemplateName( cellID )
                for currentCellID in cellInventory.keys():
                    propertyValue[currentCellID] = \
                        self.findBionetworkPropertyValue(propertyName, currentCellID)[1]
        else:
            propertyValue = (False, 0.0)
        return propertyValue
    
    #   setBionetworkValue( self, propertyName, newPropertyValue, currentID = "Global" )
    def setBionetworkValue( self, propertyName, newPropertyValue, currentID = "Global" ):
        self.setPropertyValue( propertyName, newPropertyValue, currentID )
    
    def setPropertyValueForSpecifiedCellID( self, propertyName, newPropertyValue, currentCellID):
        splitString = self.utilities.splitStringAtFirst( '_', propertyName )
        #if( propertyName == "CellType" ):
        if( propertyName == "TemplateLibrary" ):
            if( newPropertyValue in self.bionetworkTemplateLibraryInventory.keys() ):
                if( currentCellID in self.getCellIDs() ):
                    #newCellType = self.bionetworkTemplateLibraryInventory[newPropertyValue]
                    newTemplateLibrary = self.bionetworkTemplateLibraryInventory[newPropertyValue]
                    #self.getBionetworkByCellID(currentCellID).changeCellType(newCellType)
                    self.getBionetworkByCellID(currentCellID).changeTemplateLibrary(newTemplateLibrary)
                    self.getCC3DCellByID(currentCellID).type = self.getCellTypeIndexFromName(newPropertyValue)
        #elif( propertyName == "DivideVolume" ):
        #    self.getBionetworkByCellID(currentCellID).setDivideVolume( newPropertyValue )
        #elif( propertyName == "InitialVolume" ):
        #    self.getBionetworkByCellID(currentCellID).setInitialVolume( newPropertyValue )
        #elif( propertyName in self.settableCC3DProperties ):
        #    self.setCC3DCellProperty( propertyName, newPropertyValue, self.getCC3DCellByID(currentCellID) )
        #elif( splitString[1] in self.settableCC3DProperties ):
        #    if( splitString[0] == "GGH" ):
        #        self.setCC3DCellProperty( splitString[1], newPropertyValue, self.getCC3DCellByID(currentCellID) )
        else:
            self.getBionetworkByCellID(currentCellID).setPropertyValue( propertyName, newPropertyValue )
    
    def setPropertyValue( self, propertyName, newPropertyValue, cellID = "Global" ):
        if( cellID == "Global" ):
            for currentCellID in self.getCellIDs():
                self.setPropertyValueForSpecifiedCellID( propertyName, newPropertyValue, currentCellID )
        elif( cellID in self.getCellIDs() ):
            self.setPropertyValueForSpecifiedCellID( propertyName, newPropertyValue, cellID )
        elif( cellID in self.getBionetworkTemplateLibraryNames() ):
            if( cellID in self.getNonCellBionetworkNames() ):
                self.nonCellBionetworkInventory[ cellID ].setPropertyValue( propertyName, newPropertyValue )
            else:
                cellInventory = self.getBionetworksWithIDsByTemplateName( cellID )
                for currentCellID in cellInventory.keys():
                    self.setPropertyValueForSpecifiedCellID( propertyName, newPropertyValue, currentCellID )
    
    # Include this function to get contact areas of neighbors of a specified cell
    def getNeighborContactAreas(self, currentCellID):
        contactAreas = {}
        if( currentCellID in self.getCellIDs() ):
            if( not (self.neighborTracker == None) ):
                for neighbor in CellNeighborListAuto(self.neighborTracker, self.getCC3DCellByID(currentCellID) ):
                    if neighbor.neighborAddress:
                        nID = neighbor.neighborAddress.id
                        contactArea = neighbor.commonSurfaceArea
                        contactAreas[nID] = contactArea
            else:
                print "**** WARNING: Null neighbor tracker ****"
                print "--- Cannot retrieve neighbor cell data"
        else:
            print "**** WARNING: Cell ID %s does not exist in cell list. ****" % currentCellID
            print "--- Cannot retrieve neighbor data for invalid Cell ID."
        return contactAreas
    
    def getFractionalNeighborContactAreas(self, currentCellID):
        fractionalContactAreas = {}
        currentCell = self.getCC3DCellByID( currentCellID )
        contactAreas = self.getNeighborContactAreas( currentCellID )
        #currentCellSurfaceArea = self.getBionetworkValue("SurfaceArea", currentCellID)
        for neighborID in contactAreas.keys():
            fractionalContactAreas[neighborID] = contactAreas[neighborID]/currentCell.surface
        return fractionalContactAreas
    
    # Include this function to get any property of neighbors of a specified cell
    def getNeighborProperty( self, propertyName, currentCellID ):
        neighborProperties = {}
        if( currentCellID in self.getCellIDs() ):
            if( not (self.neighborTracker == None) ):
                for neighbor in CellNeighborListAuto( self.neighborTracker, self.getCC3DCellByID(currentCellID) ):
                    if neighbor.neighborAddress:
                        nID = neighbor.neighborAddress.id
                        propertyValue = self.getBionetworkValue( propertyName, nID )
                        neighborProperties[nID] = propertyValue
            else:
                print "**** WARNING: Null neighbor tracker ****"
                print "--- Cannot retrieve neighbor cell data"
        else:
            print "**** WARNING: Cell ID %s does not exist in cell list. ****" % currentCellID
            print "--- Cannot retrieve neighbor data for invalid Cell ID."
        return neighborProperties
    
    #   copyBionetworkFromParent( self, parentCC3DCell, childCC3DCell ) - here I would reverse the order of arguments copyBionetworkFromParent( _CC3DParentCell, _CC3DChildCell ) -  this is more standard
    def copyBionetworkFromParent( self, parentCC3DCell, childCC3DCell ):
        parentBionetwork = self.getBionetworkByCellID( parentCC3DCell.id )
        #newSOSCell = parentSOSCell.cloneCell() # UNLESS you will explicitely delete pointer allocated in C++ NEVER pass "freely floating ptr" from C++ to Python and think that it will be garbage collected. It wiill not 
        
        newBionetwork = Bionetwork( parentBionetwork )
        self.attachBionetworkToCell( newBionetwork, childCC3DCell )








