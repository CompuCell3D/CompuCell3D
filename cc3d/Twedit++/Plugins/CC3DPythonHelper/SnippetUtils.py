class SnippetUtils:
    def __init__(self):
        self.snippetDict={}
        
        self.initCodeSnippets()
        
    def getCodeSnippetsDict(self):
        return self.snippetDict
        
    def initCodeSnippets(self):
        
        self.snippetDict["Python Utilities Get Dir Of Current File"]="""
fileDir=os.path.dirname (os.path.abspath( __file__ ))
"""
        self.snippetDict["Python Utilities Get FullPath Of Current File"]="""
filePath=os.path.abspath( __file__ )
"""

        self.snippetDict["Python Utilities Import os"]="""
import os
"""


        self.snippetDict["Bionet Solver 1. Import bionetAPI (top level)"]="""
import bionetAPI
"""

        self.snippetDict["Bionet Solver 3. Load SBML Model (start fcn)"]="""
modelName = "MODEL_NAME"
modelNickname  = "NICKNAME" # this is usually shorter version version of model name
modelPath="PATH_TO_SBML_FILE"
integrationStep = 0.2
bionetAPI.loadSBMLModel(modelName, modelPath,modelNickname,  integrationStep)
"""

        self.snippetDict["Bionet Solver 4. Assign Model To Cell Type (start fcn)"]="""
bionetAPI.addSBMLModelToTemplateLibrary("MODEL_NAME","CELL_TYPE_NAME")
"""
        
        self.snippetDict["Bionet Solver 5. Initialize BionetSolver (start fcn)"]="""
bionetAPI.initializeBionetworks()
"""

        self.snippetDict["Bionet Solver 2. Initialize Bionetwork Manager (__init__)"]="""
bionetAPI.initializeBionetworkManager(self.simulator) # this function has to be called from __init__ function of the steppables
"""


        self.snippetDict["Bionet Solver Get Species Concentration (start or step fcn)"]="""
concentration=bionetAPI.getBionetworkValue("NICKNAME_SPECIESNAME",cell.id)
"""

        self.snippetDict["Bionet Solver Set Species Concentration (start or step fcn)"]="""
bionetAPI.setBionetworkValue("NICKNAME_SPECIESNAME",CONCENTRATION,cell.id)
"""
        self.snippetDict["Bionet Solver Time Step Bionetwork (integrate) (step fcn)"]="""
bionetAPI.timestepBionetworks() 
"""


        self.snippetDict["Cell ConstraintsChange Target Volume"]="""
cell.targetVolume=25
"""
        self.snippetDict["Cell Constraints Change Lambda Volume"]=""" 
cell.lambdaVolume=2.0
"""

        self.snippetDict["Cell Constraints Change Target Surface"]="""
cell.targetSurface=20.0
"""
        self.snippetDict["Cell Constraints Change Lambda Surface"]="""
cell.lambdaSurface=2.0
"""

        self.snippetDict["Cell Constraints Apply Force To Cell"]=""" 
# Make sure ExternalPotential plugin is loaded
cell.lambdaVecX=-0.5 # force component pointing along X axis - towards positive X's
cell.lambdaVecY=0.5 # force component pointing along Y axis - towards negative Y's
cell.lambdaVecY=0.0 # force component pointing along Z axis 
"""

        self.snippetDict["Cell Manipulation Delete Cell"]="""  
self.deleteCell(cell)  
"""
        self.snippetDict["Cell Manipulation Create Cell"]=""" 
pt=CompuCell.Point3D(50,50,0)
self.createNewCell(2,pt,5,5,1) # the arguments are (type,pt,xSize,ySize,zSize)
"""

        self.snippetDict["Cell Manipulation Move Cell"]=""" 
# Shifting entire cell by a 'shiftVector'         
shiftVector=CompuCell.Point3D(20,20,0)
self.moveCell(cell,shiftVector)
"""

        self.snippetDict["Visit All Cells"]="""
# iterating over all cells in simulation        
for cell in self.cellList:
    # you can access/manipulate cell properties here
    print "id=",cell.id, " type=",cell.type," volume=",cell.volume
"""        
        
        self.snippetDict["Visit All Cells Of Given Type"]="""
# iterating over cells of type 1        
for cell in self.cellListByType(1):
    # you can access/manipulate cell properties here
    print "id=",cell.id," type=",cell.type
"""        
    
        self.snippetDict["Visit Cell Neighbors"]="""
# Make sure NeighborTracker Plugin is loaded        
cellNeighborList=self.getCellNeighbors(cell) # generates list of neighbors of cell 'cell'
for neighborSurfaceData in cellNeighborList:
    # neighborSurfaceData.neighborAddress is a pointer to one of 'cell' neighbors stired in cellNeighborList
    #IMPORTANT: cell may have Medium (NULL pointer) as a neighbor. therefore before accessing neighbor we first check if it is no Medium
    if neighborSurfaceData.neighborAddress: 
        print "neighbor.id",neighborSurfaceData.neighborAddress.id," commonSurfaceArea=",neighborSurfaceData.commonSurfaceArea
        
    # for Medium we  cannot access ID, type etc - it is null pointer we can only access common surface area with 'cell'  
    else:                    
        print "Medium commonSurfaceArea=",neighborSurfaceData.commonSurfaceArea
"""        
        self.snippetDict["Chemical Field Manipulation Get Field Reference"]="""
field=CompuCell.getConcentrationField(self.simulator,"NAME OF THE FIELD TO OUTPUT")
"""        

        self.snippetDict["Chemical Field Manipulation Get Field Value"]="""
pt=CompuCell.Point3D()
pt.x=10 ; pt.y=10 ; pt.z=0
value=field.get(pt)
"""        
        self.snippetDict["Chemical Field Manipulation Set Field Value"]="""
pt=CompuCell.Point3D()
pt.x=10 ; pt.y=10 ; pt.z=0
value=field.set(pt,1.02)
"""        


        self.snippetDict["Chemical Field Manipulation Write To Disk"]="""
fileName="TYPE YOUR FILE NAME HERE"
field=CompuCell.getConcentrationField(self.simulator,"NAME OF THE FIELD TO OUTPUT")
pt=CompuCell.Point3D()
if field:
    try:                
        import CompuCellSetup
        fileHandle,fullFileName=CompuCellSetup.openFileInSimulationOutputDirectory(fileName,"w")
    except IOError:
        print "Could not open file ", fileName," for writing. Check if you have necessary permissions"
    
    for i in xrange(self.dim.x):
        for j in xrange(self.dim.y):
            for k in xrange(self.dim.z):
                pt.x=i
                pt.y=j
                pt.z=k
                fileHandle.write("%d\\t%d\\t%d\\t%f\\n"%(pt.x,pt.y,pt.z,field.get(pt))) # field.get(pt)  gets concentration at location 'pt'

    fileHandle.close()
"""        

        self.snippetDict["Chemical Field Manipulation Modification (aka secretion)"]="""
fileName="TYPE YOUR FILE NAME HERE"
field=CompuCell.getConcentrationField(self.simulator,"NAME OF THE FIELD TO OUTPUT")
pt=CompuCell.Point3D()
if field:    
    for i in xrange(self.dim.x):
        for j in xrange(self.dim.y):
            for k in xrange(self.dim.z):
                pt.x=i
                pt.y=j
                pt.z=k
                conc=field.get(pt) # getting current value of the field at coordinates 'pt'
                conc*=1.1 # imultiplying concentration by 1.1
                field.set(pt,conc) # assigning new concentration
"""        

        self.snippetDict["Cell Attributes Add Dictionary To Cells"]="""
pyAttributeAdder,dictAdder=CompuCellSetup.attachDictionaryToCells(sim)
"""        

        self.snippetDict["Cell Attributes Add List To Cells"]="""
pyAttributeAdder,listAdder=CompuCellSetup.attachListToCells(sim)
"""        


        self.snippetDict["Cell Attributes Access/Modify Dictionary Attribute"]="""
# access/modification of a dictionary attached to cell - make sure to decalare in main script that you will use such attribute
dict_attrib=CompuCell.getPyAttrib(cell)
dict_attrib["Double_MCS_ID"]=mcs*2*cell.id
print "dict attrib for cell.id=",cell.id, "is ",dict_attrib
"""        

        self.snippetDict["Cell Attributes Access/Modify List Attribute"]="""
# access/modification of a list attached to cell - make sure to decalare in main script that you will use such attribute
list_attrib=CompuCell.getPyAttrib(cell)
list_attrib[0:2]=[mcs,mcs*2*cell.id]
print "list attrib for cell.id=",cell.id, "is ",list_attrib
"""

        self.snippetDict["Cell Attributes Center Of Mass"]="""        
# Make sure CenterOfMass plugin is loaded
# READ ONLY ACCESS
xCOM=cell.xCOM
yCOM=cell.yCOM
zCOM=cell.zCOM
"""

        self.snippetDict["Cell Attributes Volume"]=""" 
# READ ONLY ACCESS        
volume=cell.volume
"""
        self.snippetDict["Cell Attributes Target Volume"]=""" 
# READ/WRITE  ACCESS                
targetVolume=cell.targetVolume
"""
        self.snippetDict["Cell Attributes Lambda Volume"]=""" 
# READ/WRITE  ACCESS        
lambdaVolume=cell.lambdaVolume
"""
        self.snippetDict["Cell Attributes Surface"]=""" 
# READ ONLY ACCESS        
surface=cell.surface
"""

        self.snippetDict["Cell Attributes Target Surface"]=""" 
# READ/WRITE  ACCESS                
targetSurface=cell.targetSurface
"""

        self.snippetDict["Cell Attributes Lambda Surface"]=""" 
# READ/WRITE  ACCESS                
lambdaSurface=cell.lambdaSurface
"""

        self.snippetDict["Cell Attributes Id"]=""" 
# READ ONLY ACCESS        
id=cell.id
"""
        self.snippetDict["Cell Attributes Cluster Id"]=""" 
# READ ONLY ACCESS - can be modified using reassignClusterId function        
clusterId=cell.clusterId
"""
        self.snippetDict["Cell Attributes Cluster Id Reassignment"]="""
# You cannot simply set cluster Id on a cell to make t belong to other cluster you have to use the following function call
reassignIdFlag=self.inventory.reassignClusterId(cell,1536) # changing cluster id to 1536 for cell 'cell'
"""


        # self.snippetDict["Cell Attributes Cell Type Change"]="""
# # simply assign new type number - make sure you have enough types declared in your simulation        
# cell.type=2
# """        


        self.snippetDict["Cell Attributes Cell Type"]=""" 
# READ/WRITE  ACCESS                
type=cell.type
"""

        self.snippetDict["Cell Attributes Fluctuation Ampl "]=""" 
# READ/WRITE  ACCESS                        
fluctAmpl=cell.fluctAmpl
"""
        self.snippetDict["Cell Attributes Fluctuation Ampl Reassignment"]="""
cell.fluctAmpl=50 
#uncomment line below to use globally defined FluctuationAmplitude
#cell.fluctAmpl=-1                                
"""


        self.snippetDict["Cell Attributes Inertia Tensor"]=""" 
# READ ONLY ACCESS        
iXX=cell.iXX
iYY=cell.iYY
iZZ=cell.iZZ
iXY=cell.iXY
iXZ=cell.iXZ
iYZ=cell.iYZ
eccentricity=cell.ecc
"""


        self.snippetDict["Focal Point Placticity Properties"]="""
# Make sure FocalPointPlacticity plugin is loaded
#visiting all focal links cell 'cell' has with other cells
for fppd in self.getFocalPointPlasticityDataList(cell):
    print "fppd.neighborId",fppd.neighborAddress.id, " lambda=",fppd.lambdaDistance, " targetDistance=",fppd.targetDistance
    self.focalPointPlasticityPlugin.setFocalPointPlasticityParameters(cell,fppd.neighborAddress,1.0,7.0,20.0)  # arguments are (cell1,cell2,lambda,targetDistance,maxDistance)     
"""
        self.snippetDict["Focal Point Placticity Properties (Within Cluster)"]="""
# Make sure FocalPointPlacticity plugin is loaded
#visiting all focal links cell 'cell' has with other cells
for fppd in self.getInternalFocalPointPlasticityDataList(cell):
    print "fppd.neighborId",fppd.neighborAddress.id, " lambda=",fppd.lambdaDistance, " targetDistance=",fppd.targetDistance
    self.focalPointPlasticityPlugin.setInternalFocalPointPlasticityParameters(cell,fppd.neighborAddress,1.0,7.0,20.0)  # arguments are (cell1,cell2,lambda,targetDistance,maxDistance)     
"""


        self.snippetDict["Adhesion Flex Get Molecule Dens. By Name"]="""
# Make sure AdhesionFlex plugin is loaded
self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell,"NCad") # accessing adhesion molecule density using its name
"""

        self.snippetDict["Adhesion Flex Get Molecule Dens. By Index"]="""
# Make sure AdhesionFlex plugin is loaded
self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell,0) # accessing adhesion molecule density using its index - molecules are indexed in the sdame order they are listed in the xml file                   
"""
        self.snippetDict["Adhesion Flex Set Molecule Dens. By Name"]="""
# Make sure AdhesionFlex plugin is loaded
self.adhesionFlexPlugin.setAdhesionMoleculeDensity(cell,"NCad",11.2) # setting adhesion molecule density using its name
"""

        self.snippetDict["Adhesion Flex Set Molecule Dens. By Index"]="""
# Make sure AdhesionFlex plugin is loaded
self.adhesionFlexPlugin.setAdhesionMoleculeDensity(cell,0,11.2) # setting adhesion molecule density using its index - molecules are indexed in the sdame order they are listed in the xml file
"""

        self.snippetDict["Adhesion Flex Get Molecule Dens. By Name (Medium)"]="""
# Make sure AdhesionFlex plugin is loaded
self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensity("NCad") # accessing adhesion molecule density for Medium using its name
"""

        self.snippetDict["Adhesion Flex Get Molecule Dens. By Index (Medium)"]="""
# Make sure AdhesionFlex plugin is loaded
self.getMediumAdhesionMoleculeDensityByIndex(0) # accessing adhesion molecule density for Medium using its index - molecules are indexed in the sdame order they are listed in the xml file                   
"""
        self.snippetDict["Adhesion Flex Set Molecule Dens. By Name (Medium)"]="""
# Make sure AdhesionFlex plugin is loaded
self.adhesionFlexPlugin.setMediumAdhesionMoleculeDensity("NCad",11.2) # setting adhesion molecule density for Medium using its name
"""

        self.snippetDict["Adhesion Flex Set Molecule Dens. By Index (Medium)"]="""
# Make sure AdhesionFlex plugin is loaded
self.adhesionFlexPlugin.setMediumAdhesionMoleculeDensityByIndex(0,11.2) # setting adhesion molecule density for Medium using its index - molecules are indexed in the sdame order they are listed in the xml file
"""

        self.snippetDict["Visit Cell Boundary Pixels"]="""
# Make sure BoundaryPixelTracker plugin is loaded
pixelList=self.getCellBoundaryPixelList(cell)
for boundaryPixelTrackerData in pixelList:
    print "pixel of cell id=",cell.id," type:",cell.type, " = ",boundaryPixelTrackerData.pixel," number of pixels=",pixelList.numberOfPixels()
"""

        self.snippetDict["Visit Cell Pixels"]="""
# Make sure PixelTracker plugin is loaded
pixelList=self.getCellPixelList(cell)
for pixelTrackerData in pixelList:
   print "pixel of cell id=",cell.id," type:",cell.type, " = ",pixelTrackerData.pixel," number of pixels=",pixelList.numberOfPixels()
"""

        self.snippetDict["Chemotaxis By Cell Id (Define)"]="""
# Make sure Chemotaxis Plugin is loaded
# defining chemotaxis properties of individual cell 'cell'
cd=self.chemotaxisPlugin.addChemotaxisData(cell,"FIELDNAME")
cd.setLambda(20.0)
# cd.initializeChemotactTowardsVectorTypes("Bacterium,Medium")
cd.assignChemotactTowardsVectorTypes([0,1])
"""

        self.snippetDict["Chemotaxis By Cell Id (Modify)"]="""
# Make sure Chemotaxis Plugin is loaded
# modifying chemotaxis properties of individual cell 'cell'
cd=self.chemotaxisPlugin.getChemotaxisData(cell,"FIELDNAME")
if cd:
    l=cd.getLambda()-3
    cd.setLambda(l)
"""



        self.snippetDict["Cell Constraints Length Constraint By Cell Id"]="""
# Make sure LengthConstraintLocalFlex plugin is loaded
self.lengthConstraintFlexPlugin.setLengthConstraintData(cell,20,20) # cell , lambdaLength, targetLength
"""

        self.snippetDict["Cell Constraints Connectivity Constraint By Cell Id"]="""
# Make sure ConnectivityLocalFlex plugin is loaded
self.connectivityLocalFlexPlugin.setConnectivityStrength(cell,10000000) #cell, strength
"""

        self.snippetDict["Inertia Tensor Information"]="""
# Make sure MomentOfInertia plugin is loaded
print "cell.iXX=",cell.iXX," cell.iYY=",cell.iYY," cell.iXY=",cell.iXY
#simiilarly we can get other components of intertia tensor
"""

        self.snippetDict["Inertia Tensor Semiaxes"]="""
# Make sure MomentOfInertia plugin is loaded
axes=self.momentOfInertiaPlugin.getSemiaxes(cell)            
print "minorAxis=",axes[0]," majorAxis=",axes[2], " medianAxis=",axes[1]
"""

        self.snippetDict["Elasticity Modify Existing Links"]="""
# Make sure Elasticity plugin is loaded and Local option is on
elasticityDataList=self.getElasticityDataList(cell)
for elasticityData in elasticityDataList: # visiting all elastic links of 'cell'

    targetLength=elasticityData.targetLength               
    elasticityData.targetLength=6.0
    elasticityData.lambdaLength=200.0
    elasticityNeighbor=elasticityData.neighborAddress
    
    # now we set up elastic link data stored in neighboring cell
    neighborElasticityData=None
    neighborElasticityDataList=self.getElasticityDataList(elasticityNeighbor)
    for neighborElasticityDataTmp in neighborElasticityDataList:
        if not CompuCell.areCellsDifferent(neighborElasticityDataTmp.neighborAddress,cell):
            neighborElasticityData=neighborElasticityDataTmp
            break
    
    if neighborElasticityData is None:
        print "None Type returned. Problems with FemDataNeighbors initialization or sets of elasticityNeighborData are corrupted"
        sys.exit()
    neighborElasticityData.targetLength=6.0
    neighborElasticityData.lambdaLength=200.0

"""

        self.snippetDict["Elasticity Add New Elastic Link"]="""
# Make sure Elasticity plugin is loaded and Local option is on
self.elasticityTrackerPlugin.addNewElasticLink(cell1 ,cell2, 200.0, 6.0) # arguments are cell1, cell2, lambdaElasticLink,targetLinkLength
"""
        self.snippetDict["Elasticity Remove Elastic Link"]="""
# Make sure Elasticity plugin is loaded and Local option is on
self.elasticityTrackerPlugin.removeElasticityPair(cell1 ,cell2, 200.0, 6.0) # arguments are cell1, cell2,
"""


        self.snippetDict["Secretion Inside Cell"]="""
# Make sure Secretion plugin is loaded
# make sure this field is defined in one of the PDE solvers
secretor=self.getFieldSecretor("FIELDNAME") # you may reuse secretor for many cells. Simply define it outside the loop
secretor.secreteInsideCell(cell,300)
"""

        self.snippetDict["Secretion Inside Cell At Boundary"]="""
# Make sure Secretion plugin is loaded
# make sure this field is defined in one of the PDE solvers
secretor=self.getFieldSecretor("FIELDNAME")  # you may reuse secretor for many cells. Simply define it outside the loop
secretor.secreteInsideCellAtBoundary(cell,300)
"""

        self.snippetDict["Secretion Outside Cell At Boundary"]="""
# Make sure Secretion plugin is loaded
# make sure this field is defined in one of the PDE solvers
secretor=self.getFieldSecretor("FIELDNAME") # you may reuse secretor for many cells. Simply define it outside the loop
secretor.secreteOutsideCellAtBoundary(cell,300)
"""

        self.snippetDict["Secretion Inside Cell At COM"]="""
# Make sure Secretion plugin is loaded
# make sure this field is defined in one of the PDE solvers
secretor=self.getFieldSecretor("FIELDNAME") # you may reuse secretor for many cells. Simply define it outside the loop
secretor.secreteInsideCellAtCOM(cell,300)
"""

        self.snippetDict["Extra Fields Scalar Field Cell Level - Example"]="""
clearScalarValueCellLevel(self.scalarCLField)
from random import random
for cell in self.cellList:
    fillScalarValueCellLevel(self.scalarCLField,cell,cell.id*random())
"""

        self.snippetDict["Extra Fields Scalar Field Cell Level - Create"]="""
self.scalarCLField=CompuCellSetup.createScalarFieldCellLevelPy("FIELD_NAME_SCL")
"""

        self.snippetDict["Extra Fields Scalar Field Cell Level - Clear"]="""
clearScalarValueCellLevel(self.scalarCLField)
"""

        self.snippetDict["Extra Fields Scalar Field Cell Level - Write"]="""
fillScalarValueCellLevel(self.scalarCLField,cell,FLOAT_VALUE)
"""

        self.snippetDict["Extra Fields Scalar Field Pixel Level - Example"]="""
clearScalarField(self.dim,self.scalarField)
from math import sin
for x in xrange(self.dim.x):
    for y in xrange(self.dim.y):
        for z in xrange(self.dim.z):
            pt=CompuCell.Point3D(x,y,z)
            if (not mcs%20):
                value=x*y
                fillScalarValue(self.scalarField,x,y,z,value)
            else:                
                value=sin(x*y)
                fillScalarValue(self.scalarField,x,y,z,value)
"""

        self.snippetDict["Extra Fields Scalar Field Pixel Level - Create"]="""
self.scalarField=CompuCellSetup.createScalarFieldPy(self.dim,"FIELD_NAME_S")
"""

        self.snippetDict["Extra Fields Scalar Field Pixel Level - Clear"]="""
clearScalarField(self.dim,self.scalarField)
"""

        self.snippetDict["Extra Fields Scalar Field Pixel Level - Write"]="""
fillScalarValue(self.scalarField,pt.x,pt.y,pt.z,FLOAT_VALUE) # value assigned to individual pixel
"""

        self.snippetDict["Extra Fields Vector Field Cell Level - Example"]="""
clearVectorCellLevelField(self.vectorCLField)
for cell in self.cellList:
    if cell.type==1:
        insertVectorIntoVectorCellLevelField(self.vectorCLField,cell, cell.id, cell.id, 0.0)
"""


        self.snippetDict["Extra Fields Vector Field Cell Level - Create"]="""
self.vectorCLField=CompuCellSetup.createVectorFieldCellLevelPy("FIELD_NAME_VCL")        
"""

        self.snippetDict["Extra Fields Vector Field Cell Level - Clear"]="""
clearVectorCellLevelField(self.vectorCLField)
"""

        self.snippetDict["Extra Fields Vector Field Cell Level - Write"]="""
insertVectorIntoVectorCellLevelField(self.vectorCLField,cell, VEC_X, VEC_Y, VEC_Z)            
"""
        self.snippetDict["Extra Fields Vector Field Pixel Level - Example"]="""
clearVectorField(self.dim,self.vectorField)
for x in xrange(0,self.dim.x,5):
    for y in xrange(0,self.dim.y,5):
        for z in xrange(self.dim.z):             
            pt=CompuCell.Point3D(x,y,z)            
            insertVectorIntoVectorField(self.vectorField,pt.x, pt.y,pt.z, pt.x, pt.y, pt.z)
"""

        self.snippetDict["Extra Fields Vector Field Pixel Level - Create"]="""
self.vectorField=CompuCellSetup.createVectorFieldPy(self.dim,"FIELD_NAME_V")
"""

        self.snippetDict["Extra Fields Vector Field Pixel Level - Clear"]="""
clearVectorField(self.dim,self.vectorField)
"""

        self.snippetDict["Extra Fields Vector Field Pixel Level - Write"]="""
insertVectorIntoVectorField(self.vectorField,pt.x, pt.y,pt.z,VEC_X, VEC_Y, VEC_Z)                                                
"""

        self.snippetDict["Scientific Plots Setup (start fcn)"]="""
import CompuCellSetup  
self.pW=CompuCellSetup.viewManager.plotManager.getNewPlotWindow()
if not self.pW:
    return
#Plot Title - properties           
self.pW.setTitle("PLOT TITLE")
self.pW.setTitleSize(12)
self.pW.setTitleColor("Green") # you may choose different color - type its name

#plot background
self.pW.setPlotBackgroundColor("orange") # you may choose different color - type its name

# properties of x axis
self.pW.setXAxisTitle("TITLE OF Y AXIS")
self.pW.setXAxisTitleSize(10)      
self.pW.setXAxisTitleColor("blue")  # you may choose different color - type its name            

# properties of y axis
self.pW.setYAxisTitle("TITLE OF Y AXIS")        
self.pW.setYAxisLogScale()
self.pW.setYAxisTitleSize(10)        
self.pW.setYAxisTitleColor("red")  # you may choose different color - type its name                                

# choices for style are NoCurve,Lines,Sticks,Steps,Dots
self.pW.addPlot("DATA_SERIES_1",_style='Dots')
#self.pW.addPlot("DATA SERIES 2",_style='Steps') # you may add more than one data series

# plot MCS
self.pW.changePlotProperty("DATA_SERIES_1","LineWidth",5)
self.pW.changePlotProperty("DATA_SERIES_1","LineColor","red")     

self.pW.addGrid()
#adding automatically generated legend
# default possition is at the bottom of the plot but here we put it at the top
self.pW.addAutoLegend("top")

self.clearFlag=False
"""

        self.snippetDict["Scientific Plots Add Data Points (step fcn)"]="""
#self.pW.eraseAllData() # this is how you erase previous content of the plot
self.pW.addDataPoint("DATA_SERIES_1",mcs,mcs*mcs) # arguments are (name of the data series, x, y)
"""
        self.snippetDict["Scientific Plots Refresh Plots"]="""
self.pW.showAllPlots() 
"""        

