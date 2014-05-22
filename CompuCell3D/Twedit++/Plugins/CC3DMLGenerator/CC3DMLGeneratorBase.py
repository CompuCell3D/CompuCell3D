from XMLUtils import ElementCC3D
import os.path
'''
ideally instead hard-coding snippets we should use XML Schema or RelaxNG formats to describe and help generate CC3DML

'''
import functools    
from functools import wraps

class GenerateDecorator(object):
    def __init__(self, _moduleType,_moduleName):
        self.moduleType=_moduleType
        self.moduleName=_moduleName
        
    def __call__(self, _decoratedFn):
        @functools.wraps(_decoratedFn)
        def decorator(*args, **kwds):
            obj=args[0]
            
            try:
                moduleAttributeLabel=self.moduleName[0]
            except:
                moduleAttributeLabel=''
            
            try:
                moduleAttributeVal=self.moduleName[1]
            except:
                moduleAttributeVal=''
            
            try:
                irElement=kwds['insert_root_element'] 
            except LookupError,e:
                irElement=None
                
            # existing root element 
            try:
                rElement=kwds['root_element'] 
            except LookupError,e:
                rElement=None

            attr={}
            if moduleAttributeLabel!='' and moduleAttributeVal!='':
                attr={moduleAttributeLabel:moduleAttributeVal}
            # checking for additional attributes    e.g. Frequency
            try:
                for idx in range(2,len(self.moduleName),2):
                    moduleAttributeLabel=self.moduleName[idx]
                    moduleAttributeVal=self.moduleName[idx+1]
                    attr[moduleAttributeLabel]=moduleAttributeVal
            except IndexError,e:
                pass
                
            # mElement is module element - either steppable of plugin element 
            if irElement is None:    
                    
                mElement=ElementCC3D(self.moduleType,attr)       
                
                # print '\n\n\n\n THIS IS INITIALIZED ROOT ELEMENT ROOT  irElement=',irElement
            else:
                irElement.addComment("newline")
                
                mElement=irElement.ElementCC3D(self.moduleType,attr)       
            
            try:
                cellTypeData=kwds['data']
            except LookupError,e:
                cellTypeData=None
                
            try:
                generalPropertiesData=kwds['generalPropertiesData']
            except LookupError,e:
                generalPropertiesData={}
        
            gpd=generalPropertiesData            
     
            moduleAttributeLabel=self.moduleName[0]
            
            
            print 'CELLTYPE DATA FROM DECORATOR=',cellTypeData    
            
            
            obj.cellTypeData=cellTypeData
            obj.mElement=mElement
            obj.gpd=gpd            
            
            _decoratedFn(gpd=gpd,cellTypeData=cellTypeData,*args,**kwds)
            
            return mElement
            
        return decorator


class CC3DMLGeneratorBase:
    def __init__(self,simulationDir='',simulationName=''):
        self.element=None
        version='3.6.2'
        revision=''
        try:
            
            from Version import getVersionAsString,getSVNRevision
            
            version = getVersionAsString()
            revision=getSVNRevision()            
        except ImportError:
            print 'COULD NOT IMPORT Version.py'            
        except:
            pass
            
        self.cc3d=ElementCC3D("CompuCell3D",{"Version":version,'Revision':revision})   
        # self.cc3d=ElementCC3D("CompuCell3D",{"version":"3.6.2"})
        self.simulationDir=simulationDir
        self.simulationName=simulationName
        self.fileName=''
        
        
        if self.simulationDir!='' and self.simulationName!='':
            self.fileName=os.path.join(str(self.simulationDir),str(self.simulationName)+".xml")
            
            
    def checkIfSim3D(self,_gpd):
        sim3DFlag=False
        
        if _gpd["Dim"][0]>1 and _gpd["Dim"][1]>1 and _gpd["Dim"][2]>1:
            sim3DFlag=True
            
        return sim3DFlag
        
    
        
    @GenerateDecorator('Potts',['',''])        
    def generatePottsSection(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd
        print '\n\n\n\n gpd=',gpd
        
        mElement.addComment("newline")
        mElement.addComment("Basic properties of CPM (GGH) algorithm")        
        
        
        
        
        mElement.ElementCC3D("Dimensions",{"x":gpd["Dim"][0],"y":gpd["Dim"][1],"z":gpd["Dim"][2]})
        mElement.ElementCC3D("Steps",{},gpd["MCS"])
        mElement.ElementCC3D("Temperature",{},gpd["MembraneFluctuations"])
        mElement.ElementCC3D("NeighborOrder",{},gpd["NeighborOrder"])   
        if gpd["LatticeType"] != "Square":
            mElement.ElementCC3D("LatticeType",{},gpd["LatticeType"])   

    @GenerateDecorator('Metadata',['',''])        
    def generateMetadataSimulationProperties(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        
        mElement.addComment("newline")
        mElement.addComment("Basic properties simulation")        
        
        
        mElement.ElementCC3D("NumberOfProcessors",{},2)
        mElement.ElementCC3D("DebugOutputFrequency",{},100)        
        nonParallelElem=mElement.ElementCC3D("NonParallelModule",{"Name":"Potts"})
        nonParallelElem.commentOutElement()        

    @GenerateDecorator('Metadata',['',''])        
    def generateMetadataDebugOutputFrequency(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement

        mElement.ElementCC3D("DebugOutputFrequency",{},100)        
        

    @GenerateDecorator('Metadata',['',''])        
    def generateMetadataParallelExecution(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement

        mElement.ElementCC3D("NumberOfProcessors",{},2)   


    @GenerateDecorator('Metadata',['',''])        
    def generateMetadataParallelExecutionSingleCPUPotts(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement

        mElement.ElementCC3D("NumberOfProcessors",{},2)   
        mElement.ElementCC3D("NonParallelModule",{"Name":"Potts"})
        
            
    @GenerateDecorator('Plugin',['Name','CellType'])         
    def generateCellTypePlugin(self,*args,**kwds):
    
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd
            
        mElement.addComment("newline")
        mElement.addComment("Listing all cell types in the simulation")        
            
        for id, typeTuple in cellTypeData.iteritems():
            dict={}
            dict["TypeName"]=typeTuple[0]

            dict["TypeId"]=str(id)
            if typeTuple[1]:
                dict["Freeze"]=""
            mElement.ElementCC3D("CellType", dict)
    
    def generateVolumeFlexPlugin(self,*args,**kwds):
        kwds['KeyString']='Volume'
        kwds['KeyType']='Flex'
        return  self.volSurHelper(*args,**kwds)

    def generateVolumeLocalFlexPlugin(self,*args,**kwds):
        kwds['KeyString']='Volume'
        kwds['KeyType']='LocalFlex'
        return  self.volSurHelper(*args,**kwds)

    def generateSurfaceFlexPlugin(self,*args,**kwds):
        kwds['KeyString']='Surface'
        kwds['KeyType']='Flex'
        return  self.volSurHelper(*args,**kwds)

    def generateSurfaceLocalFlexPlugin(self,*args,**kwds):
        kwds['KeyString']='Surface'
        kwds['KeyType']='LocalFlex'
        return  self.volSurHelper(*args,**kwds)
        
    def volSurHelper(self,*args,**kwds):
    
        try:
            irElement=kwds['insert_root_element'] 
        except LookupError,e:
            irElement=None
                
        # existing root element 
        try:
            rElement=kwds['root_element'] 
        except LookupError,e:
            rElement=None

        try:
            cellTypeData=kwds['data']
        except LookupError,e:
            cellTypeData=None

        try:
            keyString=str(kwds['KeyString'])
        except LookupError,e:
            keyString='Volume'

        try:
            keyType=str(kwds['KeyType'])
        except LookupError,e:
            keyType='LocalFlex'

        try:
            constraintDataDict=kwds['constraintDataDict']
        except LookupError,e:
            constraintDataDict={}
            
        # mElement is module element - either steppable of plugin element 
        if irElement is None:    
        
            mElement=ElementCC3D("Plugin",{"Name":keyString})       
            
            # print '\n\n\n\n THIS IS INITIALIZED ROOT ELEMENT ROOT  irElement=',irElement
        else:
            irElement.addComment("newline")
            mElement=irElement.ElementCC3D("Plugin",{"Name":keyString})                        
       
        if keyType=='LocalFlex':
            return mElement
            
                
        maxId=max(cellTypeData.keys())
        
        for typeId in range(0,maxId+1):
            targetVal=25.0    
            lambdaVal=2.0    
        
            if typeId==0:# Medium
                continue
            # first see if entry for this type exists
            try:
                dataList=constraintDataDict[cellTypeData[typeId][0]]
            except LookupError,e:                    
                dataList=[25,2.0]
                # continue               

            try:                
                # volumeDataList=volumeDataDict[cellTypeData[typeId][0]]
                targetVal=dataList[0]
                lambdaVal=dataList[1]
            except LookupError:
                pass
                
            attrDict={'CellType':cellTypeData[typeId][0],'Target'+keyString:targetVal,'Lambda'+keyString:lambdaVal}            
            mElement.ElementCC3D(keyString+"EnergyParameters",attrDict)
            
        return mElement

    @GenerateDecorator('Plugin',['Name','LengthConstraint'])        
    def generateLengthConstraintPlugin(self,*args,**kwds):

        
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd

        
        try:
            constraintDataDict=kwds['constraintDataDict']
        except LookupError,e:
            constraintDataDict={}
            
    
        mElement.addComment("newline")
        mElement.addComment("Applies elongation constraint to each cell. Users specify target length of major axis -TargetLength (in 3D additionally, target length of minor axis - MinorTargetLength) and a strength of the constraint -LambdaLength. Parameters are specified for each cell type")        
        mElement.addComment("IMPORTANT: To prevent cell fragmentation for large elongations you need to also use connectivity constraint")
        mElement.addComment("LengthConstrainLocalFlex allows constrain specification for each cell individually but currently works only in 2D")                                
        mElement.addComment("Comment out the constrains for cell types which dont need them")                
        

                
        sim3DFlag=self.checkIfSim3D(gpd)
            
        maxId=max(cellTypeData.keys())        
        for id1 in range(0,maxId+1):
            if cellTypeData[id1][0]=="Medium":
                continue
                
            targetLength=25.0    
            minorTargetLength=5.0
            lambdaVal=2.0    
            
        
            # if typeId==0:# Medium
                # continue
            # first see if entry for this type exists
            try:
                dataList=constraintDataDict[cellTypeData[id1][0]]
            except LookupError,e:                    
                dataList=[25,5.0,2.0]
                # continue               

            targetLength=dataList[0]
            minorTargetLength=dataList[1]
            lambdaVal=dataList[2]    
                
                
            attr={"CellType":cellTypeData[id1][0],"TargetLength":targetLength,"LambdaLength":lambdaVal} 
            if sim3DFlag:
                attr["MinorTargetLength"]=minorTargetLength
            mElement.ElementCC3D("LengthEnergyParameters",attr)
            
        
        
        
        

        
    @GenerateDecorator('Plugin',['Name','LengthConstraintLocalFlex'])        
    def generateLengthConstraintLocalFlexPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd     
        
        mElement.addComment("newline")
        mElement.addComment("Applies elongation constraint to each cell. Users specify the length major axis -TargetLength and a strength of the constraint -LambdaLength. Parameters are specified for each cell individually")        
        mElement.addComment("IMPORTANT: To prevent cell fragmentation for large elongations you need to also use connectivity constraint")
        mElement.addComment("This plugin currently works only in 2D. Use the following Python syntax to set/modify length constraint:")               
        mElement.addComment("self.lengthConstraintFlexPlugin.setLengthConstraintData(cell,20,30)  # cell , lambdaLength, targetLength  ")      
        
    @GenerateDecorator('Plugin',['Name','ExternalPotential'])                
    def generateExternalPotentialPlugin(self,*args,**kwds):
    
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd  
        
        mElement.addComment("newline")
        mElement.addComment("External force applied to cell. Each cell type has different force.")
        mElement.addComment("For more flexible specification of the constraint (done in Python) please use ExternalPotentialLocalFlex plugin")
    
        mElement.addComment("Algorithm options are: PixelBased, CenterOfMassBased")
        mElement.ElementCC3D("Algorithm",{},"PixelBased")    
        
        maxId=max(cellTypeData.keys())
        for id1 in range(0,maxId+1):
            if cellTypeData[id1][0]=="Medium":
                continue
            dict={"CellType":cellTypeData[id1][0],"x":-0.5,"y":0.0,"z":0.0}    
            mElement.ElementCC3D("ExternalPotentialParameters",dict)    
            

    @GenerateDecorator('Plugin',['Name','ExternalPotentialLocalFlex'])                        
    def generateExternalPotentialLocalFlexPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd 
        
        mElement.addComment("newline")
        mElement.addComment("External force applied to cell. Each cell has different force and force components have to be managed in Python.")
        mElement.addComment("e.g. cell.lambdaVecX=0.5; cell.lambdaVecY=0.1 ; cell.lambdaVecZ=0.3;")
                
        mElement.addComment("Algorithm options are: PixelBased, CenterOfMassBased")
        mElement.ElementCC3D("Algorithm",{},"PixelBased")    
        
    @GenerateDecorator('Plugin',['Name','CenterOfMass'])    
    def generateCenterOfMassPlugin(self,*args,**kwds):

        mElement=self.mElement
        mElement.addComment("newline")
        mElement.addComment("Module tracking center of mass of each cell")        
      
    @GenerateDecorator('Plugin',['Name','NeighborTracker'])         
    def generateNeighborTrackerPlugin(self,*args,**kwds):

        mElement=self.mElement

        mElement.addComment("newline")
        mElement.addComment("Module tracking neighboring cells of each cell")        
    
        
    @GenerateDecorator('Plugin',['Name','MomentOfInertia'])     
    def generateMomentOfInertiaPlugin(self,*args,**kwds):
        mElement=self.mElement
        
        mElement.addComment("newline")
        mElement.addComment("Module tracking moment of inertia of each cell")                    
        
    @GenerateDecorator('Plugin',['Name','PixelTracker'])     
    def generatePixelTrackerPlugin(self,*args,**kwds):
        mElement=self.mElement
    
        mElement.addComment("newline")
        mElement.addComment("Module tracking pixels of each cell")        
        
    @GenerateDecorator('Plugin',['Name','BoundaryPixelTracker'])         
    def generateBoundaryPixelTrackerPlugin(self,*args,**kwds):
        mElement=self.mElement
        
        mElement.addComment("newline")
        mElement.addComment("Module tracking boundary pixels of each cell")            
        
        mElement.ElementCC3D("NeighborOrder",{},1)
        
    @GenerateDecorator('Plugin',['Name','CellTypeMonitor'])    
    def generateCellTypeMonitorPlugin(self,*args,**kwds):
        mElement=self.mElement
        mElement.addComment("newline")
        mElement.addComment("Module tracking cell types at each lattice site - used mainly by pde solvers")            
        
    @GenerateDecorator('Plugin',['Name','ConnectivityGlobal'])  
    def generateConnectivityGlobalPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd  
    
        mElement.addComment("newline")
        mElement.addComment("Connectivity constraint applied to each cell. Energy penalty specifies how strong the constraint is. Penalty is specified for each type ")
        mElement.addComment("This constraint works in 2D and 3D on all type of lattices. It might be slowdown your simulation. For faster option - 2D and square lattice you may use Connectivity or ConnectivityLocalFlex")
        mElement.addComment("To speed up simulation comment out unnecessary constraints for types which don't need the constraint")
        
                
        mElement.addComment("By default we will always precheck connectivity BUT in simulations in which there is no risk of having unfragmented cell one can add this flag to speed up computations")
        mElement.addComment("To turn off precheck uncomment line below")
        precheckElem=mElement.ElementCC3D("DoNotPrecheckConnectivity")
        precheckElem.commentOutElement()
        
        maxId=max(cellTypeData.keys())
        for id1 in range(0,maxId+1):
            if cellTypeData[id1][0]=="Medium":
                continue
            attr={"Type":cellTypeData[id1][0]}    
            mElement.ElementCC3D("Penalty",attr,1000000)

    @GenerateDecorator('Plugin',['Name','ConnectivityGlobal'])      
    def generateConnectivityGlobalByIdPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd      
    
        mElement.addComment("newline")
        mElement.addComment("Connectivity constraint applied to each cell. Energy penalty specifies how strong the constraint is. Penalty is specified for each cell type individually ")
        mElement.addComment("Use Python scripting to setup penalty (connectivity strength) for each cell")  
        mElement.addComment("e.g. self.connectivityGlobalPlugin.setConnectivityStrength(cell,20000000) #cell, connectivity strength")        
        mElement.addComment("This constraint works in 2D and 3D on all type of lattices. It might be slowdown your simulation. For faster option - 2D and square lattice you may use Connectivity or ConnectivityLocalFlex")
        mElement.addComment("To speed up simulation comment out unnecessary constraints for types which don't need the constraint")

    @GenerateDecorator('Plugin',['Name','Connectivity'])       
    def generateConnectivityPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd      
                    
        mElement.addComment("Connectivity constraint applied to each cell. Energy penalty specifies how strong the constraint is. Penalty is specified globally for each cell ")        
        mElement.addComment("This constraint works in 2D and on square lattice only!")
        mElement.addComment("For more flexible version of this plugin use ConnectivityLocalFlex where constraint penalty is specified for each cell individually using Python scripting using the following syntax")
        mElement.addComment("self.connectivityLocalFlexPlugin.setConnectivityStrength(cell,10000000)" )
        
        mElement.ElementCC3D("Penalty",{},10000000)
        
    @GenerateDecorator('Plugin',['Name','Contact'])       
    def generateContactPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd      
        
        try:    
            contactMatrix=kwds['contactMatrix']
        except LookupError,e:
            contactMatrix={}
        
        try:    
            nOrder=kwds['NeighborOrder']
        except LookupError,e:
            nOrder=1
                    
        
        mElement.addComment("Specification of adhesion energies")        

        maxId=max(cellTypeData.keys())
        for id1 in range(0,maxId+1):
            for id2 in range(id1,maxId+1):   
                try:
                    attrDict={"Type1":cellTypeData[id1][0],"Type2":cellTypeData[id2][0]}
                except LookupError,e:                    
                    continue                
                    
                try:                                
                    # first see if energy exists
                    energy=contactMatrix[cellTypeData[id1][0]][cellTypeData[id2][0]][0]
                except LookupError:
                    try: # try reverse order
                        energy=contactMatrix[cellTypeData[id2][0]][cellTypeData[id1][0]][0]

                    except LookupError: # use default value
                        energy=10.0
                    
                mElement.ElementCC3D("Energy",attrDict,energy)
        mElement.ElementCC3D("NeighborOrder",{},nOrder)        
                    
        

    @GenerateDecorator('Plugin',['Name','ContactInternal']) 
    def generateContactInternalPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd         
            
        try:    
            contactMatrix=kwds['contactMatrix']
        except LookupError,e:
            contactMatrix={}
        
        try:    
            nOrder=kwds['NeighborOrder']
        except LookupError,e:
            nOrder=1
                    
        
        mElement.addComment("Specification of internal adhesion energies")        

        maxId=max(cellTypeData.keys())
        for id1 in range(1,maxId+1):
            for id2 in range(id1,maxId+1):   
                try:
                    attrDict={"Type1":cellTypeData[id1][0],"Type2":cellTypeData[id2][0]}
                except LookupError,e:                    
                    continue                
                    
                try:                                
                    # first see if energy exists
                    energy=contactMatrix[cellTypeData[id1][0]][cellTypeData[id2][0]][0]
                except LookupError:
                    try: # try reverse order
                        energy=contactMatrix[cellTypeData[id2][0]][cellTypeData[id1][0]][0]

                    except LookupError: # use default value
                        energy=10.0
                    
                mElement.ElementCC3D("Energy",attrDict,energy)
        mElement.ElementCC3D("NeighborOrder",{},nOrder)                            
        
    @GenerateDecorator('Plugin',['Name','Compartment'])
    def generateCompartmentPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd      
            
        try:    
            contactMatrix=kwds['contactMatrix']
        except LookupError,e:
            contactMatrix={}


        try:    
            internalContactMatrix=kwds['contactMatrix']
        except LookupError,e:
            internalContactMatrix={}

            
        try:    
            nOrder=kwds['NeighborOrder']
        except LookupError,e:
            nOrder=1
                    
        
        mElement.addComment("newline")
        mElement.addComment("Specification of adhesion energies in the presence of compartmental cells")        
        mElement.addComment("This plugin is deprecated - please consider using Contact and ContactInternal plugins instead")
        mElement.addComment("to specify adhesions bewtween members of same cluster")


        maxId=max(cellTypeData.keys())
        for id1 in range(0,maxId+1):
            for id2 in range(id1,maxId+1):   
                try:
                    attrDict={"Type1":cellTypeData[id1][0],"Type2":cellTypeData[id2][0]}
                except LookupError,e:                    
                    continue                
                    
                try:                                
                    # first see if energy exists
                    energy=contactMatrix[cellTypeData[id1][0]][cellTypeData[id2][0]][0]
                except LookupError:
                    try: # try reverse order
                        energy=contactMatrix[cellTypeData[id2][0]][cellTypeData[id1][0]][0]

                    except LookupError: # use default value
                        energy=10.0
                    
                mElement.ElementCC3D("Energy",attrDict,energy)
        
    
        # energy between members of same clusters
        for id1 in range(1,maxId+1):
            for id2 in range(id1,maxId+1):            
                                
                try:
                    attrDict={"Type1":cellTypeData[id1][0],"Type2":cellTypeData[id2][0]}
                except LookupError,e:                    
                    continue                        
                # first see if energy exists
                try:
                    
                    internalEnergy=internalContactMatrix[cellTypeData[id1][0]][cellTypeData[id2][0]][0]
                    
                    
                except LookupError:
                    try: # try reverse order
                        internalEnergy=internalContactMatrix[cellTypeData[id2][0]][cellTypeData[id1][0]][0]

                    except LookupError: # use default value
                        internalEnergy=5.0
                
                mElement.ElementCC3D("InternalEnergy",attrDict,internalEnergy)
        mElement.ElementCC3D("NeighborOrder",{},nOrder)        
        
        
        
    @GenerateDecorator('Plugin',['Name','ContactLocalProduct'])
    def generateContactLocalProductPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd      
            
        try:    
            specificityMatrix=kwds['specificityMatrix']
        except LookupError,e:
            specificityMatrix={}
            
        try:    
            nOrder=kwds['NeighborOrder']
        except LookupError,e:
            nOrder=1

    
        mElement.addComment("newline")
        mElement.addComment("Specification of adhesion energies as a function of cadherin concentration at cell membranes")
        mElement.addComment("Adhesion energy is a function of two cells in ocntact. the functional form is specified by the user")
        mElement.addComment("Please consider using more flexible version of this plugin - AdhesionFlex")
        
        
        maxId=max(cellTypeData.keys())
        for id1 in range(0,maxId+1):
            for id2 in range(id1,maxId+1):            
                attr={"Type1":cellTypeData[id1][0],"Type2":cellTypeData[id2][0]}

                try:                    
                    specificity=specificityMatrix[cellTypeData[id1][0]][cellTypeData[id2][0]][0]
                except LookupError:
                    try: # try reverse order
                        specificity=specificityMatrix[cellTypeData[id2][0]][cellTypeData[id1][0]][0]

                    except LookupError: # use default value
                        specificity=-1.0
                
                mElement.ElementCC3D("ContactSpecificity",attr,specificity)
                
        mElement.ElementCC3D("ContactFunctionType",{},"linear")
        
        mElement.ElementCC3D("EnergyOffset",{},0.0)                        
        mElement.ElementCC3D("NeighborOrder",{},nOrder)     
        
    @GenerateDecorator('Plugin',['Name','FocalPointPlasticity'])    
    def generateFocalPointPlasticityPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd     
    
                        
        try:    
            nOrder=kwds['NeighborOrder']
        except LookupError,e:
            nOrder=1

    
        mElement.addComment("newline")
        mElement.addComment("Specification of focal point junctions")
        mElement.addComment("We separetely specify links between members of same cluster - InternalParameters and members of different clusters Parameters. When not using compartmental  cells comment out InternalParameters specification")
                
        mElement.addComment("To modify FPP links individually for each cell pair uncomment line below")        
        localmElement=mElement.ElementCC3D("Local")
        localmElement.commentOutElement()        
        mElement.addComment("Note that even though you may manipulate lambdaDistance, targetDistance and maxDistance using Python you still need to set activation energy from XML level")
        mElement.addComment("See CC3D manual for details on FPP plugin ")
        
        
        maxId=max(cellTypeData.keys())

        for id1 in range(1,maxId+1):
            for id2 in range(id1,maxId+1):            
                mElement.addComment("newline")
                attr={"Type1":cellTypeData[id1][0],"Type2":cellTypeData[id2][0]}                            
                paramElement=mElement.ElementCC3D("Parameters",attr)
                paramElement.ElementCC3D("Lambda",{},10)
                paramElement.ElementCC3D("ActivationEnergy",{},-50)
                paramElement.ElementCC3D("TargetDistance",{},7)
                paramElement.ElementCC3D("MaxDistance",{},20)
                paramElement.ElementCC3D("MaxNumberOfJunctions",{"NeighborOrder":1},1)
                
                
        for id1 in range(1,maxId+1):
            for id2 in range(id1,maxId+1):            
                mElement.addComment("newline")                
                attr={"Type1":cellTypeData[id1][0],"Type2":cellTypeData[id2][0]}                 
                paramElement=mElement.ElementCC3D("InternalParameters",attr)
                paramElement.ElementCC3D("Lambda",{},10)
                paramElement.ElementCC3D("ActivationEnergy",{},-50)
                paramElement.ElementCC3D("TargetDistance",{},7)
                paramElement.ElementCC3D("MaxDistance",{},20)
                paramElement.ElementCC3D("MaxNumberOfJunctions",{"NeighborOrder":1},1)
                
        mElement.addComment("newline")        
        mElement.ElementCC3D("NeighborOrder",{},1)
        
    @GenerateDecorator('Plugin',['Name','ElasticityTracker'])       
    def generateElasticityTrackerPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd      
                            
        mElement.addComment("newline")
        mElement.addComment("Elastic constraints between Center of mass of cells. Need to be accompanied by ElasticityTracker plugin to work. Only cells in contact at MCS=0 will be affected by the constraint")        
        
        mElement.addComment("ElasticityTracker keeps track of cell neighbors which are participating in the elasticity constraint calculations")        
        
        mElement.addComment("Comment out cell types which should be unaffected by the constraint")
        
        maxId=max(cellTypeData.keys())
        for id1 in range(0,maxId+1):
            if cellTypeData[id1][0]=="Medium":
                continue            
            mElement.ElementCC3D("IncludeType",{},cellTypeData[id1][0]) 
            

    @GenerateDecorator('Plugin',['Name','Elasticity'])             
    def generateElasticityPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd      
                        
        mElement.addComment("newline")            
        mElement.addComment("This plugin calculates elastic constraints between cells Center of Mass")            
        mElement.addComment("To enable specification of elastic links individually for each link uncomment line below ")
        localElem=mElement.ElementCC3D("Local")
        localElem.commentOutElement()        
        mElement.addComment("See CC3D manual for details")

        mElement.ElementCC3D("LambdaElasticity",{},200) 
        mElement.ElementCC3D("TargetLengthElasticity",{},6) 
        
    @GenerateDecorator('Plugin',['Name','AdhesionFlex'])      
    def generateAdhesionFlexPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd     
        
        try:
            afData=kwds['afData']
        except LookupError,e:
            afData={}

        try:
            formula=kwds['formula']
        except LookupError,e:
            formula=''
            
    
        mElement.addComment("newline")
        mElement.addComment("Specification of adhesion energies as a function of cadherin concentration at cell membranes")
        mElement.addComment("Adhesion energy is a function of two cells in ocntact. the functional form is specified by the user")
    
        #writing AdhesionMolecule elements
        for idx,props in afData.iteritems():            
            attrDict={"Molecule":props}                            
            mElement.ElementCC3D("AdhesionMolecule",attrDict)
            
        #writing AdhesionMoleculeDensity elements    
        
        for typeId,props in cellTypeData.iteritems():
            for idx,afprops in afData.iteritems():      
                
                attrDict={"CellType":props[0],"Molecule":afprops,"Density":1.1}                            
                mElement.ElementCC3D("AdhesionMoleculeDensity",attrDict)
        
        # writing binding formula
        bfElement=mElement.ElementCC3D("BindingFormula",{'Name':'Binary'})
        bfElement.ElementCC3D("Formula",{},formula)
        varElement=bfElement.ElementCC3D("Variables")
        adhMatrixElement=varElement.ElementCC3D("AdhesionInteractionMatrix")
        
        repetitionDict={}
        
        for idx1,afprops1 in afData.iteritems():      
            for idx2,afprops2 in afData.iteritems():                  
                if afprops2+'_'+afprops1 in repetitionDict.keys(): # to avoid duplicate entries
                    continue
                else:    
                    repetitionDict[afprops1+'_'+afprops2]=0
                
                attrDict={"Molecule1":afprops1,"Molecule2":afprops2}                            
                adhMatrixElement.ElementCC3D("BindingParameter",attrDict,0.5)
                
        mElement.ElementCC3D("NeighborOrder",{},2)      
        

        
 
    @GenerateDecorator('Plugin',['Name','Chemotaxis'])         
    def generateChemotaxisPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd       
        
        try:
            chemotaxisData=kwds['chemotaxisData']
        except LookupError,e:
            chemotaxisData={}

        try:
            pdeFieldData=kwds['pdeFieldData']
        except LookupError,e:
            pdeFieldData={}
            
    
        mElement.addComment("newline")
        mElement.addComment('You may repeat ChemicalField element for each chemical field declared in the PDE solvers')        
        mElement.addComment("Specification of chemotaxis properties of select cell types.")
    
        for chemFieldName, chemDictList in chemotaxisData.iteritems():
            
            try:
                fieldSourceName=pdeFieldData[chemFieldName]
            except LookupError,e:
                fieldSourceName='PDE_SOLVER'
                
            chemFieldElement=mElement.ElementCC3D("ChemicalField",{"Source":fieldSourceName,"Name":chemFieldName})
            for chemDict in chemDictList:
                
                lambda_=chemDict["Lambda"]
                chemotaxTowards=chemDict["ChemotaxTowards"]
                satCoef=chemDict["SatCoef"]
                chemotaxisType=chemDict["ChemotaxisType"]
                
                attributeDict={}
                attributeDict["Type"]=chemDict["CellType"]
                attributeDict["Lambda"]=chemDict["Lambda"]
                if chemDict["ChemotaxTowards"]!='':
                    attributeDict["ChemotactTowards"]=chemDict["ChemotaxTowards"]
                if chemDict["ChemotaxisType"]=='saturation':
                    attributeDict["SaturationCoef"]=chemDict["SatCoef"]
                elif chemDict["ChemotaxisType"]=='saturation linear':    
                    attributeDict["SaturationLinearCoef"]=chemDict["SatCoef"]
                                   
                chemFieldElement.ElementCC3D("ChemotaxisByType",attributeDict)
                        
    @GenerateDecorator('Plugin',['Name','Secretion'])      
    def generateSecretionPlugin(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd     
        
        try:
            secretionData=kwds['secretionData']
        except LookupError,e:
            secretionData={}

    
        mElement.addComment("newline")
        mElement.addComment("Specification of secretion properties of select cell types.")
        mElement.addComment('You may repeat Field element for each chemical field declared in the PDE solvers')         
        mElement.addComment("Specification of secretion properties of individual cells can be done in Python")        
            
        for chemFieldName, secrDictList in secretionData.iteritems():
            secrFieldElement=mElement.ElementCC3D("Field",{"Name":chemFieldName})
            for secrDict in secrDictList:
                
                rate=secrDict["Rate"]
                
                attributeDict={}
                attributeDict["Type"]=secrDict["CellType"]
                if secrDict["SecretionType"]=='uniform':
                    secrFieldElement.ElementCC3D("Secretion",attributeDict,rate)
                elif secrDict["SecretionType"]=='on contact':
                    attributeDict["SecreteOnContactWith"]=secrDict["OnContactWith"]
                    secrFieldElement.ElementCC3D("SecretionOnContact",attributeDict,rate)
                elif secrDict["SecretionType"]=='constant concentration':      
                    secrFieldElement.ElementCC3D("ConstantConcentration",attributeDict,rate)
        
    @GenerateDecorator('Steppable',['Type','UniformInitializer'])          
    def generateUniformInitializerSteppable(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd  
                
        mElement.addComment("newline")
        mElement.addComment("Initial layout of cells in the form of rectangular slab")
        
        region=mElement.ElementCC3D("Region")
                        
        xMin=int(gpd["Dim"][0]*0.2)
        xMax=int(gpd["Dim"][0]*0.8)
        if xMax==0:
            xMax+=1
        yMin=int(gpd["Dim"][1]*0.2)
        yMax=int(gpd["Dim"][1]*0.8)
        if yMax==0:
            yMax+=1
            
        zMin=int(gpd["Dim"][2]*0.2)
        zMax=int(gpd["Dim"][2]*0.8)
        if zMax==0:
            zMax+=1
            
        region.ElementCC3D("BoxMin",{"x":xMin,"y":yMin,"z":zMin})
        region.ElementCC3D("BoxMax",{"x":xMax,"y":yMax,"z":zMax})
        region.ElementCC3D("Gap",{},0)
        region.ElementCC3D("Width",{},5)
        typesString=""
        
        maxId=max(cellTypeData.keys())
        for id1 in range(0,maxId+1):
            if cellTypeData[id1][0]=="Medium":
                continue
                
            typesString+=cellTypeData[id1][0]
            if id1<maxId:
                typesString+=","
            
        region.ElementCC3D("Types",{},typesString)
                        
    @GenerateDecorator('Steppable',['Type','BlobInitializer'])              
    def generateBlobInitializerSteppable(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd     
        
        mElement.addComment("newline")
        mElement.addComment("Initial layout of cells in the form of spherical (circular in 2D) blob")
        region=mElement.ElementCC3D("Region")
                
        
        xCenter=int(gpd["Dim"][0]/2)
        
        yCenter=int(gpd["Dim"][1]/2)
            
        zCenter=int(gpd["Dim"][2]/2)
        maxDim=max([xCenter,yCenter,zCenter])
            
        region.ElementCC3D("Center",{"x":xCenter,"y":yCenter,"z":zCenter})        
        region.ElementCC3D("Radius",{},int(maxDim/2.5))
        region.ElementCC3D("Gap",{},0)
        region.ElementCC3D("Width",{},5)
        
        
        typesString=""
        
        maxId=max(cellTypeData.keys())
        for id1 in range(0,maxId+1):
            if cellTypeData[id1][0]=="Medium":
                continue
                
            typesString+=cellTypeData[id1][0]
            if id1<maxId:
                typesString+=","
            
        region.ElementCC3D("Types",{},typesString)
        
    @GenerateDecorator('Steppable',['Type','PIFInitializer'])                  
    def generatePIFInitializerSteppable(self,*args,**kwds):  
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd     
        
        mElement.addComment("newline")
        mElement.addComment("Initial layout of cells using PIFF file. Piff files can be generated using PIFGEnerator")
        
        try:    
            if gpd["Initializer"][0]=="piff":
                mElement.ElementCC3D("PIFName",{},gpd["Initializer"][1])
            else:
                mElement.ElementCC3D("PIFName",{},"PLEASE_PUT_PROPER_FILE_NAME_HERE")
        except:    
            mElement.ElementCC3D("PIFName",{},"PLEASE_PUT_PROPER_FILE_NAME_HERE")
            

    @GenerateDecorator('Steppable',['Type','PIFDumper','Frequency','100'])                      
    def generatePIFDumperSteppable(self,*args,**kwds):  
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd     
        
    
        mElement.addComment("newline")
        mElement.addComment("Periodically stores cell layout configuration in a piff format")
        
        mElement.ElementCC3D("PIFName",{},gpd['SimulationName'])
        mElement.ElementCC3D("PIFFileExtension",{},"piff")
        
        
    @GenerateDecorator('Steppable',['Type','BoxWatcher'])                      
    def generateBoxWatcherSteppable(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd       
        
        mElement.addComment("newline")
        mElement.addComment("Module tracing boundaries of the minimal box enclosing all the cells. May speed up calculations. May have no effect for parallel version")
                
        mElement.ElementCC3D("XMargin",{},5)
        mElement.ElementCC3D("YMargin",{},5)
        mElement.ElementCC3D("ZMargin",{},5)                
        
    @GenerateDecorator('Steppable',['Type','DiffusionSolverFE'])      
    def generateDiffusionSolverFE(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd       
        
        try:
            pdeFieldData=kwds['pdeFieldData']
        except LookupError,e:
            pdeFieldData={}
                                
        sim3DFlag=self.checkIfSim3D(gpd)
                
        mElement.addComment("newline")
        mElement.addComment("Specification of PDE solvers")
                    
        for fieldName,solver in pdeFieldData.iteritems():

            if solver=='DiffusionSolverFE':

                diffFieldElem=mElement.ElementCC3D("DiffusionField",{"Name":fieldName})
                
                diffData=diffFieldElem.ElementCC3D("DiffusionData")                    
                diffData.ElementCC3D("FieldName",{},fieldName)
                diffData.ElementCC3D("GlobalDiffusionConstant",{},0.1)
                diffData.ElementCC3D("GlobalDecayConstant",{},0.00001)

                diffData.addComment("Additional options are:")                    
                concEqnElem=diffData.ElementCC3D("InitialConcentrationExpression",{},"x*y")
                concEqnElem.commentOutElement()                    
                concFieldNameElem=diffData.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
                concFieldNameElem.commentOutElement()
                
                maxId=max(cellTypeData.keys())
                for id1 in range(0,maxId+1):
                    if cellTypeData[id1][0]=="Medium":
                        continue
                    diffData.ElementCC3D('DiffusionCoefficient',{'CellType':cellTypeData[id1][0]},0.1)
                        
                for id1 in range(0,maxId+1):
                    if cellTypeData[id1][0]=="Medium":
                        continue
                    diffData.ElementCC3D('DecayCoefficient',{'CellType':cellTypeData[id1][0]},0.0001)

                secrData=diffFieldElem.ElementCC3D("SecretionData")                    
                secrData.addComment('When secretion is defined inside DissufionSolverFEall secretio nconstants are scaled automaticly to account for extra calls of the solver when handling large diffusion constants')
                secrData.addComment('newline')
                secrData.addComment('Uniform secretion Definition')

                maxId=max(cellTypeData.keys())
                for id1 in range(0,maxId+1):
                    if cellTypeData[id1][0]=="Medium":
                        continue
                    secrData.ElementCC3D("Secretion",{"Type":cellTypeData[id1][0]},0.1)                        
                                
                secreteOnContactWith=''
                exampleType=''
                for id1 in range(0,maxId+1):
                    if cellTypeData[id1][0]=="Medium":
                        continue
                    if secreteOnContactWith!='':
                        secreteOnContactWith+=','
                    secreteOnContactWith+=cellTypeData[id1][0]    
                    exampleType=cellTypeData[id1][0]
                    
                    
                secrOnContactElem=secrData.ElementCC3D("SecretionOnContact",{'Type':exampleType,"SecreteOnContactWith":secreteOnContactWith},0.2)                
                secrOnContactElem.commentOutElement()                
                    
                constConcElem=secrData.ElementCC3D("ConstantConcentration",{"Type":exampleType},0.1)
                constConcElem.commentOutElement()
                
                
                #Boiundary Conditions
                bcData=diffFieldElem.ElementCC3D("BoundaryConditions")                    
                planeXElem=bcData.ElementCC3D("Plane",{'Axis':'X'})                     
                planeXElem.ElementCC3D("ConstantValue",{'PlanePosition':'Min','Value':10.0}) 
                planeXElem.ElementCC3D("ConstantValue",{'PlanePosition':'Max','Value':5.0}) 
                planeXElem.addComment("Other options are (examples):")
                periodicXElem=planeXElem.ElementCC3D("Periodic") 
                periodicXElem.commentOutElement()                    
                cdElem=planeXElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                cdElem.commentOutElement()                    

                
                planeYElem=bcData.ElementCC3D("Plane",{'Axis':'Y'}) 
                planeYElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                planeYElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Max','Value':5.0}) 
                planeYElem.addComment("Other options are (examples):")
                periodicYElem=planeYElem.ElementCC3D("Periodic") 
                periodicYElem.commentOutElement()                    
                cvElem=planeYElem.ElementCC3D('ConstantValue',{'PlanePosition':'Min','Value':10.0}) 
                cvElem.commentOutElement()                    
                
                if sim3DFlag:                    
                    planeZElem=bcData.ElementCC3D("Plane",{'Axis':'Z'}) 
                    planeZElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                    planeZElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Max','Value':5.0}) 
                    planeZElem.addComment("Other options are (examples):")
                    periodicZElem=planeZElem.ElementCC3D("Periodic") 
                    periodicZElem.commentOutElement()                    
                    cvzElem=planeZElem.ElementCC3D('ConstantValue',{'PlanePosition':'Min','Value':10.0}) 
                    cvzElem.commentOutElement()                    
                
                    
    @GenerateDecorator('Steppable',['Type','FlexibleDiffusionSolverFE'])   
    def generateFlexibleDiffusionSolverFE(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd 
        
        try:
            pdeFieldData=kwds['pdeFieldData']
        except LookupError,e:
            pdeFieldData={}
                        
        sim3DFlag=self.checkIfSim3D(gpd)
                
        mElement.addComment("newline")
        mElement.addComment("Specification of PDE solvers")
                    
        for fieldName,solver in pdeFieldData.iteritems():

            if solver=='FlexibleDiffusionSolverFE':

                diffFieldElem=mElement.ElementCC3D("DiffusionField",{"Name":fieldName})
                
                diffData=diffFieldElem.ElementCC3D("DiffusionData")                    
                diffData.ElementCC3D("FieldName",{},fieldName)
                diffData.ElementCC3D("DiffusionConstant",{},0.1)
                diffData.ElementCC3D("DecayConstant",{},0.00001)

                diffData.addComment("Additional options are:")                    
                donotdiffuseElem=diffData.ElementCC3D("DoNotDiffuseTo",{},"LIST YOUR CELL TYPES HERE")
                donotdiffuseElem.commentOutElement()                    
                donotdecayinElem=diffData.ElementCC3D("DoNotDecayIn",{},"LIST YOUR CELL TYPES HERE")
                donotdecayinElem.commentOutElement()
                concEqnElem=diffData.ElementCC3D("InitialConcentrationExpression",{},"x*y")
                concEqnElem.commentOutElement()                    
                concFieldNameElem=diffData.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
                concFieldNameElem.commentOutElement()
                diffData.addComment("To run solver for large diffusion constants you typically call solver multiple times - ExtraTimesPerMCS to specify additional calls to the solver in each MCS ")                    
                diffData.addComment("IMPORTANT: make sure not to mix this setting with the PDESolverCaller module! See manual for more information")                    
                extraTimesPerMCSElem=diffData.ElementCC3D("ExtraTimesPerMCS",{},0)
                extraTimesPerMCSElem.commentOutElement()
                
                deltaXElem=diffData.ElementCC3D("DeltaX",{},1.0)
                deltaXElem.commentOutElement()
                deltaTElem=diffData.ElementCC3D("DeltaT",{},1.0)
                deltaTElem.commentOutElement()
                
                #Boiundary Conditions
                bcData=diffFieldElem.ElementCC3D("BoundaryConditions")                    
                planeXElem=bcData.ElementCC3D("Plane",{'Axis':'X'})                     
                planeXElem.ElementCC3D("ConstantValue",{'PlanePosition':'Min','Value':10.0}) 
                planeXElem.ElementCC3D("ConstantValue",{'PlanePosition':'Max','Value':5.0}) 
                planeXElem.addComment("Other options are (examples):")
                periodicXElem=planeXElem.ElementCC3D("Periodic") 
                periodicXElem.commentOutElement()                    
                cdElem=planeXElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                cdElem.commentOutElement()                    

                
                planeYElem=bcData.ElementCC3D("Plane",{'Axis':'Y'}) 
                planeYElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                planeYElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Max','Value':5.0}) 
                planeYElem.addComment("Other options are (examples):")
                periodicYElem=planeYElem.ElementCC3D("Periodic") 
                periodicYElem.commentOutElement()                    
                cvElem=planeYElem.ElementCC3D('ConstantValue',{'PlanePosition':'Min','Value':10.0}) 
                cvElem.commentOutElement()                    
                
                if sim3DFlag:                    
                    planeZElem=bcData.ElementCC3D("Plane",{'Axis':'Z'}) 
                    planeZElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                    planeZElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Max','Value':5.0}) 
                    planeZElem.addComment("Other options are (examples):")
                    periodicZElem=planeZElem.ElementCC3D("Periodic") 
                    periodicZElem.commentOutElement()                    
                    cvzElem=planeZElem.ElementCC3D('ConstantValue',{'PlanePosition':'Min','Value':10.0}) 
                    cvzElem.commentOutElement()                    
                
                            


    @GenerateDecorator('Steppable',['Type','FastDiffusionSolver2DFE'])     
    def generateFastDiffusionSolver2DFE(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd         
        
        try:
            pdeFieldData=kwds['pdeFieldData']
        except LookupError,e:
            pdeFieldData={}
    
        mElement.addComment("newline")
        mElement.addComment("Specification of PDE solvers")
                    
        for fieldName,solver in pdeFieldData.iteritems():

            if solver=='FastDiffusionSolver2DFE':

                diffFieldElem=mElement.ElementCC3D("DiffusionField",{"Name":fieldName})
                
                diffData=diffFieldElem.ElementCC3D("DiffusionData")                    
                diffData.ElementCC3D("FieldName",{},fieldName)
                diffData.ElementCC3D("DiffusionConstant",{},0.1)
                diffData.ElementCC3D("DecayConstant",{},0.00001)

                diffData.addComment("Additional options are:")                    
                donotdiffuseElem=diffData.ElementCC3D("DoNotDiffuseTo",{},"LIST YOUR CELL TYPES HERE")
                donotdiffuseElem.commentOutElement()                    
                donotdecayinElem=diffData.ElementCC3D("DoNotDecayIn",{},"LIST YOUR CELL TYPES HERE")
                donotdecayinElem.commentOutElement()
                concEqnElem=diffData.ElementCC3D("InitialConcentrationExpression",{},"x*y")
                concEqnElem.commentOutElement()                    
                concFieldNameElem=diffData.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
                concFieldNameElem.commentOutElement()
                diffData.addComment("To run solver for large diffusion constants you typically call solver multiple times - ExtraTimesPerMCS to specify additional calls to the solver in each MCS ")                    
                diffData.addComment("IMPORTANT: make sure not to mix this setting with the PDESolverCaller module! See manual for more information")                    
                extraTimesPerMCSElem=diffData.ElementCC3D("ExtraTimesPerMCS",{},0)
                extraTimesPerMCSElem.commentOutElement()
                
                deltaXElem=diffData.ElementCC3D("DeltaX",{},1.0)
                deltaXElem.commentOutElement()
                deltaTElem=diffData.ElementCC3D("DeltaT",{},1.0)
                deltaTElem.commentOutElement()
                
                #Boiundary Conditions
                bcData=diffFieldElem.ElementCC3D("BoundaryConditions")                    
                planeXElem=bcData.ElementCC3D("Plane",{'Axis':'X'})                     
                planeXElem.ElementCC3D("ConstantValue",{'PlanePosition':'Min','Value':10.0}) 
                planeXElem.ElementCC3D("ConstantValue",{'PlanePosition':'Max','Value':5.0}) 
                planeXElem.addComment("Other options are (examples):")
                periodicXElem=planeXElem.ElementCC3D("Periodic") 
                periodicXElem.commentOutElement()                    
                cdElem=planeXElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                cdElem.commentOutElement()                    
            
                planeYElem=bcData.ElementCC3D("Plane",{'Axis':'Y'}) 
                planeYElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                planeYElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Max','Value':5.0}) 
                planeYElem.addComment("Other options are (examples):")
                periodicYElem=planeYElem.ElementCC3D("Periodic") 
                periodicYElem.commentOutElement()                    
                cvElem=planeYElem.ElementCC3D('ConstantValue',{'PlanePosition':'Min','Value':10.0}) 
                cvElem.commentOutElement()                    
                                    

    @GenerateDecorator('Steppable',['Type','KernelDiffusionSolver'])  
    def generateKernelDiffusionSolver(self,*args,**kwds):
        cellTypeData=self.cellTypeData
        mElement=self.mElement
        gpd=self.gpd          
        try:
            pdeFieldData=kwds['pdeFieldData']
        except LookupError,e:
            pdeFieldData={}
            
        mElement.addComment("newline")
        mElement.addComment("Specification of PDE solvers")
                    
        for fieldName,solver in pdeFieldData.iteritems():

            if solver=='KernelDiffusionSolver':

                diffFieldElem=mElement.ElementCC3D("DiffusionField",{"Name":fieldName})
                diffFieldElem.ElementCC3D("Kernel",{},"4")
                diffData=diffFieldElem.ElementCC3D("DiffusionData")
                diffData.ElementCC3D("FieldName",{},fieldName)
                diffData.ElementCC3D("DiffusionConstant",{},0.1)
                diffData.ElementCC3D("DecayConstant",{},0.00001)
                
                diffData.addComment("Additional options are:")                    
                # diffData.addComment("in XML <DoNotDiffuseTo>LIST CELL TYPES</DoNotDiffuseTo> or Python xxx.ElementCC3D(\"DoNotDiffuseTo\",{},LIST CELL TYPES)")
                # diffData.addComment("in XML <DoNotDecayIn>LIST CELL TYPES</DoNotDecayIn> or Python xxx.ElementCC3D(\"DoNotDecayIn\",{},LIST CELL TYPES)")                    
                donotdiffuseElem=diffData.ElementCC3D("DoNotDiffuseTo",{},"LIST YOUR CELL TYPES HERE")
                donotdiffuseElem.commentOutElement()                    
                donotdecayinElem=diffData.ElementCC3D("DoNotDecayIn",{},"LIST YOUR CELL TYPES HERE")
                donotdecayinElem.commentOutElement()
                concEqnElem=diffData.ElementCC3D("InitialConcentrationExpression",{},"x*y")
                concEqnElem.commentOutElement()
                
                concFieldNameElem=diffData.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
                concFieldNameElem.commentOutElement()
                
                diffData.ElementCC3D("DeltaX",{},1.0)
                diffData.ElementCC3D("DeltaT",{},1.0)
                
        
    # @GenerateDecorator('Steppable',['Type','SteadyStateDiffusionSolver'])                  
    def generateSteadyStateDiffusionSolver(self,*args,**kwds):
    
        # cellTypeData=self.cellTypeData
        # mElement=self.mElement
        # gpd=self.gpd  
        
        try:
            irElement=kwds['insert_root_element'] 
        except LookupError,e:
            irElement=None
        
        
        # existing root element 
        try:
            rElement=kwds['root_element'] 
        except LookupError,e:
            rElement=None
        
        try:
            generalPropertiesData=kwds['generalPropertiesData']
        except LookupError,e:
            generalPropertiesData={}
    
        gpd=generalPropertiesData 
        
        sim3DFlag=self.checkIfSim3D(gpd)
        solverName='SteadyStateDiffusionSolver'
        if not sim3DFlag:
            solverName+='2D'
        
        # mElement is module element - either steppable of plugin element 
        if irElement is None:    
        
            mElement=ElementCC3D("Steppable",{"Type":solverName})       
            
        else:
            irElement.addComment("newline")
            
            mElement=irElement.ElementCC3D("Steppable",{"Type":solverName})
        
        try:
            pdeFieldData=kwds['pdeFieldData']
        except LookupError,e:
            pdeFieldData={}
            
        try:
            cellTypeData=kwds['data']
        except LookupError,e:
            cellTypeData=None
    
        mElement.addComment("newline")
        mElement.addComment("Specification of PDE solvers")
    
                
        for fieldName,solver in pdeFieldData.iteritems():

            if solver=='SteadyStateDiffusionSolver':

                diffFieldElem=mElement.ElementCC3D("DiffusionField",{"Name":fieldName})              
                
                diffData=diffFieldElem.ElementCC3D("DiffusionData")                    
                diffData.ElementCC3D("FieldName",{},fieldName)
                diffData.ElementCC3D("DiffusionConstant",{},1.0)
                diffData.ElementCC3D("DecayConstant",{},0.00001)
                concEqnElem=diffData.ElementCC3D("InitialConcentrationExpression",{},"x*y")
                concEqnElem.commentOutElement()                   
                concFieldNameElem=diffData.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
                concFieldNameElem.commentOutElement()
                
                secrData=diffFieldElem.ElementCC3D("SecretionData")                    
                secrData.addComment('Secretion has to be defined inside SteadyStateDissufion solver - Secretion Plugin doe s not work with this solver.')
                secrData.addComment('newline')
                secrData.addComment('Uniform secretion Definition')
                secrData.ElementCC3D("Secretion",{"Type":'CELL TYPE 1'},0.1)
                secrData.ElementCC3D("Secretion",{"Type":'CELL TYPE 2'},0.2)
                                
                #Boiundary Conditions
                bcData=diffFieldElem.ElementCC3D("BoundaryConditions")                    
                planeXElem=bcData.ElementCC3D("Plane",{'Axis':'X'})                     
                planeXElem.ElementCC3D("ConstantValue",{'PlanePosition':'Min','Value':10.0}) 
                planeXElem.ElementCC3D("ConstantValue",{'PlanePosition':'Max','Value':5.0}) 
                planeXElem.addComment("Other options are (examples):")
                periodicXElem=planeXElem.ElementCC3D("Periodic") 
                periodicXElem.commentOutElement()                    
                cdElem=planeXElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                cdElem.commentOutElement()                    
                
                planeYElem=bcData.ElementCC3D("Plane",{'Axis':'Y'}) 
                planeYElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                planeYElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Max','Value':5.0}) 
                planeYElem.addComment("Other options are (examples):")
                periodicYElem=planeYElem.ElementCC3D("Periodic") 
                periodicYElem.commentOutElement()                    
                cvElem=planeYElem.ElementCC3D('ConstantValue',{'PlanePosition':'Min','Value':10.0}) 
                cvElem.commentOutElement()                    
                
                if sim3DFlag:                    
                    planeZElem=bcData.ElementCC3D("Plane",{'Axis':'Z'}) 
                    planeZElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Min','Value':10.0}) 
                    planeZElem.ElementCC3D('ConstantDerivative',{'PlanePosition':'Max','Value':5.0}) 
                    planeZElem.addComment("Other options are (examples):")
                    periodicZElem=planeZElem.ElementCC3D("Periodic") 
                    periodicZElem.commentOutElement()                    
                    cvzElem=planeZElem.ElementCC3D('ConstantValue',{'PlanePosition':'Min','Value':10.0}) 
                    cvzElem.commentOutElement()                    
                
                    
        return mElement 
        
            
        
    # def generatePDESolverCaller(self):
        # self.cc3d.addComment("newline")
        # self.cc3d.addComment("Module allowing multiple calls of the PDE solver. By default number of extra calls is set to 0. ")
        # self.cc3d.addComment("Change these settings to desired values after consulting CC3D manual on how to work with large diffusion constants (>0.16 in 3D, >0.25 in 2D with DeltaX=1.0 and DeltaT=1.0)")
        
        # # first translte chem field table into dictionary sorted by solver type
        # solverDict={}
        # for fieldTuple in self.chemFieldsTable:
            # try:
                # solverDict[fieldTuple[1]].append(fieldTuple[0])
            # except LookupError:
                # solverDict[fieldTuple[1]]=[fieldTuple[0]]
          
        # pcElem=self.cc3d.ElementCC3D("Plugin",{"Name":"PDESolverCaller"})
        
        # for solver in solverDict.keys():
            # pcElem.ElementCC3D("CallPDE",{"PDESolverName":solver,"ExtraTimesPerMC":0})
            
        
        
        
        
    def saveCC3DXML(self,_fileName):
        
        self.cc3d.CC3DXMLElement.saveXML(_fileName) 
        # self.cc3d.CC3DXMLElement.saveXMLInPython(str(self.fileName+".py")) 
        print "SAVING XML = ",_fileName
        # print "SAVING XML in Python= ",self.fileName+".py"