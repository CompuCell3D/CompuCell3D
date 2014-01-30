from XMLUtils import ElementCC3D
import os.path

class CC3DXMLGenerator:
    def __init__(self,simulationDir,simulationName):
        self.simulationDir=simulationDir
        self.simulationName=simulationName
        
        
        self.fileName=os.path.join(str(self.simulationDir),str(self.simulationName)+".xml")
        self.generalPropertiesDict={}
        
        self.cellTypeTable=[["Medium",False]]
        self.cc3d=ElementCC3D("CompuCell3D",{"version":"3.6.0"})
        self.afMolecules=[]
        self.afFormula='min(Molecule1,Molecule2)'
        self.cmcCadherins=[]
        self.chemFieldsTable=[]
        self.secretionTable={}
        self.chemotaxisTable={}
        
    def setGeneralPropertiesDict(self,_dict):
        self.generalPropertiesDict=_dict
        
    def setChemFieldsTable(self,_table):
        self.chemFieldsTable=_table
        
    def setSecretionTable(self,_table):
        self.secretionTable=_table
        
    def setChemotaxisTable(self,_table):
        self.chemotaxisTable=_table
        
    def setCMCTable(self,_table):
        self.cmcCadherins=_table

    def setAFFormula(self,_formula):        
        self.afFormula=_formula
        
    def setAFTable(self,_table):
        self.afMolecules=_table
        
    def setCellTypeTable(self,_table):
        self.cellTypeTable=_table
        #generate typeId to typeTuple lookup dictionary
        
        self.idToTypeTupleDict={}
        typeCounter=0
        
        for typeTupple in self.cellTypeTable:
            self.idToTypeTupleDict[typeCounter]=typeTupple            
            typeCounter+=1
            
    def  checkIfSim3D(self):
        sim3DFlag=False
        gpd=self.generalPropertiesDict
        if gpd["Dim"][0]>1 and gpd["Dim"][1]>1 and gpd["Dim"][2]>1:
            sim3DFlag=True
            
        return sim3DFlag
     
            
    def generatePottsSection(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Basic properties of CPM (GGH) algorithm")        
        
        potts=self.cc3d.ElementCC3D("Potts")
        gpd=self.generalPropertiesDict
        potts.ElementCC3D("Dimensions",{"x":gpd["Dim"][0],"y":gpd["Dim"][1],"z":gpd["Dim"][2]})
        potts.ElementCC3D("Steps",{},gpd["MCS"])
        potts.ElementCC3D("Temperature",{},gpd["MembraneFluctuations"])
        potts.ElementCC3D("NeighborOrder",{},gpd["NeighborOrder"])   
        if gpd["LatticeType"] != "Square":
            potts.ElementCC3D("LatticeType",{},gpd["LatticeType"])   
        
        
    def generateCellTypePlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Listing all cell types in the simulation")        
    
        cellTypePluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"CellType"})
        for id, typeTuple in self.idToTypeTupleDict.iteritems():
            dict={}
            dict["TypeName"]=typeTuple[0]

            dict["TypeId"]=str(id)
            if typeTuple[1]:
                dict["Freeze"]=""
            cellTypePluginElement.ElementCC3D("CellType", dict)

    def generateContactPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies")        
    
        contactPluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Contact"})
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            for id2 in range(id1,maxId+1):            
                dict={"Type1":self.idToTypeTupleDict[id1][0],"Type2":self.idToTypeTupleDict[id2][0]}            
                contactPluginElement.ElementCC3D("Energy",dict,10)
                
        contactPluginElement.ElementCC3D("NeighborOrder",{},2)        
            
            
    def generateCompartmentPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies in the presence of compartmental cells")
        self.cc3d.addComment("Please note that this is obsolete way of handling compartmets")        
        self.cc3d.addComment("Please consider using ContactInternal contact to specify adhesions bewtween members of same cluster")
        
        compratmentPluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ContactCompartment"})
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            for id2 in range(id1,maxId+1):            
                dict={"Type1":self.idToTypeTupleDict[id1][0],"Type2":self.idToTypeTupleDict[id2][0]}            
                compratmentPluginElement.ElementCC3D("Energy",dict,10)

        for id1 in range(0,maxId+1):
            for id2 in range(id1,maxId+1):            
                dict={"Type1":self.idToTypeTupleDict[id1][0],"Type2":self.idToTypeTupleDict[id2][0]}            
                compratmentPluginElement.ElementCC3D("InternalEnergy",dict,5)
                
        compratmentPluginElement.ElementCC3D("NeighborOrder",{},2)        

    def generateContactInternalPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies in the presence of compartmental cells")
        self.cc3d.addComment("Typically it is used in conjunction with \"reguar\" adhesion plugins e.g. Contact, AdhesionFlex etc...")
        
        contactInternalPluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ContactInternal"})
        maxId=max(self.idToTypeTupleDict.keys())

        for id1 in range(1,maxId+1):
            for id2 in range(id1,maxId+1):            
                dict={"Type1":self.idToTypeTupleDict[id1][0],"Type2":self.idToTypeTupleDict[id2][0]}            
                contactInternalPluginElement.ElementCC3D("Energy",dict,5)
                
        contactInternalPluginElement.ElementCC3D("NeighborOrder",{},2)        
        
    def generateFocalPointPlasticityPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of focal point junctions")
        self.cc3d.addComment("We separetely specify links between members of same cluster - InternalParameters and members of different clusters Parameters. When not using compartmental  cells comment out InternalParameters specification")
        
        fpppElement=self.cc3d.ElementCC3D("Plugin",{"Name":"FocalPointPlasticity"})
        fpppElement.addComment("To modify FPP links individually for each cell pair uncomment line below")        
        localFpppElement=fpppElement.ElementCC3D("Local")
        localFpppElement.commentOutElement()        
        fpppElement.addComment("Note that even though you may manipulate lambdaDistance, targetDistance and maxDistance using Python you still need to set activation energy from XML level")
        fpppElement.addComment("See CC3D manual for details on FPP plugin ")
        
        
        maxId=max(self.idToTypeTupleDict.keys())

        for id1 in range(1,maxId+1):
            for id2 in range(id1,maxId+1):            
                fpppElement.addComment("newline")
                dict={"Type1":self.idToTypeTupleDict[id1][0],"Type2":self.idToTypeTupleDict[id2][0]}                            
                paramElement=fpppElement.ElementCC3D("Parameters",dict)
                paramElement.ElementCC3D("Lambda",{},10)
                paramElement.ElementCC3D("ActivationEnergy",{},-50)
                paramElement.ElementCC3D("TargetDistance",{},7)
                paramElement.ElementCC3D("MaxDistance",{},20)
                paramElement.ElementCC3D("MaxNumberOfJunctions",{"NeighborOrder":1},1)
                
                
        for id1 in range(1,maxId+1):
            for id2 in range(id1,maxId+1):            
                fpppElement.addComment("newline")                
                dict={"Type1":self.idToTypeTupleDict[id1][0],"Type2":self.idToTypeTupleDict[id2][0]}                 
                paramElement=fpppElement.ElementCC3D("InternalParameters",dict)
                paramElement.ElementCC3D("Lambda",{},10)
                paramElement.ElementCC3D("ActivationEnergy",{},-50)
                paramElement.ElementCC3D("TargetDistance",{},7)
                paramElement.ElementCC3D("MaxDistance",{},20)
                paramElement.ElementCC3D("MaxNumberOfJunctions",{"NeighborOrder":1},1)
                
        fpppElement.addComment("newline")        
        fpppElement.ElementCC3D("NeighborOrder",{},1)
        
    def generateElasticityPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Elastic constraints between Center of mass of cells. Need to be accompanied by ElasticityTracker plugin to work. Only cells in contact at MCS=0 will be affected by the constraint")        
        
        self.cc3d.addComment("ElasticityTracker keeps track of cell neighbors which are participating in the elasticity constraint calculations")        
        etpElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ElasticityTracker"})        
        etpElement.addComment("Comment out cell types which should be unaffected by the constraint")
        
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            if self.idToTypeTupleDict[id1][0]=="Medium":
                continue            
            etpElement.ElementCC3D("IncludeType",{},self.idToTypeTupleDict[id1][0]) 
            
        self.cc3d.addComment("newline")            
        self.cc3d.addComment("This plugin calculates elastic constraints between cells Center of Mass")            
        epElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Elasticity"})        
        epElement.addComment("To enable specification of elastic links individually for each link uncomment line below ")
        localElem=epElement.ElementCC3D("Local")
        localElem.commentOutElement()        
        epElement.addComment("See CC3D manual for details")

        epElement.ElementCC3D("LambdaElasticity",{},200) 
        epElement.ElementCC3D("TargetLengthElasticity",{},6) 
            
    
    def generateContactLocalProductPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies as a function of cadherin concentration at cell membranes")
        self.cc3d.addComment("Adhesion energy is a function of two cells in ocntact. the functional form is specified by the user")
        self.cc3d.addComment("Please consider using more flexible version of this plugin - AdhesionFlex")
        
        clpPluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ContactLocalProduct"})
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            for id2 in range(id1,maxId+1):            
                dict={"Type1":self.idToTypeTupleDict[id1][0],"Type2":self.idToTypeTupleDict[id2][0]}            
                clpPluginElement.ElementCC3D("ContactSpecificity",dict,-1)
                
        clpPluginElement.ElementCC3D("ContactFunctionType",{},"linear")
        
        clpPluginElement.ElementCC3D("EnergyOffset",{},0.0)                        
        clpPluginElement.ElementCC3D("NeighborOrder",{},2)        
        
    def generateContactMultiCadPlugin(self):
        cmcPluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ContactMultiCad"})
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            for id2 in range(id1,maxId+1):            
                dict={"Type1":self.idToTypeTupleDict[id1][0],"Type2":self.idToTypeTupleDict[id2][0]}            
                cmcPluginElement.ElementCC3D("Energy",dict,10)
                
        specificityCadherinElement=cmcPluginElement.ElementCC3D("SpecificityCadherin")                        
        
        cadMax=len(self.cmcCadherins)
        for cad1 in range(cadMax):
            for cad2 in range(cad1,cadMax):
                
                dict={"Cadherin1":self.cmcCadherins[cad1],"Cadherin2":self.cmcCadherins[cad2]}            
                specificityCadherinElement.ElementCC3D("Specificity",dict,-1)
                
        cmcPluginElement.ElementCC3D("ContactFunctionType",{},"linear")
        cmcPluginElement.ElementCC3D("EnergyOffset",{},0.0)                        
        cmcPluginElement.ElementCC3D("NeighborOrder",{},2)        
        
    def generateAdhesionFlexPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of adhesion energies as a function of cadherin concentration at cell membranes")
        self.cc3d.addComment("Adhesion energy is a function of two cells in ocntact. the functional form is specified by the user")
    
        afPluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"AdhesionFlex"})
        maxId=max(self.idToTypeTupleDict.keys())
        # listing adhesion molecules
        for molecule in self.afMolecules:
            afPluginElement.ElementCC3D("AdhesionMolecule",{"Molecule":molecule})
            
        for id1 in range(0,maxId+1):
            for molecule in self.afMolecules:
                afPluginElement.ElementCC3D("AdhesionMoleculeDensity",{"CellType":self.idToTypeTupleDict[id1][0],  "Molecule":molecule,"Density":0.8})
                
        formulaElement=afPluginElement.ElementCC3D("BindingFormula",{"Name":"Binary"})    
        formulaElement.ElementCC3D("Formula",{},self.afFormula)
        variablesElement=formulaElement.ElementCC3D("Variables")
        aimElement=variablesElement.ElementCC3D("AdhesionInteractionMatrix")
        
        moleculeMax=len(self.afMolecules)        
        for molecule1 in range(moleculeMax):
            for molecule2 in range(molecule1,moleculeMax):
                dict={"Molecule1":self.afMolecules[molecule1],"Molecule2":self.afMolecules[molecule2]}
                aimElement.ElementCC3D("BindingParameter",dict,-1.0)
            
        afPluginElement.ElementCC3D("NeighborOrder",{},2)        

    def generateVolumeFlexPlugin(self):
    
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Constraint on cell volume. Each cell type has different constraint.")
        self.cc3d.addComment("For more flexible specification of the constraint (done in Python) please use VolumeLocalFlex plugin")
    
        vfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Volume"})
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            if self.idToTypeTupleDict[id1][0]=="Medium":
                continue
            dict={"CellType":self.idToTypeTupleDict[id1][0],"TargetVolume":25,"LambdaVolume":2.0}    
            vfElement.ElementCC3D("VolumeEnergyParameters",dict)    
        
    def generateSurfaceFlexPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Constraint on cell surface. Each cell type has different constraint.")
        self.cc3d.addComment("For more flexible specification of the constraint (done in Python) please use SurfaceLocalFlex plugin")
    
        sfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Surface"})
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            if self.idToTypeTupleDict[id1][0]=="Medium":
                continue
            dict={"CellType":self.idToTypeTupleDict[id1][0],"TargetSurface":20,"LambdaSurface":0.5}    
            sfElement.ElementCC3D("SurfaceEnergyParameters",dict) 
            
    def generateVolumeLocalFlexPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Constraint on cell volume. Each cell has different constraint - constraints have to be initialized and managed in Python")
            
        vfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Volume"})
        
              
        
    def generateSurfaceLocalFlexPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Constraint on cell surface. Each cell has different constraint - constraints have to be initialized and managed in Python")
    
        sfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Surface"})

    def generateExternalPotentialPlugin(self):
    
        self.cc3d.addComment("newline")
        self.cc3d.addComment("External force applied to cell. Each cell type has different force.")
        self.cc3d.addComment("For more flexible specification of the constraint (done in Python) please use ExternalPotentialLocalFlex plugin")
    
        epElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ExternalPotential"})
        

        epElement.addComment("Algorithm options are: PixelBased, CenterOfMassBased")
        epElement.ElementCC3D("Algorithm",{},"PixelBased")    
        
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            if self.idToTypeTupleDict[id1][0]=="Medium":
                continue
            dict={"CellType":self.idToTypeTupleDict[id1][0],"x":-0.5,"y":0.0,"z":0.0}    
            epElement.ElementCC3D("ExternalPotentialParameters",dict)    
        
    def generateExternalPotentialLocalFlexPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("External force applied to cell. Each cell has different force and force components have to be managed in Python.")
        self.cc3d.addComment("e.g. cell.lambdaVecX=0.5; cell.lambdaVecY=0.1 ; cell.lambdaVecZ=0.3;")
        
        epElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ExternalPotential"})
        epElement.addComment("Algorithm options are: PixelBased, CenterOfMassBased")
        epElement.ElementCC3D("Algorithm",{},"PixelBased")    
            
    def generateConnectivityGlobalPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Connectivity constraint applied to each cell. Energy penalty specifies how strong the constraint is. Penalty is specified for each type ")
        self.cc3d.addComment("This constraint works in 2D and 3D on all type of lattices. It might be slowdown your simulation. For faster option - 2D and square lattice you may use Connectivity or ConnectivityLocalFlex")
        self.cc3d.addComment("To speed up simulation comment out unnecessary constraints for types which don't need the constraint")
        cpElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ConnectivityGlobal"})
                
        cpElement.addComment("By default we will always precheck connectivity BUT in simulations in which there is no risk of having unfragmented cell one can add this flag to speed up computations")
        cpElement.addComment("To turn off precheck uncomment line below")
        precheckElem=cpElement.ElementCC3D("DoNotPrecheckConnectivity")
        precheckElem.commentOutElement()
        
        
        
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            if self.idToTypeTupleDict[id1][0]=="Medium":
                continue
            dict={"Type":self.idToTypeTupleDict[id1][0]}    
            cpElement.ElementCC3D("Penalty",dict,1000000)
  
    def generateConnectivityGlobalByIdPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Connectivity constraint applied to each cell. Energy penalty specifies how strong the constraint is. Penalty is specified for each cell type individually ")
        self.cc3d.addComment("Use Python scripting to setup penalty (connectivity strength) for each cell")  
        self.cc3d.addComment("e.g. self.connectivityGlobalPlugin.setConnectivityStrength(cell,20000000) #cell, connectivity strength")        
        self.cc3d.addComment("This constraint works in 2D and 3D on all type of lattices. It might be slowdown your simulation. For faster option - 2D and square lattice you may use Connectivity or ConnectivityLocalFlex")
        self.cc3d.addComment("To speed up simulation comment out unnecessary constraints for types which don't need the constraint")
        cpElement=self.cc3d.ElementCC3D("Plugin",{"Name":"ConnectivityGlobal"})
  
    def generateConnectivityPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Connectivity constraint applied to each cell. Energy penalty specifies how strong the constraint is. Penalty is specified globally for each cell ")        
        self.cc3d.addComment("This constraint works in 2D and on square lattice only!")
        self.cc3d.addComment("For more flexible version of this plugin use ConnectivityLocalFlex where constraint penalty is specified for each cell individually using Python scripting using the following syntax")
        self.cc3d.addComment("self.connectivityLocalFlexPlugin.setConnectivityStrength(cell,10000000)" )
        
        cpElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Connectivity"})
        cpElement.ElementCC3D("Penalty",{},10000000)
        
    def generateLengthConstraintPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Applies elongation constraint to each cell. Users specify target length of major axis -TargetLength (in 3D additionally, target length of minor axis - MinorTargetLength) and a strength of the constraint -LambdaLength. Parameters are specified for each cell type")        
        self.cc3d.addComment("IMPORTANT: To prevent cell fragmentation for large elongations you need to also use connectivity constraint")
        self.cc3d.addComment("LengthConstrainLocalFlex allows constrain specification for each cell individually but currently works only in 2D")                                
        self.cc3d.addComment("Comment out the constrains for cell types which dont need them")                
        
        lcpElement=self.cc3d.ElementCC3D("Plugin",{"Name":"LengthConstraint"})
                
        sim3DFlag=self.checkIfSim3D()
            
        
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            if self.idToTypeTupleDict[id1][0]=="Medium":
                continue
            dict={"CellType":self.idToTypeTupleDict[id1][0],"TargetLength":20,"LambdaLength":100} 
            if sim3DFlag:
                dict["MinorTargetLength"]=5
            lcpElement.ElementCC3D("LengthEnergyParameters",dict)
        
    def generateLengthConstraintLocalFlexPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Applies elongation constraint to each cell. Users specify the length major axis -TargetLength and a strength of the constraint -LambdaLength. Parameters are specified for each cell individually")        
        self.cc3d.addComment("IMPORTANT: To prevent cell fragmentation for large elongations you need to also use connectivity constraint")
        self.cc3d.addComment("This plugin currently works only in 2D. Use the following Python syntax to set/modify length constraint:")               
        self.cc3d.addComment("self.lengthConstraintFlexPlugin.setLengthConstraintData(cell,20,30)  # cell , lambdaLength, targetLength  ")      
        lcpElement=self.cc3d.ElementCC3D("Plugin",{"Name":"LengthConstraintLocalFlex"})
        
        
    def generateCenterOfMassPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module tracking center of mass of each cell")        
    
        sfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"CenterOfMass"})
        
    def generateNeighborTrackerPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module tracking neighboring cells of each cell")        
    
        sfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"NeighborTracker"})
        
    def generateMomentOfInertiaPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module tracking moment of inertia of each cell")        
    
        sfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"MomentOfInertia"})
        
    def generatePixelTrackerPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module pixels of each cell")        
    
        sfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"PixelTracker"})        
        
    def generateBoundaryPixelTrackerPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module boundary pixels of each cell")        
    
        sfElement=self.cc3d.ElementCC3D("Plugin",{"Name":"BoundaryPixelTracker"})
        sfElement.ElementCC3D("NeighborOrder",{},1)
        
    def generateSecretionPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of secretion properties of select cell types.")
        self.cc3d.addComment("Specification of secretion properties of individual cells can be done in Python")        
    
        secretionPluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Secretion"})
        for chemFieldName, secrDictList in self.secretionTable.iteritems():
            secrFieldElement=secretionPluginElement.ElementCC3D("Field",{"Name":chemFieldName})
            for secrDict in secrDictList:
                
                rate=secrDict["Rate"]
                # chemotaxTowards=chemDict["ChemotaxTowards"]
                # satCoef=chemDict["SatCoef"]
                # chemotaxisType=chemDict["ChemotaxisType"]
                
                attributeDict={}
                attributeDict["Type"]=secrDict["CellType"]
                if secrDict["SecretionType"]=='uniform':
                    secrFieldElement.ElementCC3D("Secretion",attributeDict,rate)
                elif secrDict["SecretionType"]=='on contact':
                    attributeDict["SecreteOnContactWith"]=secrDict["OnContactWith"]
                    secrFieldElement.ElementCC3D("SecretionOnContact",attributeDict,rate)
                elif secrDict["SecretionType"]=='constant concentration':      
                    secrFieldElement.ElementCC3D("SecretionOnContact",attributeDict,rate)
                
        
    def generateChemotaxisPlugin(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of chemotaxis properties of select cell types.")
    
        # first translte chem field table into dictionary sorted by FieldName
        solverDict={}
        for fieldTuple in self.chemFieldsTable:
            solverDict[fieldTuple[0]]=fieldTuple[1]
    
        chemotaxisPluginElement=self.cc3d.ElementCC3D("Plugin",{"Name":"Chemotaxis"})
        for chemFieldName, chemDictList in self.chemotaxisTable.iteritems():
            chemFieldElement=chemotaxisPluginElement.ElementCC3D("ChemicalField",{"Source":solverDict[chemFieldName],"Name":chemFieldName})
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
            
            
        
    def generatePDESolverCaller(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module allowing multiple calls of the PDE solver. By default number of extra calls is set to 0. ")
        self.cc3d.addComment("Change these settings to desired values after consulting CC3D manual on how to work with large diffusion constants (>0.16 in 3D, >0.25 in 2D with DeltaX=1.0 and DeltaT=1.0)")
        
        # first translte chem field table into dictionary sorted by solver type
        solverDict={}
        for fieldTuple in self.chemFieldsTable:
            try:
                solverDict[fieldTuple[1]].append(fieldTuple[0])
            except LookupError:
                solverDict[fieldTuple[1]]=[fieldTuple[0]]
          
        pcElem=self.cc3d.ElementCC3D("Plugin",{"Name":"PDESolverCaller"})
        
        for solver in solverDict.keys():
            pcElem.ElementCC3D("CallPDE",{"PDESolverName":solver,"ExtraTimesPerMC":0})
            
        
    def generatePDESolvers(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Specification of PDE solvers")
    
        # first translte chem field table into dictionary sorted by solver type
        solverDict={}
        for fieldTuple in self.chemFieldsTable:
            try:
                solverDict[fieldTuple[1]].append(fieldTuple[0])
            except LookupError:
                solverDict[fieldTuple[1]]=[fieldTuple[0]]
                
        for solver,fieldNames in solverDict.iteritems():
            if solver=='KernelDiffusionSolver':
                kdiffSolverElem=self.cc3d.ElementCC3D("Steppable",{"Type":"KernelDiffusionSolver"})
                
                for fieldName in fieldNames:
                    diffFieldElem=kdiffSolverElem.ElementCC3D("DiffusionField",{"Name":fieldName})
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
                    # add commented out concentration field specification file
            elif solver in ('FlexibleDiffusionSolverFE','FastDiffusionSolver2DFE'):
                diffSolverElem=self.cc3d.ElementCC3D("Steppable",{"Type":solver})
                
                for fieldName in fieldNames:
                    diffFieldElem=diffSolverElem.ElementCC3D("DiffusionField",{"Name":fieldName})
                    
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
                    diffData.addComment("To run solver for large diffusion constants you typically call solver multiple times - ExtraTimesPerMCS to specify additional calls to the solver in each MCS ")                    
                    diffData.addComment("IMPORTANT: make sure not to mix this setting with the PDESolverCaller module! See manual for more information")                    
                    extraTimesPerMCSElem=diffData.ElementCC3D("ExtraTimesPerMCS",{},0)
                    extraTimesPerMCSElem.commentOutElement()
                    
                    diffData.ElementCC3D("DeltaX",{},1.0)
                    diffData.ElementCC3D("DeltaT",{},1.0)
                    
            elif solver in ('SteadyStateDiffusionSolver'):
                solverName='SteadyStateDiffusionSolver2D'
                sim3DFlag=self.checkIfSim3D()
                if sim3DFlag:
                    solverName='SteadyStateDiffusionSolver'
            
                diffSolverElem=self.cc3d.ElementCC3D("Steppable",{"Type":solverName})
                
                for fieldName in fieldNames:
                    diffFieldElem=diffSolverElem.ElementCC3D("DiffusionField",{"Name":fieldName})
                    
                    diffData=diffFieldElem.ElementCC3D("DiffusionData")                    
                    diffData.ElementCC3D("FieldName",{},fieldName)
                    diffData.ElementCC3D("DiffusionConstant",{},1.0)
                    diffData.ElementCC3D("DecayConstant",{},0.00001)
                    concEqnElem=diffData.ElementCC3D("InitialConcentrationExpression",{},"x*y")
                    concEqnElem.commentOutElement()                   
                    concFieldNameElem=diffData.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
                    concFieldNameElem.commentOutElement()
                    
                    
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
                        
                    
            else:
                return
                    
                    # add commented out concentration field specification file
                    
    def generateBoxWatcherSteppable(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Module tracing boundaries of the minimal box enclosing all the cells. May speed up calculations. May have no effect for parallel version")
        
        bwElem=self.cc3d.ElementCC3D("Steppable",{"Type":"BoxWatcher"})
        bwElem.ElementCC3D("XMargin",{},5)
        bwElem.ElementCC3D("YMargin",{},5)
        bwElem.ElementCC3D("ZMargin",{},5)
        
    def generateUniformInitializerSteppable(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Initial layout of cells in the form of rectangular slab")
    
        uiElement=self.cc3d.ElementCC3D("Steppable",{"Type":"UniformInitializer"})
        region=uiElement.ElementCC3D("Region")
        
        gpd=self.generalPropertiesDict
        
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
        
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            if self.idToTypeTupleDict[id1][0]=="Medium":
                continue
                
            typesString+=self.idToTypeTupleDict[id1][0]
            if id1<maxId:
                typesString+=","
            
        region.ElementCC3D("Types",{},typesString)
        
    def generateBlobInitializerSteppable(self):
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Initial layout of cells in the form of spherical (circular in 2D) blob")
    
        uiElement=self.cc3d.ElementCC3D("Steppable",{"Type":"BlobInitializer"})
        region=uiElement.ElementCC3D("Region")
        
        gpd=self.generalPropertiesDict
        
        xCenter=int(gpd["Dim"][0]/2)
        
        yCenter=int(gpd["Dim"][1]/2)
            
        zCenter=int(gpd["Dim"][2]/2)
        maxDim=max([xCenter,yCenter,zCenter])
            
        region.ElementCC3D("Center",{"x":xCenter,"y":yCenter,"z":zCenter})        
        region.ElementCC3D("Radius",{},int(maxDim/2.5))
        region.ElementCC3D("Gap",{},0)
        region.ElementCC3D("Width",{},5)
        
        
        typesString=""
        
        maxId=max(self.idToTypeTupleDict.keys())
        for id1 in range(0,maxId+1):
            if self.idToTypeTupleDict[id1][0]=="Medium":
                continue
                
            typesString+=self.idToTypeTupleDict[id1][0]
            if id1<maxId:
                typesString+=","
            
        region.ElementCC3D("Types",{},typesString)
        
        
        
    def generatePIFFInitializerSteppable(self):  
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Initial layout of cells using PIFF file. Piff files can be generated using PIFGEnerator")
    
        uiElement=self.cc3d.ElementCC3D("Steppable",{"Type":"PIFInitializer"})
        gpd=self.generalPropertiesDict
        if gpd["Initializer"][0]=="piff":
            uiElement.ElementCC3D("PIFName",{},gpd["Initializer"][1])
        else:
            uiElement.ElementCC3D("PIFName",{},"PLEASE_PUT_PROPER_FILE_NAME_HERE")
        
    def generatePIFFDumperSteppable(self):  
        self.cc3d.addComment("newline")
        self.cc3d.addComment("Periodically stores cell layout configuration in a piff format")
        
        pifdElement=self.cc3d.ElementCC3D("Steppable",{"Type":"PIFDumper","Frequency":100})
        pifdElement.ElementCC3D("PIFName",{},self.simulationName)
        pifdElement.ElementCC3D("PIFFileExtension",{},"piff")
        
        
        
    def saveCC3DXML(self):
        
        self.cc3d.CC3DXMLElement.saveXML(str(self.fileName)) 
        # self.cc3d.CC3DXMLElement.saveXMLInPython(str(self.fileName+".py")) 
        print "SAVING XML = ",self.fileName
        # print "SAVING XML in Python= ",self.fileName+".py"