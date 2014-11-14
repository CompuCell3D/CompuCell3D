import Configuration
from enums import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtXml import *
import time

class CATabView(object):
    
    def __init__(self,_masterTabView):
        self.masterTabView = _masterTabView
        self.mtv = _masterTabView
        self.fieldDim = None
        self.mysim = None
        
        
        
        self.model2D=None
        self.model3D=None
        self.view2D=None
        self.view3D=None
        
    def produceModelSpecificGraphicsModelsAndViews(self):
        from Graphics.CAMVCDrawModel2D import CAMVCDrawModel2D
        from Graphics.CAMVCDrawModel3D import CAMVCDrawModel3D
        from Graphics.CAMVCDrawView2D import CAMVCDrawView2D
        from Graphics.CAMVCDrawView3D import CAMVCDrawView3D
        
        return CAMVCDrawModel2D() , CAMVCDrawModel3D() , CAMVCDrawView2D() , CAMVCDrawView3D() # model2D model3D view2D view3D
        
    def initializeSimulationViewWidgetRegular(self):
        print 'sim=',self.mtv.simulation
        caManager = self.mtv.simulation.sim()
        print 'caManager=',caManager
        if caManager:
            self.mtv.fieldDim = caManager.getCellFieldS().getDim()
            # any references to simulator shuold be weak to avoid possible memory leaks - when not using weak references one has to be super careful to set to Non all references to sim to break any reference cycles
            # weakref is much easier to handle and code is cleaner
            from weakref import ref
            self.mtv.mysim = ref(caManager)

        
        
        simObj=self.mtv.mysim() # extracting object from weakref object wrapper
        if not simObj:
            sys.exit()
            return        

        self.mtv.basicSimulationData.fieldDim = self.mtv.fieldDim
        self.mtv.basicSimulationData.sim = simObj
        # # # self.basicSimulationData.numberOfSteps = simObj.getNumSteps() # we will use number of steps later
        self.mtv.basicSimulationData.numberOfSteps = simObj.getNumSteps() # hard-coded for now 
        
        print 'self.mtv.fieldExtractor=',self.mtv.fieldExtractor
        print 'dir(self.mtv.fieldExtractor)=',dir(self.mtv.fieldExtractor)
        print 'self=',self
        print 'self.mtv=',self.mtv
        print 'self.mtv.fieldExtractor=',self.mtv.fieldExtractor
        print 'self.mtv.fieldExtractor.init=',self.mtv.fieldExtractor.init
        print 'simObj=',simObj
        print 'self.mtv.fieldExtractor.caManager=',self.mtv.fieldExtractor.caManager
        self.mtv.fieldExtractor.pointOrder('XY')
        self.mtv.fieldExtractor.caManager=simObj
        
        # self.mtv.fieldExtractor.init(simObj)
        
        
        
        
        # self.mtv.fieldExtractor.init(self.mtv.mysim())
        
        # # # import time
        # # # time.sleep(3)
        # # # print 'AFTER self.fieldStorage.allocateCellField'
        # # # time.sleep(5)
                
        
        self.mtv.screenshotNumberOfDigits = len(str(self.mtv.basicSimulationData.numberOfSteps))
        self.mtv.prepareSimulationView()
        
        print 'self.simulationIsStepping=',self.mtv.simulationIsStepping
        if self.mtv.simulationIsStepping:
            # print "BEFORE STEPPING PAUSE REGULAR SIMULATION"
            self.mtv.pauseSim()          
            
    def drawFieldRegular(self):
        if not self.mtv.simulationIsRunning:
            return    
            
        if self.mtv.newDrawingUserRequest:
            self.mtv.newDrawingUserRequest = False
            if self.mtv.pauseAct.isEnabled():
                self.mtv.pauseSim()
        self.mtv.simulation.drawMutex.lock()
        
        # self.__step = self.mtv.simulation.getCurrentStep()
        self.mtv.setCurrentStep(self.mtv.simulation.getCurrentStep())
        print ' CATABVIEW drawFieldRegularCA(): __step=',self.mtv.getCurrentStep()

        # self.simulation.drawMutex.unlock()
        # return
      
        
        if self.mtv.mysim:
#            print MODULENAME,'  drawFieldRegular(): in self.mysim block; windowDict.keys=',self.graphicsWindowDict.keys()
            for windowName in self.mtv.graphicsWindowDict.keys():
                graphicsFrame = self.mtv.windowDict[windowName]
#                print MODULENAME,"drawFieldRegular():   windowName, graphicsFrame=",windowName, graphicsFrame
                                
                
                #rwh: error if we try to invoke switchdim earlier    
                (currentPlane, currentPlanePos) = graphicsFrame.getPlane()
                print '(currentPlane, currentPlanePos)=',(currentPlane, currentPlanePos)
                # print 'BEFORE graphicsFrame.drawFieldLocal'    
                # time.sleep(5)
                
                # this is main drawing function
                graphicsFrame.drawFieldLocal(self.mtv.basicSimulationData)
                
                # print 'AFTER graphicsFrame.drawFieldLocal'    
                # time.sleep(5)
                
                    
                self.mtv.updateStatusBar(self.mtv.getCurrentStep(), graphicsFrame.conMinMax())   # show MCS in lower-left GUI
        
        self.mtv.simulation.drawMutex.unlock()
            
    def updateSimulationProperties(self):       
        # print 'INSIDE updateSimulationProperties ',self.fieldDim

        caManagerObj=self.mtv.mysim()
        if not caManagerObj:return
        
        fieldDim = caManagerObj.getCellField().getDim()
        # # # fieldDim = self.simulation.sim.getPotts().getCellFieldG().getDim()
        
        if fieldDim.x==self.mtv.fieldDim.x and fieldDim.y==self.mtv.fieldDim.y and fieldDim.z==self.mtv.fieldDim.z:
            return False
            
        self.mtv.fieldDim= fieldDim   
        self.mtv.basicSimulationData.fieldDim = self.mtv.fieldDim
        self.mtv.basicSimulationData.sim = caManagerObj
        self.mtv.basicSimulationData.numberOfSteps = caManagerObj.getNumSteps()
        return True
        
    
    
    def showSimView(self, file):
        


        self.mtv.setupArea()
        

        
        isTest = False
        
        """      
        # For testing. Leave for a while
        if isTest:
            self.mainGraphicsWindow = QVTKRenderWidget(self)
            self.insertTab(0, self.mainGraphicsWindow, QIcon("player/icons/sim.png"), os.path.basename(str(self.__fileName)))
            self.setupArea()
        else:
        """
        
        # Create self.mainGraphicsWindow  
        # # # self.mainGraphicsWindow = self.mainGraphicsWindow # Graphics2D by default
        self.mtv.setCurrentStep(0)       
        

        
        self.mtv.showDisplayWidgets()



        caManagerObj=None
        if self.mtv.mysim:
            caManagerObj=self.mtv.mysim()
            # if not simObj:return

        
        self.mtv.setFieldType( ("Cell_Field", FIELD_TYPES[0]) )
        
        # self.__fieldType = ("FGF", FIELD_TYPES[1])
        
        # print MODULENAME,'  ------- showSimView \n\n'
        
        
        if self.mtv.basicSimulationData.sim:
            # # # cellField = simObj.getPotts().getCellFieldG()
            cellField = caManagerObj.getCellField()
            # self.simulation.graphicsWidget.fillCellFieldData(cellField,"xy",0)
            
            # print "        BEFORE DRAW FIELD(1) FROM showSimView()"
            # time.sleep(5)
            
            self.mtv._drawField()
            
            # print "        AFTER DRAW FIELD(1) FROM showSimView()"
            # time.sleep(5)
     
            
        
            # # Fields are available only after simulation is loaded
            self.mtv.setFieldTypes() 
        else:
            # print "        BEFORE DRAW FIELD(2) FROM showSimView()"
            # if not self.simulation.dimensionChange():
            
                

            self.mtv._drawField()
            
            self.mtv.setFieldTypesCML() 
            # print "        AFTER DRAW FIELD(2) FROM showSimView()"
        
#        import pdb; pdb.set_trace()

      
        
        
        Configuration.initFieldsParams(self.mtv.fieldTypes.keys())
        
        # # # self.__setCrossSection()
        basicSimulationData=self.mtv.basicSimulationData
        
        print 'self.basicSimulationData=',dir(basicSimulationData)
        print 'self.basicSimulationData.fieldDim=',basicSimulationData.fieldDim
        print 'self.basicSimulationData.numberOfSteps=',basicSimulationData.numberOfSteps
        print 'self.basicSimulationData.sim=',basicSimulationData.sim

        
        self.mtv.setInitialCrossSection(basicSimulationData)
        print '   AFTER setInitialCrossSection'    
        self.mtv.initGraphicsWidgetsFieldTypes()
        # self.closeTab.show()
        self.mtv.drawingAreaPrepared = True
#        self.mainGraphicsWindow.parentWidget.move(400,300)   # temporarily moves, but jumps back

        self.mtv.layoutGraphicsWindows()
        
    def initializeSimulationThreadAndStorageObjects(self):
             
        # # # self.__viewManagerType = "CA"
        from Simulation.SimulationThreadCA import SimulationThreadCA
        self.mtv.simulation = SimulationThreadCA(self.mtv)
        simulation = self.mtv.simulation
        # print 'AFTER CONSTRUCTING NEW SIMULATINO THREAD'
        
        import CAFieldUtils
        print 'this is CAPyUtils', dir(CAFieldUtils)
        self.mtv.fieldStorage = None
        self.mtv.fieldExtractor = CAFieldUtils.CAFieldExtractor()
        
        
        self.mtv.connect(simulation,SIGNAL("simulationInitialized(bool)"),self.mtv.initializeSimulationViewWidget)
        self.mtv.connect(simulation,SIGNAL("steppablesStarted(bool)"),self.mtv.runSteppablePostStartPlayerPrep)            
        self.mtv.connect(simulation,SIGNAL("simulationFinished(bool)"),self.mtv.handleSimulationFinished)
        self.mtv.connect(simulation,SIGNAL("completedStep(int)"),self.mtv.handleCompletedStep)
        self.mtv.connect(simulation,SIGNAL("finishRequest(bool)"),self.mtv.handleFinishRequest)   
        
        self.mtv.plotManager.initSignalAndSlots()  

        
    def setFieldTypes(self):        
        print 'sim=',self.mtv.simulation
        caManager = self.mtv.simulation.sim()
        print 'caManager=',caManager
        if caManager:
            self.mtv.fieldDim = caManager.getCellFieldS().getDim()
            # any references to simulator shuold be weak to avoid possible memory leaks - when not using weak references one has to be super careful to set to Non all references to sim to break any reference cycles
            # weakref is much easier to handle and code is cleaner
            from weakref import ref
            self.mtv.mysim = ref(caManager)

        
        
        simObj=self.mtv.mysim() # extracting object from weakref object wrapper
        if not simObj:
            sys.exit()
            return            

            
        self.mtv.fieldTypes["Cell_Field"] = FIELD_TYPES[0]  #"CellField" 
        
        
        # Add concentration fields How? I don't care how I got it at this time
        # print self.mysim.getPotts()
        concFieldNameVec = simObj.getConcentrationFieldNameVector()
        # print 'concFieldNameVec=',concFieldNameVec
        # print 'concFieldNameVec=',concFieldNameVec.size()
        # print 'simObj=',simObj
        # sys.exit()
        # # # concFieldNameVec = self.mysim.getConcentrationFieldNameVector()
        
        #putting concentration fields from simulator
        for fieldName in concFieldNameVec:
#            print MODULENAME,"setFieldTypes():  Got this conc field: ",fieldName
            self.mtv.fieldTypes[fieldName] = FIELD_TYPES[1]            
            
    def createOutputDirs(self):
        print 'CA createOutputDirs'
            
            