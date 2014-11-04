from collections import OrderedDict
class CASimulationPy(object):
    def __init__(self):
        self.caManager=None
        self.neighborOrder=2 # 2 for 2D,  3 for 3D
        self.globalCarryingCapacity = 1
        self.simthread =None
        self.cellField=None
        self.numSteps=0
        from PyDictAdder import DictAdder        
        self.dictAdder=DictAdder()
        
        self.moduleRegistry={} # stores modules in the registry
        self.probabilityFunctionRegistry={}    
        self.steppableRegistry = OrderedDict() # {steppablename:[list of steppables with a given name]}
        self.steppableBeforeMCSRegistry = OrderedDict() # {steppablename:[list of steppables with a given name]}
        
#         self.pySteppableRegistry = {}
#         self.pySteppableBeforeMCSRegistry = {}
    
        
    def registerProbabilityFunction(self,_probFunction):
        _probFunction.init(self.caManager)
        self.probabilityFunctionRegistry[_probFunction.toString()] = _probFunction
        
        self.caManager.registerProbabilityFunction(_probFunction)
        return _probFunction
    
    def registerProbabilityFunctionByName(self,_probFunctionName):
        
        try:
            exec("from "+_probFunctionName+" import "+_probFunctionName+"\n")
            probFunction=eval(_probFunctionName+"()")
            self.probabilityFunctionRegistry[probFunction.toString()]=probFunction                
            probFunction.init(self.caManager)
            self.caManager.registerProbabilityFunction(probFunction)
            return probFunction
        except ImportError:
            print 'COULD NOT IMPORT ',_probFunctionName        
            return None
    

    def __registerModule(self,_module,_registry):
        try: # not all modules will havbe init function. only C++ steppables have it. Python steppables dont have this fcn
            _module.init(self.caManager)
        except AttributeError:
            pass
            
        try:
            _registry[_module.__class__.__name__].append(_module)
        except LookupError:
            _registry[_module.__class__.__name__]=[_module]        
        return _module
    
    def __modules(self,_container):
        print '_container=',_container
        for moduleName, moduleList in _container.iteritems():
            for module in moduleList:
                yield module
        
    def steppables(self):
        return self.__modules(self.steppableRegistry)
        
    def steppablesBeforeMCS(self):
        return self.__modules(self.steppableBeforeMCSRegistry)
        
        
    def registerSteppable(self,_steppable):
        return self.__registerModule(_steppable,self.steppableRegistry)
        

    def registerBeforeMCSSteppable(self,_steppable):
        return self.__registerModule(_steppable,self.steppableBeforeMCSRegistry)


    def registerSolver(self,_steppable):
        return self.registerSteppable(_steppable)
        
    def registerBeforeMCSSolver(self,_steppable):
        return self.registerBeforeSteppable(_steppable)
        

    def registerSteppableByName(self,_steppableName):
        
        try:
            exec("from "+_steppableName+" import "+_steppableName+"\n")
            steppable=eval(_steppableName+"()")
            return self.registerSteppable(steppable)
        except ImportError:
            print 'COULD NOT IMPORT ',_steppableName
            return None        
        
        
    def registerBeforeMCSSteppableByName(self,_steppableName):
        
        try:
            exec("from "+_steppableName+" import "+_steppableName+"\n")
            steppable=eval(_steppableName+"()")
            return self.registerBeforeMCSSteppable(steppable)
        except ImportError:
            print 'COULD NOT IMPORT ',_steppableName
            return None  
            
    def start(self):
        for steppable in self.steppables():
            steppable.start()
            
        for steppable in self.steppablesBeforeMCS():
            steppable.start()
    
    def step(self, i):        
        for steppable in self.steppables():
            steppable.step(i) 
    
    def stepBeforeMCS(self, i):        
        for steppable in self.steppablesBeforeMCS():
            steppable.step(i) 
    
    def getArg(self,kwds,argName):
        try:
            return kwds[argName]            
        except LookupError:
            print 'Could not find attribute ', argName
            return None
    
    def initialize(self, *args, **kwds):
        '''
        Allowed entries in the kwds dictionary are 
        dim - specifies dimension of the lattice
        globalCarryingCapacity  - specifies global carrying capacity of each lattice site
        numSteps - specifies number of monte carlo steps the simulation is supposed to run for
        
        '''
        self.dim = self.getArg(kwds,'dim')
        self.globalCarryingCapacity = self.getArg(kwds,'globalCarryingCapacity')        
        self.numSteps = self.getArg(kwds,'numSteps')        
        
        import CompuCellSetup
        self.caManager,self.simthread = CompuCellSetup.getCoreCASimulationObjects(False)  
      
        if self.dim.x != 1 and self.dim.y != 1 and self.dim.z != 1:
            self.neighborOrder = 3
            
        self.caManager.setNeighborOrder(self.neighborOrder) # setting neighbor order CA copies
        self.caManager.setCellCarryingCapacity(self.globalCarryingCapacity) # setting global carrying capacity
        self.caManager.registerPythonAttributeAdderObject(self.dictAdder) # setting python dictionary adder so that each cell has python dictionary attached to it
        self.caManager.setNumSteps(self.numSteps)
        
        self.caManager.createCellField(self.dim)
        self.cellField = self.caManager.getCellField()
        
        import CenterOfMassMonitor

        self.moduleRegistry['CenterOfMassMonitor']=CenterOfMassMonitor.CenterOfMassMonitor()    

        print "self.moduleRegistry['CenterOfMassMonitor']=",self.moduleRegistry['CenterOfMassMonitor']
        print 'dir=',dir(self.moduleRegistry['CenterOfMassMonitor'])
#         sys.exit()
        self.moduleRegistry['CenterOfMassMonitor'].init(self.caManager)
        self.caManager.registerCellFieldChangeWatcher(self.moduleRegistry['CenterOfMassMonitor'])        

        print 'self.dim=',self.dim 
        print 'self.globalCarryingCapacity=',self.globalCarryingCapacity
        
#     def createCAManager(self, fieldDim):
        
#         import CA
#         self.caManager=CA.CAmanager()
        
#         if fieldDim.x != 1 and fieldDim.y != 1 and fieldDim.z != 1:
#             self.neighborOrder = 3
            
#         self.caManager.setNeighborOrder(self.neighborOrder)
        
#     def setGlobalCarryingCapacity(self, _carryingCapacity):
#         caManager.setCellCarryingCapacity(carryingCapacity)
    
    
    def mainLoop(self):
        import CompuCellSetup
        simthread=CompuCellSetup.simulationThreadObject

        simthread.stopped=False
        simthread.setSimulator(self.caManager)


        # # # if not steppableRegistry is None:    
        # # #     steppableRegistry.start()


        simthread.postStartInit()
        simthread.waitForPlayerTaskToFinish()


        # # # if not steppableRegistry is None:
        # # #     steppableRegistry.start()


        self.start()

        beginingStep=0
        i=beginingStep

# # #         print '\n\n\n\nBEFORE MAIN LOOP CA.PY'
# # #         print 'caManager.getCellInventory().getSize()=',caManager.getCellInventory().getSize()

        while True:
            simthread.beforeStep(i)
            
            # cell=caManager.createAndPositionCell(CA.Point3D(1,1,0))

            
            self.caManager.step(i)
            
            self.step(i)
            

            if simthread.getStopSimulation() or CompuCellSetup.userStopSimulationFlag:
                runFinishFlag=False;
                break 
                
                
            screenUpdateFrequency = simthread.getScreenUpdateFrequency()
        #        imgOutFlag = simthread.getImageOutputFlag()
            imgOutFlag = False
            screenshotFrequency = simthread.getScreenshotFrequency()
            latticeFlag = simthread.getLatticeOutputFlag()
            latticeFrequency = simthread.getLatticeOutputFrequency()
            
        #        print MYMODULENAME,"mainLoopNewPlayer:  screenUpdateFrequency=",screenUpdateFrequency," screenshotFrequency=",screenshotFrequency
            if simthread is not None:
                if (i % screenUpdateFrequency == 0) or (imgOutFlag and (i % screenshotFrequency == 0)):
                    simthread.loopWork(i)
                    simthread.loopWorkPostEvent(i)
                    screenUpdateFrequency = simthread.getScreenUpdateFrequency()
            
                
        # # #     simthread.loopWork(i)
        # # #     simthread.loopWorkPostEvent(i)
        # # #     screenUpdateFrequency = simthread.getScreenUpdateFrequency()
           
            i+=1        
            if i>=self.caManager.getNumSteps():
                break    
            
        self.caManager.cleanAfterSimulation()
        # sim.unloadModules()
        print "CALLING UNLOAD MODULES NEW PLAYER"
        if simthread is not None:
            simthread.sendStopSimulationRequest()
            print '\n\n\n\n\n\n\n SENDING HERE SIMULATION FINISHED REQUEST'
            simthread.simulationFinishedPostEvent(True)

 


        