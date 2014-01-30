# # # global sbmlModelCompilations
# # # global freeFloatingSBMLSimulator
# # # sbmlModelCompilations={} #{sbml file name - full path:''}
# # # freeFloatingSBMLSimulator={} # {name:RoadRunnerPy}



# this function will replace all API functions which refer to SBMLSolver in the event that no SBMLSolver event is installed
def SBMLSolverError(self, *args,**kwrds):
    import inspect
    line=inspect.stack()[1][2] 
    call=inspect.stack()[1][4]
    raise  AttributeError('SBMLSolverError line :'+str(line)+' call:'+str(call)+' Trying to access one of the SBML solver methods but SBMLSolver engine (e.g. RoadRunner) has not been installed with your CompuCell3D package')     


class SBMLSolverHelper(object):
    @classmethod
    def removeAttribute(cls, name):
        print 'cls=',cls
        return delattr(cls, name)    
    
    def __init__(self):        

        try:
        
            import roadrunner
            import os
            import sys
           
        except ImportError,e:
            #replacing SBMLSolver API with wrror messages 
            #delattr(SteppableBasePy, 'addSBMLToCell')
            SBMLSolverAPI=['addSBMLToCell','addSBMLToCellTypes','addSBMLToCellIds','addFreeFloatingSBML',\
            'deleteSBMLFromCellIds','deleteSBMLFromCellTypes','deleteSBMLFromCell',\
            'timestepCellSBML','timestepFreeFloatingSBML','timestepSBML',
            'setStepSizeForCell','setStepSizeForCellIds','setStepSizeForCellTypes','setStepSizeForFreeFloatingSBML',\
            'getSBMLSimulator','getSBMLState','setSBMLState','getSBMLValue','setSBMLValue','normalizePath','copySBMLs']
            
            import types            
            for apiName in SBMLSolverAPI:
                SBMLSolverHelper.removeAttribute(apiName)
                setattr(SBMLSolverHelper, apiName, types.MethodType(SBMLSolverError, SBMLSolverHelper)) 

    def addSBMLToCell(self,_modelFile,_modelName='',_cell=None,_stepSize=1.0,_initialConditions={},_coreModelName='',_modelPathNormalized=''):
        import os
        import sys
        import CompuCell
        
        
        coreModelName=_modelName
        if coreModelName=='':
            coreModelName,ext=os.path.splitext(os.path.basename(_modelFile))
            
        
        modelPathNormalized=self.normalizePath(_modelFile)
        
        dict_attrib = CompuCell.getPyAttrib(_cell)
        
        sbmlDict={}
        if dict_attrib.has_key('SBMLSolver'):
            sbmlDict=dict_attrib['SBMLSolver']
        else:
            dict_attrib['SBMLSolver']=sbmlDict    
        
        
        
                
        from RoadRunnerPy import RoadRunnerPy         
        rr=RoadRunnerPy(_path=_modelFile)
        # # # rr=RoadRunnerPy(_sbmlFullPath=modelPathNormalized)
        
        #setting stepSize
        rr.stepSize=_stepSize                    
        #loading SBML and LLVM-ing it           
        rr.loadSBML(_externalPath=modelPathNormalized)
        
        # # # rr.load(modelPathNormalized)
        #storing rr instance in the cell dictionary
        sbmlDict[coreModelName]=rr
            
        #setting initial conditions - this has to be done after loadingSBML       
        for name,value in _initialConditions.iteritems():
            rr.model[name]=value
        
        
        
    def addSBMLToCellTypes(self,_modelFile='',_modelName='',_types=[],_stepSize=1.0,_initialConditions={}):        
        coreModelName=_modelName
        if coreModelName=='':
            coreModelName,ext=os.path.splitext(os.path.basename(_modelFile))
        
        modelPathNormalized=self.normalizePath(_modelFile)   
        for cell in self.cellListByType(*_types):
            self.addSBMLToCell(_modelFile=_modelFile,_modelName=_modelName,_cell=cell,_stepSize=_stepSize,_initialConditions=_initialConditions,_coreModelName=coreModelName,_modelPathNormalized=modelPathNormalized)

    def addSBMLToCellIds(self,_modelFile,_modelName='',_ids=[],_stepSize=1.0,_initialConditions={}):
        
        coreModelName=_modelName
        if coreModelName=='':
            coreModelName,ext=os.path.splitext(os.path.basename(_modelFile))
            
        modelPathNormalized=self.normalizePath(_modelFile)
        # print 'will try to add SBML to ids=',_ids
        for id in _ids:
            cell=self.inventory.attemptFetchingCellById(id)
            # print 'THIS IS CELL ID=',cell.id
            if not cell:
                
                continue
                
            self.addSBMLToCell(_modelFile=_modelFile,_modelName=_modelName,_cell=cell,_stepSize=_stepSize,_initialConditions=_initialConditions,_coreModelName=coreModelName,_modelPathNormalized=modelPathNormalized)

    def addFreeFloatingSBML(self,_modelFile,_modelName,_stepSize=1.0,_initialConditions={}):
        
            
#         modelPathNormalized=os.path.abspath(_modelFile)    
        
        modelPathNormalized=self.normalizePath(_modelFile)
        try:
            f=open(modelPathNormalized,'r')
            f.close()
        except IOError, e:
            if self.simulator.getBasePath()!='':
                modelPathNormalized=os.path.abspath(os.path.join(self.simulator.getBasePath(),modelPathNormalized))

        from RoadRunnerPy import RoadRunnerPy         
        # # # rr=RoadRunnerPy(_sbmlFullPath=modelPathNormalized)
        # # # rr.load(modelPathNormalized)
        rr=RoadRunnerPy(_path=_modelFile)
        rr.loadSBML(_externalPath=modelPathNormalized)
        
        #setting stepSize
        rr.stepSize=_stepSize

        #storing         
        
        # # # global freeFloatingSBMLSimulator
        # # # freeFloatingSBMLSimulator[_modelName]=rr        
        
        import CompuCellSetup
        CompuCellSetup.freeFloatingSBMLSimulator[_modelName]=rr        
        
        #setting initial conditions - this has to be done after loadingSBML
        for name,value in _initialConditions.iteritems():
            rr.model[name]=value
            # # # rr.setValue(name,value)

    def deleteSBMLFromCellIds(self,_modelName,_ids=[]):
        import CompuCell
        for id in _ids:
            cell=self.inventory.attemptFetchingCellById(id)
            if not cell:
                continue
                
            dict_attrib=CompuCell.getPyAttrib(cell)            
            try:
                sbmlDict=dict_attrib['SBMLSolver']
                del sbmlDict[_modelName]
            except LookupError,e:
                pass    

    def deleteSBMLFromCellTypes(self,_modelName,_types=[]):
        import CompuCell
        for cell in self.cellListByType(*_types):
            dict_attrib=CompuCell.getPyAttrib(cell)            
            try:
                sbmlDict=dict_attrib['SBMLSolver']
                del sbmlDict[_modelName]
            except LookupError,e:
                pass    
                
    def deleteSBMLFromCell(self,_modelName='',_cell=None):
            import CompuCell
            dict_attrib=CompuCell.getPyAttrib(_cell)            
            try:
                sbmlDict=dict_attrib['SBMLSolver']
                del sbmlDict[_modelName]
            except LookupError,e:
                pass            
        
    def deleteFreeFloatingSBML(self,_modelName):
        # # # global freeFloatingSBMLSimulator        
        # # # try:

            # # # global freeFloatingSBMLSimulator
            # # # del freeFloatingSBMLSimulator[_modelName]
        # # # except LookupError,e:
            # # # pass    
            
        import CompuCellSetup
        try:
            del CompuCellSetup.freeFloatingSBMLSimulator[_modelName]
        except LookupError,e:
            pass    
        
        
    def timestepCellSBML(self):
        import CompuCell
        #timestepping SBML attached to cells
        for cell in self.cellList:
            dict_attrib=CompuCell.getPyAttrib(cell)
            if dict_attrib.has_key('SBMLSolver'):
                sbmlDict=dict_attrib['SBMLSolver']
            
                for modelName, rrTmp in sbmlDict.iteritems():
                    rrTmp.timestep()
        
#                     print 'modelName=',modelName,'id=',cell.id,'rr t=',rrTmp.timeStart,' S1=',rrTmp.getValue('S1'),' S2=',rrTmp.getValue('S2')
                    
        
    def setStepSizeForCell(self, _modelName='',_cell=None,_stepSize=1.0):
        import CompuCell        
        dict_attrib = CompuCell.getPyAttrib(_cell)
        
        try:
            sbmlSolver=dict_attrib['SBMLSolver'][_modelName]
        except LookupError,e:
            return
            
        sbmlSolver.stepSize=_stepSize    
        
    def setStepSizeForCellIds(self, _modelName='',_ids=[],_stepSize=1.0):
        for id in _ids:
            cell=self.inventory.attemptFetchingCellById(id)
            if not cell:
                continue
            self.setStepSizeForCell(_modelName=_modelName,_cell=cell,_stepSize=_stepSize)
            
    def setStepSizeForCellTypes(self, _modelName='',_types=[],_stepSize=1.0):
        for cell in self.cellListByType(*_types):
            self.setStepSizeForCell(_modelName=_modelName,_cell=cell,_stepSize=_stepSize)

    def setStepSizeForFreeFloatingSBML(self, _modelName='',_stepSize=1.0):
        
        # # # try:

            # # # global freeFloatingSBMLSimulator
            # # # sbmlSolver=freeFloatingSBMLSimulator[_modelName]
        # # # except LookupError,e:
            # # # return
            
        # # # sbmlSolver.stepSize=_stepSize    

        try:

            import CompuCellSetup
            sbmlSolver=CompuCellSetup.freeFloatingSBMLSimulator[_modelName]
        except LookupError,e:
            return
            
        sbmlSolver.stepSize=_stepSize    
        

    def timestepFreeFloatingSBML(self):    
        #timestepping free-floating SBML        
        # # # global freeFloatingSBMLSimulator
        # # # for modelName, rr in freeFloatingSBMLSimulator.iteritems():
           # # # rr.timestep() 
        
        import CompuCellSetup
        for modelName, rr in CompuCellSetup.freeFloatingSBMLSimulator.iteritems():
           rr.timestep() 
            
    def timestepSBML(self):        
        self.timestepCellSBML()
        self.timestepFreeFloatingSBML()
        
    def getSBMLSimulator(self,_modelName,_cell=None):
        '''
        This function returns a reference to RoadRunnerPy or None 
        '''
        # # # import CompuCell
        # # # if not _cell:
            # # # try:
                # # # global freeFloatingSBMLSimulator
                # # # return freeFloatingSBMLSimulator[_modelName]

            # # # except LookupError,e:
                # # # return None    
        # # # else:  
            # # # try:
                # # # dict_attrib=CompuCell.getPyAttrib(_cell)                
                # # # return dict_attrib['SBMLSolver'][_modelName]
            # # # except LookupError,e:
                # # # return None    
        import CompuCell
        import CompuCellSetup
        if not _cell:
            try:
                
                return CompuCellSetup.freeFloatingSBMLSimulator[_modelName]

            except LookupError,e:
                return None    
        else:  
            try:
                dict_attrib=CompuCell.getPyAttrib(_cell)                
                return dict_attrib['SBMLSolver'][_modelName]
            except LookupError,e:
                return None    
            
    def getSBMLState(self,_modelName,_cell=None):
        ''' returns instance of the RoadRunner.model which behaves as a python dictionary but has many entries some of which are non-assignable /non-mutable
        '''
        sbmlSimulator=self.getSBMLSimulator(_modelName,_cell)
        if not sbmlSimulator:
            if _cell:
                raise RuntimeError("Could not find model "+_modelName+' attached to cell.id=',_cell.id) 
            else:
                raise RuntimeError("Could not find model "+_modelName+' in the list of free floating SBML models')                 
        else:
        
            # print 'sbmlSimulator.model.keys()=',sbmlSimulator.model.keys()
            return sbmlSimulator.model
            


    def getSBMLStateAsPythonDict(self,_modelName,_cell=None):
        ''' returns  python dictionary which has floating species, boundary species and global variables
        '''
    
        sbmlSimulator=self.getSBMLSimulator(_modelName,_cell)
        if not sbmlSimulator:
            if _cell:
                raise RuntimeError("Could not find model "+_modelName+' attached to cell.id=',_cell.id) 
            else:
                raise RuntimeError("Could not find model "+_modelName+' in the list of free floating SBML models')                 
        else:
        
            
            state={}
            for name in sbmlSimulator.model.getFloatingSpeciesIds()+sbmlSimulator.model.getBoundarySpeciesIds() + sbmlSimulator.model.getGlobalParameterIds():
                state[name]=sbmlSimulator.model[name]
            return state    
        
    def setSBMLState(self,_modelName,_cell=None,_state={}):
        
        sbmlSimulator=self.getSBMLSimulator(_modelName,_cell)
        if not sbmlSimulator:
            return False
        else:
        
            if _state==sbmlSimulator.model: # no need to do anything when all the state changes are done on model
                return True             
            
            for name,value in _state.iteritems():      
                sbmlSimulator.model[name]=value            
            return True
            
    def getSBMLValue(self,_modelName,_valueName='',_cell=None):
        sbmlSimulator=self.getSBMLSimulator(_modelName,_cell)
        if not sbmlSimulator:
            if _cell:
                raise RuntimeError("Could not find model "+_modelName+' attached to cell.id=',_cell.id) 
            else:
                raise RuntimeError("Could not find model "+_modelName+' in the list of free floating SBML models')     
        else:
            return sbmlSimulator[_valueName]            

    def setSBMLValue(self,_modelName,_valueName='',_value=0.0,_cell=None):
        sbmlSimulator=self.getSBMLSimulator(_modelName,_cell)
        if not sbmlSimulator:
            return False
        else:
            sbmlSimulator.model[_valueName]=_value
            return True
            
            

    def copySBMLs(self,_fromCell,_toCell,_sbmlNames=[]):
        sbmlNamesToCopy=[]
        import CompuCell
        if not(len(_sbmlNames)): 
            #if user does not specify _sbmlNames we copy all SBML networks
            try:
                dict_attrib=CompuCell.getPyAttrib(_fromCell)                
                sbmlDict=dict_attrib['SBMLSolver']
                sbmlNamesToCopy=sbmlDict.keys()
            except LookupError,e:
                pass 
        else:
            sbmlNamesToCopy=_sbmlNames
            
        try:
            dict_attrib_from=CompuCell.getPyAttrib(_fromCell)                
            sbmlDictFrom=dict_attrib_from['SBMLSolver']
        except LookupError,e:    
            # if  _fromCell does not have SBML networks there is nothing to copy
            return
        
        try:
            dict_attrib_to=CompuCell.getPyAttrib(_toCell)                
            sbmlDictTo=dict_attrib_to['SBMLSolver']
        except LookupError,e:    
            #if _toCell does not have SBMLSolver dictionary entry we simply add it
            dict_attrib_to['SBMLSolver']={}
            sbmlDictTo=dict_attrib_to['SBMLSolver']
            
        for sbmlName in sbmlNamesToCopy:

            # # # stateFrom=self.getSBMLState(_modelName=sbmlName,_cell=_fromCell)
            stateFrom=self.getSBMLStateAsPythonDict(_modelName=sbmlName,_cell=_fromCell)            
            
            rrFrom=sbmlDictFrom[sbmlName]
            pathFrom=rrFrom.path
            pathFromNormalized=self.normalizePath(pathFrom)
            self.addSBMLToCell(_modelFile=pathFrom,_modelName=sbmlName,_cell=_toCell,_stepSize=rrFrom.stepSize,_initialConditions=stateFrom,_coreModelName=sbmlName,_modelPathNormalized=pathFromNormalized)
                                
        
    def normalizePath(self,_path):
        '''
        this function checks if file exists and if not it joins basepath (path to the root of the cc3d project) with path
        '''        
        import os
        import sys
        
        pathNormalized=_path
        try:
            f=open(pathNormalized,'r')
            f.close()            
        except IOError, e:
            if self.simulator.getBasePath()!='':
                pathNormalized=os.path.abspath(os.path.join(self.simulator.getBasePath(),pathNormalized))
            
        return pathNormalized    
