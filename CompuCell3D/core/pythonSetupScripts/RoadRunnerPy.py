# make this class inner
import os
import sys
import cPickle

from roadrunner import RoadRunner

class RoadRunnerPy(RoadRunner):
    def __init__(self,_path=''):
        RoadRunner.__init__(self)
        
        # self.sbmlFullPath=_sbmlFullPath
        self.path=_path # relative path to the SBML model - CC3D uses only relative paths . In some rare cases when users do some hacking they may set self.path to be absolute path. 
        self.absPath='' # absolute path of the SBML file. Internal use only e.g.  for debugging purposes - and is not serialized
        self.stepSize=1.0
        self.timeStart=0.0
        self.timeEnd=1.0
        
        self.__state={}
        
    # add properties    
    def setStepSize(self,_stepSize):
        self.stepSize=_stepSize
        
    def timestep(self,_numSteps=1,_stepSize=-1.0):
        
        if _stepSize>0.0:
            self.timeEnd=self.timeStart+_numSteps*_stepSize # we integrate with custom step size
        else:    
            self.timeEnd=self.timeStart+_numSteps*self.stepSize #we integrate with predefined step size
        
        self.simulateOptions.steps=1
        self.simulateOptions.start=self.timeStart
        self.simulateOptions.end=self.timeEnd        
        self.simulate()
        self.timeStart=self.timeEnd
        
    def prepareState(self):
        self.__state={}
        self.__state['SimulateOptions'] ={'stepSize':self.stepSize,'timeStart':self.timeStart,'timeEnd':self.timeEnd}# integratorSettings
        self.__state['ModelState']={}
        modelState=self.__state['ModelState']
        m=self.model
        for name in m.getFloatingSpeciesIds()+m.getBoundarySpeciesIds() + m.getGlobalParameterIds():
            modelState[name]=m[name]
            
    def __reduce__(self):    
        self.prepareState()
        return RoadRunnerPy,(self.path,) , self.__state
        
    def __setstate__(self,_state):
        self.__state=_state

    def loadSBML(self,_externalPath=''):
        '''external path can be either absolute path to SBML or a directory relative to which self.path is specified or empty string (in which case self.path is assumed to store absolute path to SBML file)
        '''
        
        if _externalPath=='': #if external
            
            if not os.path.exists(self.path):
                raise IOError('loadSBMLError (self.path): RoadRunnerPy could not find '+self.path+' in the filesystem')
            self.absPath=os.path.abspath(self.path)
            
        else:
            if os.path.isdir(_externalPath): # if path is a directory then we attempt to join it with  
                self.absPath=os.path.join(_externalPath,self.path)
                if not os.path.exists(self.absPath) or os.path.isdir(self.absPath):
                    raise IOError('loadSBMLError Wrong constructed path: RoadRunnerPy could not find '+self.absPath+' in the filesystem')
            else:
                if os.path.exists(_externalPath):
                    self.absPath=_externalPath
                else:
                    raise IOError('loadSBMLError : RoadRunnerPy could not find '+_externalPath+' in the filesystem')
                    
        # # # print 'self.absPath=',self.absPath                
        
        self.load(self.absPath)
        try:
            modelState=self.__state['ModelState']
            for name,value in modelState.iteritems():      
                self.model[name]=value            
                
            simulateOptions=self.__state['SimulateOptions']            
            self.stepSize=simulateOptions['stepSize']
            self.timeStart=simulateOptions['timeStart']
            self.timeEnd=simulateOptions['timeEnd']
        except LookupError,e:
            pass
        # after using self.__state to initialize state of the model we set state dictionary to empty dicctionary
        self.__state={}    
        
