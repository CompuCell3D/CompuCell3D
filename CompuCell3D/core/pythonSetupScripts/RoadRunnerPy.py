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
            
        
#         print 'absolute=',self.simulateOptions.absolute
#         print 'relative=',self.simulateOptions.relative
#         print 'stiff=',self.simulateOptions.stiff        
#         print 'steps=',self.simulateOptions.steps        

        # note that in general wnumber of steps should be higher - CVODE will use bigger steps when it can but setting steps to low number might actually caus instabilities.
        # note that using higher numbers does not really increase simulation time, actuallt it may shorten it - CVODE is better in going from short to long step than the other way around    

#         steps is 1 by default    
#         self.simulateOptions.steps=1
        self.simulateOptions.start=self.timeStart
        self.simulateOptions.end=self.timeEnd        

        self.simulate()
        self.timeStart=self.timeEnd
    
        
    def raise_api_exception(self, attribute_name):
        raise RuntimeError("RoadRunner API Mismatch - could not handle properly integrator setting: "+attribute_name)
            
    def prepareState(self):
        self.__state={}
        #first line covers RRPython variables, second addresses rr.simulateOptions entries 
        # integrator settings        
        self.__state['SimulateOptions'] = {}        
        
        self.__state['SimulateOptions']['start'] = self.simulateOptions.start
        self.__state['SimulateOptions']['end'] = self.simulateOptions.end
        self.__state['SimulateOptions']['steps'] = self.simulateOptions.steps
        self.__state['SimulateOptions']['duration'] = self.simulateOptions.duration
        self.__state['SimulateOptions']['IntegratorSettings'] = {}
        
        integrator_setting_list = self.getIntegrator().getSettings() 
        for integrator_setting in integrator_setting_list:
            self.__state['SimulateOptions']['IntegratorSettings'][integrator_setting] = self.getIntegrator().getValue(integrator_setting)
                
        self.__state['ModelState']={}
        modelState=self.__state['ModelState']
        m=self.model
        for name in m.getFloatingSpeciesIds()+m.getBoundarySpeciesIds() + m.getGlobalParameterIds():
            modelState[name]=m[name]
        
        print 'self.__state=',self.__state
        sys.exit()
        
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
                    
          
        
        self.load(self.absPath)
        try:
            modelState=self.__state['ModelState']
            for name,value in modelState.iteritems():      
                self.model[name]=value            
                
            simulateOptions = self.__state['SimulateOptions']            
            self.simulateOptions['steps'] = simulateOptions['steps']
            self.stepSize = simulateOptions['duration']
            self.simulateOptions['start'] = simulateOptions['start']
            self.simulateOptions['end'] = simulateOptions['end']
            
            integrator_setting_list = self.getIntegrator().getSettings()
            
            for integrator_setting in integrator_setting_list:                
                try:
                    setting = self.__state['SimulateOptions']['IntegratorSettings'][integrator_setting] 
                    self.getIntegrator().setValue(integrator_setting,setting) 
                except:
                    pass    
            
        except LookupError,e:
            pass
        # after using self.__state to initialize state of the model we set state dictionary to empty dicctionary
        self.__state={}    
        
