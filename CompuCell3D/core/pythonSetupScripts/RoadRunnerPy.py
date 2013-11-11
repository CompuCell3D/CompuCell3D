# make this class inner

from roadrunner import RoadRunner

class RoadRunnerPy(RoadRunner):
    def __init__(self,_sbmlFullPath=''):
        RoadRunner.__init__(self)
        
        self.sbmlFullPath=_sbmlFullPath
        
        self.stepSize=1.0
        self.timeStart=0.0
        self.timeEnd=1.0
        
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
