# make this class inner

from RoadRunner import RoadRunner

class RoadRunnerPy(RoadRunner):
    def __init__(self,_sbmlFullPath='',_tempDirPath='',_compilerSupportPath='',_compilerExeFile=''):
        RoadRunner.__init__(self,_tempDirPath,_compilerSupportPath,_compilerExeFile)
        
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
        
        self.setNumPoints(_numSteps)
        self.setTimeStart(self.timeStart)
        self.setTimeEnd(self.timeEnd)
        self.simulate()
        self.timeStart=self.timeEnd
