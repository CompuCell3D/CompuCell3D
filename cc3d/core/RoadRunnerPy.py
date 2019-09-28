# make this class inner
import os
import sys
import pickle

from roadrunner import RoadRunner


class RoadRunnerPy(RoadRunner):
    def __init__(self, _path='', _modelString='', sbml=None):
        if sbml is None:
            RoadRunner.__init__(self)
        else:
            RoadRunner.__init__(self, sbml)

        # self.sbmlFullPath=_sbmlFullPath
        self.path = _path  # relative path to the SBML model - CC3D uses only relative paths . In some rare cases when users do some hacking they may set self.path to be absolute path.
        self.absPath = ''  # absolute path of the SBML file. Internal use only e.g.  for debugging purposes - and is not serialized
        self.modelString = ''  # SBML model string - used when translating model specifications
        self.stepSize = 1.0
        self.timeStart = 0.0
        self.timeEnd = 1.0

        self.__state = {}

    # add properties    
    def setStepSize(self, _stepSize):
        self.stepSize = _stepSize

    def timestep(self, _numSteps=1, _stepSize=-1.0):

        if _stepSize > 0.0:
            self.timeEnd = self.timeStart + _numSteps * _stepSize  # we integrate with custom step size
        else:
            self.timeEnd = self.timeStart + _numSteps * self.stepSize  # we integrate with predefined step size

        # note that in general wnumber of steps should be higher - CVODE will use bigger steps when it can but setting steps to low number might actually caus instabilities.
        # note that using higher numbers does not really increase simulation time, actuallt it may shorten it - CVODE is better in going from short to long step than the other way around

        self.simulate(self.timeStart, self.timeEnd, steps=_numSteps)
        # print '_numSteps=',_numSteps
        # self.simulate(self.timeStart, self.timeEnd, steps=2)
        self.timeStart = self.timeEnd


        # def timestep(self,_numSteps=1,_stepSize=-1.0):

        # if _stepSize>0.0:
        # self.timeEnd=self.timeStart+_numSteps*_stepSize # we integrate with custom step size
        # else:    
        # self.timeEnd=self.timeStart+_numSteps*self.stepSize #we integrate with predefined step size

    # #         print 'absolute=',self.simulateOptions.absolute
    # #         print 'relative=',self.simulateOptions.relative
    # #         print 'stiff=',self.simulateOptions.stiff
    # #         print 'steps=',self.simulateOptions.steps

    # # note that in general wnumber of steps should be higher - CVODE will use bigger steps when it can but setting steps to low number might actually caus instabilities.
    # # note that using higher numbers does not really increase simulation time, actuallt it may shorten it - CVODE is better in going from short to long step than the other way around

    # #         steps is 1 by default
    # #         self.simulateOptions.steps=1
    # self.simulateOptions.start=self.timeStart
    # self.simulateOptions.end=self.timeEnd

    # self.simulate()
    # self.timeStart=self.timeEnd

    def prepareState(self):
        self.__state = {}
        # first line covers RRPython variables, second addresses rr.simulateOptions entries
        # self.__state['SimulateOptions'] ={'stepSize':self.stepSize,'timeStart':self.timeStart,'timeEnd':self.timeEnd,\
        # 'relative':self.simulateOptions.relative,'absolute':self.simulateOptions.absolute,'stiff':self.simulateOptions.stiff,'steps':self.simulateOptions.steps}# integratorSettings
        self.__state['SimulateOptions'] = {'stepSize': self.stepSize, 'timeStart': self.timeStart,
                                           'timeEnd': self.timeEnd, \
                                           'relative': self.getIntegrator().relative_tolerance,
                                           'absolute': self.getIntegrator().absolute_tolerance,
                                           'stiff': self.getIntegrator().stiff, \
                                           'steps': self.getIntegrator().maximum_num_steps}  # integratorSettings

        self.__state['ModelState'] = {}
        modelState = self.__state['ModelState']
        m = self.model
        for name in m.getFloatingSpeciesIds() + m.getBoundarySpeciesIds() + m.getGlobalParameterIds():
            modelState[name] = m[name]

    def __reduce__(self):
        self.prepareState()
        return RoadRunnerPy, (self.path,), self.__state

    def __setstate__(self, _state):
        self.__state = _state

    def loadSBML(self, _externalPath='', _modelString=''):
        """
        loads SBML model into RoadRunner instance
        external path can be either absolute path to SBML or a directory relative to which self.path is specified or empty string (in
        which case self.path is assumed to store absolute path to SBML file)

        TO BE REVISED - SOMEWHAT  STRANGE PATH MANIPULATIONS

        :param _externalPath:{str}
        :return: None
        """

        if _modelString == '':
            if _externalPath == '':  # if external

                if not os.path.exists(self.path):
                    raise IOError(
                        'loadSBMLError (self.path): RoadRunnerPy could not find ' + self.path + ' in the filesystem')
                self.absPath = os.path.abspath(self.path)

            else:
                if os.path.isdir(_externalPath):  # if path is a directory then we attempt to join it with
                    self.absPath = os.path.join(_externalPath, self.path)
                    if not os.path.exists(self.absPath) or os.path.isdir(self.absPath):
                        raise IOError(
                            'loadSBMLError Wrong constructed path: RoadRunnerPy could not find ' + self.absPath + ' in the filesystem')
                else:
                    if os.path.exists(_externalPath):
                        self.absPath = _externalPath
                    else:
                        raise IOError('loadSBMLError : RoadRunnerPy could not find ' + _externalPath + ' in the filesystem')

            self.load(self.absPath)
        else:
            self.modelString = _modelString
            self.load(self.modelString)

        try:
            modelState = self.__state['ModelState']
            for name, value in modelState.items():
                self.model[name] = value

            simulateOptions = self.__state['SimulateOptions']
            self.stepSize = simulateOptions['stepSize']
            self.timeStart = simulateOptions['timeStart']
            self.timeEnd = simulateOptions['timeEnd']

            # setting rr.simulateOPtions object entries
            # try: # older restart files might not have these options so will try to import what I can
            # self.simulateOptions.relative = simulateOptions['relative']
            # self.simulateOptions.absolute = simulateOptions['absolute']
            # self.simulateOptions.stiff = simulateOptions['stiff']
            # self.simulateOptions.steps = simulateOptions['steps']
            # except :
            # pass

            try:  # older restart files might not have these options so will try to import what I can
                self.getIntegrator().relative_tolarance = simulateOptions['relative']
                self.getIntegrator().absolute_tolarance = simulateOptions['absolute']
                self.getIntegrator().stiff = simulateOptions['stiff']
                self.getIntegrator().maximum_num_steps = simulateOptions['steps']
            except:
                pass


        except LookupError as e:
            pass
        # after using self.__state to initialize state of the model we set state dictionary to empty dicctionary
        self.__state = {}
