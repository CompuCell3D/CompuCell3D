# -*- coding: utf-8 -*-
import os
import weakref
from cc3d.cpp import PlayerPython
from PyQt5 import QtCore
from PyQt5.QtCore import *
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.sim_runner import run_cc3d_project


class SimulationThread(QtCore.QThread):
    """
    QtThread - this is the object that is responsible for running simulation and
    communicating between PLayer code, CompuCellSetup simulation look and actual simulation
    """
    simulationInitializedSignal = pyqtSignal(int)
    completedStep = pyqtSignal(int)
    steppablesStarted = pyqtSignal(bool)
    simulationFinished = pyqtSignal(bool)
    finishRequest = pyqtSignal(bool)
    errorOccured = pyqtSignal(str, str)
    errorFormatted = pyqtSignal(str)
    errorOccuredDetailed = pyqtSignal(str, str,int,int,str)
    visFieldCreatedSignal = pyqtSignal(str, int)

    #
    # CONSTRUCTOR
    def __init__(self, parent=None):
        QtCore.QThread.__init__(self, parent)

        # NOTE: to implement synchronization between threads we use semaphores.
        # If yuou use mutexes for this then if you lokc mutex in one thread and try to unlock
        # from another thread than on Linux it will not work. Semaphores are better for this
        self.__ui = parent
        self.sem = QSemaphore(1)
        self.semPause = QSemaphore(1)
        # self.mutex = QtCore.QMutex()
        self.drawMutex = QtCore.QMutex()
        self.finishMutex = QtCore.QMutex()

        # this mutex will be unlocked externally when it is ok for simulation to finish
        self.finishMutex.lock()

        # self.pauseMutex = QtCore.QMutex()
        self.simulationInitialized = False
        self.stopThreadFlag = False
        self.stopped = False
        self.runUserPythonScriptFlag = False
        self.xmlFileName = ""
        self.pythonFileName = ""

        # following params updated in SimpleTabView.py: __paramsChanged() and elsewhere
        self.screenUpdateFrequency = 100
        self.imageOutputFlag = True
        self.screenshotFrequency = 100
        self.latticeOutputFlag = False
        self.latticeOutputFrequency = 100

        self.restartManager = None

        self.callingWidget = None
        self.__simModel = None
        self.__mcs = 0
        self.__fileWriter = None
        self.sim = None  # reference to CompuCell.Simulator()
        self.last_mcs_handled = 0

    # Python 2.6 requires importing of Example, CompuCell and Player Python modules from an instance
    # of QThread class (here Simulation Thread inherits from QThread)
    # Simulation Thread communicates with SimpleTabView using SignalSlot method.
    # If we dont import the three modules then when SimulationThread emits siglan and SimpleTabView
    # processes this signal in a slot (e.g. initializeSimulationViewWidget)
    # than calling a member function of an object from e.g. Player Python
    # (self.fieldStorage.allocateCellField(self.fieldDim))
    # results in segfault. Python 2.5 does not have this issue. Anyway this seems to work on Linux with Python 2.6
    # This might be a problem only on Linux
    # TODO UNCOMMENT
    # import Example
    # import CompuCell
    # import PlayerPython

    def emitErrorOccured(self,_errorType,_traceback_message):

        self.errorOccured.emit(_errorType,_traceback_message)

    def emitErrorFormatted(self,_errorMessage):
        self.errorFormatted.emit(_errorMessage)

    def redoCompletedStep(self):
        self.sem.tryAcquire()
        self.sem.release()
        self.loopWorkPostEvent(self.last_mcs_handled)
        # print('inside redoCompletedStep')
        # self.completedStep.emit(self.last_mcs_handled)

    def emitCompletedStep(self,_mcs=None):
        self.last_mcs_handled = _mcs
        self.completedStep.emit(_mcs)

    def emitSimulationInitialized(self,_flag=True):
        self.simulationInitializedSignal.emit(_flag)

    def emitSteppablesStarted(self,_flag=True):
        self.steppablesStarted.emit(_flag)

    def emitSimulationFinished(self,_flag=True):
        self.simulationFinished.emit(_flag)

    def emitFinishRequest(self,_flag=True):
        self.finishRequest.emit(_flag)

    def emitErrorOccuredDetailed(self,_errorType,_file,_line,_col,_traceback_message):
        self.errorOccuredDetailed.emit(_errorType,_file,_line,_col,_traceback_message)

    def emitVisFieldCreatedSignal(self, field_name, field_type):
        self.visFieldCreatedSignal.emit(field_name, field_type)

    def setSimulator(self, _sim):

        if _sim:
            self.sim = weakref.ref(_sim)

    def cleanup(self):
        self.restartManager = None

        self.callingWidget = None
        self.__simModel = None
        self.__mcs = 0
        self.__fileWriter = None
        self.sim = None

    def generatePIFFromRunningSimulation(self, _pifFileName):
        if self.__fileWriter is None:

            self.__fileWriter = PlayerPython.FieldWriter()
            # note self.sim is a weak reference so to pass underlying object to swigged-fcn
            # we need to derefernce it by using self.sim() expression
            pg = CompuCellSetup.persistent_globals

            self.__fileWriter.init(pg.simulator)
        self.__fileWriter.generatePIFFileFromCurrentStateOfSimulation(_pifFileName)

    def getCurrentStep(self):
        return self.__mcs

    # def getSimFileName(self):
    #     if self.__ui:
    #         return self.__ui.getSimFileName()
    #     else:
    #         return ''

    def setCurrentStep(self, _mcs):
        self.__mcs = _mcs

    def setCallingWidget(self, _callingWidget):
        self.callingWidget = _callingWidget

    def setGraphicsWidget(self, _graphicsWidget):
        self.graphicsWidget = _graphicsWidget

    def setSimulationXMLFileName(self, _xmlFileName):
        CompuCellSetup.simulationPaths.simulationXMLFileName = _xmlFileName
        self.xmlFileName = _xmlFileName

    def setSimulationPythonFileName(self, _pythonFileName):
        CompuCellSetup.simulationPaths.simulationPythonScriptName = _pythonFileName
        self.pythonFileName = _pythonFileName

    def setRunUserPythonScriptFlag(self, _flag):
        self.runUserPythonScriptFlag = _flag

    # added for compatibility reasons
    def clearGraphicsFields(self):
        pass

    # added for compatibility reasons
    def preStartInit(self):
        pass

    # added for compatibility reasons
    def postStartInit(self):

        self.sem.acquire()
        self.emitSimulationInitialized()

    def steppablePostStartPrep(self):

        self.sem.acquire()
        self.emitSteppablesStarted()

    def waitForPlayerTaskToFinish(self):
        self.sem.acquire()
        self.sem.release()

    def waitForFinishingTasksToConclude(self):
        self.finishMutex.lock()
        self.finishMutex.unlock()

    # added for compatibility reasons
    def setStopSimulation(self, _flag):
        self.stopped = _flag

    # added for compatibility reasons
    def getStopSimulation(self):
        return self.stopped

    def simulationFinishedPostEvent(self, _flag=True):
        self.emitSimulationFinished(_flag)

    # added for compatibility reasons
    def loopWork(self, _mcs):
        self.drawMutex.lock()
        self.drawMutex.unlock()

    # added for compatibility reasons
    def loopWorkPostEvent(self, _mcs):
        if self.getStopSimulation():
            return
        self.sem.acquire()
        self.emitCompletedStep(_mcs)

    # added for compatibility reasons
    def sendStopSimulationRequest(self):
        pass

    def createVectorFieldPy(self, _dim, _fieldName):

        return CompuCellSetup.createVectorFieldPy(_dim, _fieldName)

    def createVectorFieldCellLevelPy(self, _fieldName):

        return CompuCellSetup.createVectorFieldCellLevelPy(_fieldName)

    def createFloatFieldPy(self, _dim, _fieldName):

        return CompuCellSetup.createFloatFieldPy(_dim, _fieldName)

    def createScalarFieldCellLevelPy(self, _fieldName):

        return CompuCellSetup.createScalarFieldCellLevelPy(_fieldName)

    def getScreenUpdateFrequency(self):
        return self.screenUpdateFrequency

    def getImageOutputFlag(self):
        return self.imageOutputFlag

    def getScreenshotFrequency(self):
        return self.screenshotFrequency

    def getLatticeOutputFlag(self):
        return self.latticeOutputFlag

    def getLatticeOutputFrequency(self):
        return self.latticeOutputFrequency

    def beforeStep(self, _mcs):

        self.sem.acquire()
        self.sem.release()
        self.semPause.acquire()
        self.semPause.release()

        self.__mcs = _mcs

    def steerUsingGUI(self, _sim):

        if self.__simModel:

            dirtyModulesDict = self.__simModel.getDirtyModules()
            if dirtyModulesDict and len(list(dirtyModulesDict.keys())) > 0:
                if "Potts" in dirtyModulesDict:
                    _sim.updateCC3DModule(_sim.getCC3DModuleData("Potts"))
                    # print "NEW Temperature=",_sim.getCC3DModuleData("Potts").getFirstElement("Temperature").getText()
                if "Metadata" in dirtyModulesDict:
                    _sim.updateCC3DModule(_sim.getCC3DModuleData("Metadata"))

                if "Plugin" in dirtyModulesDict:
                    dirtyPluginDict = dirtyModulesDict["Plugin"]
                    for pluginName in dirtyPluginDict:
                        _sim.updateCC3DModule(_sim.getCC3DModuleData("Plugin", pluginName))
                if "Steppable" in dirtyModulesDict:
                    dirtySteppableDict = dirtyModulesDict["Steppable"]
                    for steppableName in dirtySteppableDict:
                        _sim.updateCC3DModule(_sim.getCC3DModuleData("Steppable", steppableName))

                _sim.steer()

                dirtyModulesDict.clear()

    def setSimModel(self, _simModel):
        self.__simModel = _simModel

    def __del__(self):
        try:
            self.sem.acquire()
            self.semPause.acquire()

            self.stopped = True
        finally:
            self.sem.release()
            self.semPause.release()


        self.wait()

    def stop(self):
        self.sem.tryAcquire()
        self.sem.release()

        self.stopped = True
        self.drawMutex.tryLock()
        self.drawMutex.unlock()

        self.semPause.tryAcquire()
        self.semPause.release()

    def prepareSimulation(self):

        (self.sim, self.simthread) = CompuCellSetup.get_core_simulation_objects()

        CompuCellSetup.initialize_simulation_objects(self.sim, self.simthread)
        CompuCellSetup.extra_init_simulation_objects(self.sim, self.simthread)
        self.simulationInitialized = True
        self.callingWidget.sim = self.sim

    def handleErrorMessage(self, _errorType, _traceback_message):
        print("INSIDE handleErrorMessage")
        print("_traceback_message=", _traceback_message)
        self.emitErrorOccured(_errorType, _traceback_message)

    def handleErrorMessageDetailed(self, _errorType, _file, _line, _col, _traceback_message):
        self.emitErrorOccuredDetailed(_errorType, _file, _line, _col, _traceback_message)

    def handleErrorFormatted(self, _errorMessage):
        self.emitErrorFormatted(_errorMessage)

    def runUserPythonScript(self, _scriptFileName, _globals, _locals):

        CompuCellSetup.persistent_globals.simthread = self

        # execfile("pythonSetupScripts/CompuCellPythonSimulationNewPlayer.py")
        # determinig the path of the CompuCellPythonSimulationNewPlayer.py based on the location of the current scrit (SimlulationThread.py)


        _path = os.path.abspath(os.path.dirname(__file__))

        # print '_path1 = ',_path

        _path = os.path.abspath(os.path.join(_path + '../../../'))

        # run_script_name = os.path.abspath(os.path.join(_path, 'pythonSetupScripts/CompuCellPythonSimulationNewPlayer.py'))

        # alternative
        # assume PREFIX_CC3D points to the CC3D installation directory
        run_script_name = os.path.abspath(
            os.path.join(os.environ.get('PREFIX_CC3D'), 'pythonSetupScripts/CompuCellPythonSimulationNewPlayer.py'))

        if not os.path.isfile(run_script_name):
            # assume PREFIX_CC3D points to the top of git repository
            run_script_name = os.path.abspath(os.path.join(os.environ.get('PREFIX_CC3D'),
                                                           'core/pythonSetupScripts/CompuCellPythonSimulationNewPlayer.py'))

        if not os.path.isfile(run_script_name):
            raise RuntimeError('Could not locate: CompuCellPythonSimulationNewPlayer.py run script')
        # print '_path2 = ',_path
        # print 'run_script_name =', run_script_name

        # this is in case player5 dire is soft-linked from git repository into installation repository
        # if not os.path.isfile(run_script_name):
        #
        #     run_script_name = os.path.abspath(os.path.join(_path, 'core/pythonSetupScripts/CompuCellPythonSimulationNewPlayer.py'))
        #
        # import py_compile
        # py_compile.compile(file=run_script_name)
        exec(compile(open(run_script_name).read(), run_script_name, 'exec'))

    def add_visualization_field(self, field_name, field_type):
        """

        :param field_name:
        :param field_type:
        :return:
        """

        print(" field_name, field_type=",( field_name, field_type))
        self.emitVisFieldCreatedSignal(field_name=field_name, field_type=field_type)

    def run(self):
        # from cc3d.CompuCellSetup.sim_runner import run_cc3d_project
        cc3d_sim_fname = CompuCellSetup.persistent_globals.simulation_file_name
        CompuCellSetup.persistent_globals.simthread = self
        run_cc3d_project(cc3d_sim_fname=cc3d_sim_fname)
        return
        # print('SIMTHREAD: GOT INSIDE RUN FUNCTION')
        # print('self.runUserPythonScriptFlag=',self.runUserPythonScriptFlag)
        #
        # if self.runUserPythonScriptFlag:
        #     # print "runUserPythonScriptFlag=",self.runUserPythonScriptFlag
        #     globalDict = {'simTabView': 20}
        #     localDict = {}
        #
        #     print('GOT INSIDE RUN FUNCTION');
        #
        #     self.runUserPythonScript(self.pythonFileName, globalDict, localDict)
