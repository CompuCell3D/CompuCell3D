# -*- coding: utf-8 -*-
import os, sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *


# from PyQt5.QtGui import *



class SimulationThread(QtCore.QThread):
    simulationInitializedSignal = pyqtSignal(int)
    completedStep = pyqtSignal(int)
    steppablesStarted = pyqtSignal(bool)
    simulationFinished = pyqtSignal(bool)
    finishRequest = pyqtSignal(bool)
    errorOccured = pyqtSignal(str, str)
    errorFormatted = pyqtSignal(str)
    errorOccuredDetailed = pyqtSignal(str, str,int,int,str)




    def emitErrorOccured(self,_errorType,_traceback_message):
        self.errorOccured.emit(_errorType,_traceback_message)


    def emitErrorFormatted(self,_errorMessage):
        self.errorFormatted.emit(_errorMessage)


    def emitCompletedStep(self,_mcs=None):
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



    pass

    #     __pyqtSignals__ = ("completedStep(int)","simulationInitialized(bool)","simulationInitialized(bool)",
    #   "errorOccured(QString,QString)","simulationFinished(bool)",
    # "finishRequest(bool)","errorOccuredDetailed(QString,QString,int,int,QString)","errorFormatted(QString)",)
    #
    #     # @QtCore.pyqtSignature("emitCompletedStep(int)")
    #     @QtCore.pyqt("emitCompletedStep(int)")
    #     def emitCompletedStep(self,_mcs=None):
    #         self.emit(SIGNAL("completedStep(int)") , _mcs)
    #
    #     @QtCore.pyqtSignature("emitFinishRequest(bool)")
    #     def emitFinishRequest(self,_flag=True):
    #         print 'emiting finish request'
    #         self.emit(SIGNAL("finishRequest(bool)"), _flag)
    #
    #
    #     @QtCore.pyqtSignature("simulationInitialized(bool)")
    #     def emitSimulationInitialized(self,_flag=True):
    #         self.emit(SIGNAL("simulationInitialized(bool)") , _flag)
    #
    #     @QtCore.pyqtSignature("steppablesStarted(bool)")
    #     def emitSteppablesStarted(self,_flag=True):
    #         self.emit(SIGNAL("steppablesStarted(bool)") , _flag)
    #
    #
    #
    #     @QtCore.pyqtSignature("errorOccured(QString,QString)")
    #     def emitErrorOccured(self,_errorType,_traceback_message):
    #         self.emit(SIGNAL("errorOccured(QString,QString)") , QString(_errorType),QString(_traceback_message))
    #
    #     @QtCore.pyqtSignature("errorOccuredDetailed(QString,QString,int,int,QString)")
    #     def emitErrorOccuredDetailed(self,_errorType,_file,_line,_col,_traceback_message):
    #         print "emitting errorOccuredDetailed"
    #         # print
    #         #
    #         self.emit(SIGNAL("errorOccuredDetailed(QString,QString,int,int,QString)") , QString(_errorType),QString(_file),_line,_col,QString(_traceback_message))
    #
    #     @QtCore.pyqtSignature("errorFormatted(QString)")
    #     def emitErrorFormatted(self,_errorMessage):
    #         print "emitting errorOccuredDetailed"
    #         # print
    #         #
    #         self.emit(SIGNAL("errorFormatted(QString)") , QString(_errorMessage))
    #
    #     @QtCore.pyqtSignature("errorFormatted(QString)")
    #     def emitErrorOccuredDetailed(self,_errorMessage):
    #         print "emitting errorFormatted"
    #         # print
    #         #
    #         self.emit(SIGNAL("errorFormatted(QString,QString,int,int,QString)") , QString(_errorMessage))
    #
    #     @QtCore.pyqtSignature("simulationFinished(bool)")
    #     def emitSimulationFinished(self,_flag=True):
    #         self.emit(SIGNAL("simulationFinished(bool)") , _flag)
    #
    # CONSTRUCTOR
    def __init__(self, parent=None):
        QtCore.QThread.__init__(self, parent)

        # NOTE: to implement synchronization between threads we use semaphores. If yuou use mutexes for this then if you lokc mutex in one thread and try to unlock
        # from another thread than on Linux it will not work. Semaphores are better for this
        self.__ui = parent
        self.sem = QSemaphore(1)
        self.semPause = QSemaphore(1)
        # self.mutex = QtCore.QMutex()
        self.drawMutex = QtCore.QMutex()
        self.finishMutex = QtCore.QMutex()
        self.finishMutex.lock()  # this mutex will be unlocked externally when it is ok for simulation to finish

        # self.pauseMutex = QtCore.QMutex()
        self.simulationInitialized = False
        self.stopThreadFlag = False
        self.stopped = False
        self.runUserPythonScriptFlag = False
        self.www = 21
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

    # Python 2.6 requires importing of Example, CompuCell and Player Python modules from an instance  of QThread class (here Simulation Thread inherits from QThread)
    # Simulation Thread communicates with SimpleTabView using SignalSlot method. If we dont import the three modules then when SimulationThread emits siglan and SimpleTabView
    # processes this signal in a slot (e.g. initializeSimulationViewWidget) than calling a member function of an object from e.g. Player Python (self.fieldStorage.allocateCellField(self.fieldDim))
    # results in segfault. Python 2.5 does not have this issue. Anyway this seems to work on Linux with Python 2.6
    # This might be a problem only on Linux
    # TODO UNCOMMENT
    import Example
    import CompuCell
    import PlayerPython

    def setSimulator(self, _sim):
        # self.sim=_sim
        # return
        import weakref
        if _sim:
            self.sim = weakref.ref(_sim)

    def cleanup(self):
        self.restartManager = None

        self.callingWidget = None
        self.__simModel = None
        self.__mcs = 0
        self.__fileWriter = None
        self.sim = None


        # self.condition = QtCore.QWaitCondition()

    def generatePIFFromRunningSimulation(self, _pifFileName):
        if self.__fileWriter is None:
            import PlayerPython
            self.__fileWriter = PlayerPython.FieldWriter()
            self.__fileWriter.init(
                self.sim())  # note self.sim is a weak reference so to pass underlying object to swigged-fcn we need to derefernce it by using self.sim() expression
        self.__fileWriter.generatePIFFileFromCurrentStateOfSimulation(_pifFileName)

    def getCurrentStep(self):
        return self.__mcs

    def getSimFileName(self):
        if self.__ui:
            return self.__ui.getSimFileName()
        else:
            return ''

    def setCurrentStep(self, _mcs):
        self.__mcs = _mcs

    def setCallingWidget(self, _callingWidget):
        self.callingWidget = _callingWidget

    def setGraphicsWidget(self, _graphicsWidget):
        self.graphicsWidget = _graphicsWidget

    def setSimulationXMLFileName(self, _xmlFileName):
        import CompuCellSetup
        CompuCellSetup.simulationPaths.simulationXMLFileName = _xmlFileName
        self.xmlFileName = _xmlFileName

    def setSimulationPythonFileName(self, _pythonFileName):
        import CompuCellSetup
        CompuCellSetup.simulationPaths.simulationPythonScriptName = _pythonFileName
        self.pythonFileName = _pythonFileName

    def setRunUserPythonScriptFlag(self, _flag):
        self.runUserPythonScriptFlag = _flag

    def clearGraphicsFields(self):  # added for compatibility reasons
        pass

    def preStartInit(self):  # added for compatibility reasons
        pass

    def postStartInit(self):  # added for compatibility reasons

        self.sem.acquire()

        # print 'prepareSimulation'
        # print self.restartManager

        # print self.callingWidget
        # print self.__simModel
        # print self.__mcs
        # print self.__fileWriter
        # print '\n\n\n\nself.sim=',self.sim



        # # sys.exit()

        # print 'POSTSTART INIT'
        # import time
        # time.sleep(5)

        self.emitSimulationInitialized()

    def steppablePostStartPrep(self):

        self.sem.acquire()

        self.emitSteppablesStarted()

    def waitForPlayerTaskToFinish(self):
        self.sem.acquire()
        self.sem.release()

    def setStopSimulation(self, _flag):  # added for compatibility reasons
        self.stopped = _flag

    def getStopSimulation(self):  # added for compatibility reasons
        return self.stopped

    def simulationFinishedPostEvent(self, _flag=True):
        self.emitSimulationFinished(_flag)

    def loopWork(self, _mcs):  # added for compatibility reasons
        self.drawMutex.lock()
        self.drawMutex.unlock()

    def loopWorkPostEvent(self, _mcs):  # added for compatibility reasons
        if self.getStopSimulation():
            return
        self.sem.acquire()
        #        print '-------- Sim-Thread.py:  loopWorkPostEvent(), _mcs=',_mcs
        self.emitCompletedStep(_mcs)

    def sendStopSimulationRequest(self):  # added for compatibility reasons
        pass

    def createVectorFieldPy(self, _dim, _fieldName):
        import CompuCellSetup
        return CompuCellSetup.createVectorFieldPy(_dim, _fieldName)

    def createVectorFieldCellLevelPy(self, _fieldName):
        import CompuCellSetup
        return CompuCellSetup.createVectorFieldCellLevelPy(_fieldName)

    def createFloatFieldPy(self, _dim, _fieldName):
        import CompuCellSetup
        return CompuCellSetup.createFloatFieldPy(_dim, _fieldName)

    def createScalarFieldCellLevelPy(self, _fieldName):
        import CompuCellSetup
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
            # print "self.__simModel=",self.__simModel

            dirtyModulesDict = self.__simModel.getDirtyModules()
            if dirtyModulesDict and len(dirtyModulesDict.keys()) > 0:
                if dirtyModulesDict.has_key("Potts"):
                    _sim.updateCC3DModule(_sim.getCC3DModuleData("Potts"))
                    # print "NEW Temperature=",_sim.getCC3DModuleData("Potts").getFirstElement("Temperature").getText()
                if dirtyModulesDict.has_key("Metadata"):
                    _sim.updateCC3DModule(_sim.getCC3DModuleData("Metadata"))

                if dirtyModulesDict.has_key("Plugin"):
                    dirtyPluginDict = dirtyModulesDict["Plugin"]
                    for pluginName in dirtyPluginDict:
                        # print "pluginName=",pluginName
                        _sim.updateCC3DModule(_sim.getCC3DModuleData("Plugin", pluginName))
                        # print "TARGET VOLUME=",_sim.getCC3DModuleData("Plugin","Volume").getFirstElement("TargetVolume").getText()
                        # print "_sim.getCC3DModuleData(\"Plugin\",\"Volume\").getFirstElement(\"TargetVolume\")=",_sim.getCC3DModuleData("Plugin","Volume").getFirstElement("TargetVolume")
                if dirtyModulesDict.has_key("Steppable"):
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

            # # # self.mutex.lock()
            # # # self.drawMutex.lock()
            self.stopped = True
        finally:
            self.sem.release()
            self.semPause.release()()

            # # # self.mutex.unlock()
            # # # self.drawMutex.unlock()

        # self.mutex.lock()
        # # self.abort = True
        # # self.condition.wakeOne()
        # self.mutex.unlock()

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

        import CompuCellSetup
        # CompuCellSetup.setSimulationXMLFileName(self.xmlFileName)
        (self.sim, self.simthread) = CompuCellSetup.getCoreSimulationObjects()

        import CompuCell
        CompuCellSetup.initializeSimulationObjects(self.sim, self.simthread)
        CompuCellSetup.extraInitSimulationObjects(self.sim, self.simthread)
        self.simulationInitialized = True
        self.callingWidget.sim = self.sim

    def handleErrorMessage(self, _errorType, _traceback_message):
        print "INSIDE handleErrorMessage"
        print "_traceback_message=", _traceback_message
        self.emitErrorOccured(_errorType, _traceback_message)
        # self.callingWidget.handleErrorMessage(_errorType,_traceback_message)

    def handleErrorMessageDetailed(self, _errorType, _file, _line, _col, _traceback_message):
        self.emitErrorOccuredDetailed(_errorType, _file, _line, _col, _traceback_message)

    def handleErrorFormatted(self, _errorMessage):
        print 'DUPA handleErrorFormatted'
        self.emitErrorFormatted(_errorMessage)

    def runUserPythonScript(self, _scriptFileName, _globals, _locals):
        import CompuCellSetup

        CompuCellSetup.simulationThreadObject = self

        # execfile("pythonSetupScripts/CompuCellPythonSimulationNewPlayer.py")
        # determinig the path of the CompuCellPythonSimulationNewPlayer.py based on the location of the current scrit (SimlulationThread.py)
        import os

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
        execfile(run_script_name)


    def run(self):

        if self.runUserPythonScriptFlag:
            # print "runUserPythonScriptFlag=",self.runUserPythonScriptFlag
            globalDict = {'simTabView': 20}
            localDict = {}

            self.runUserPythonScript(self.pythonFileName, globalDict, localDict)
