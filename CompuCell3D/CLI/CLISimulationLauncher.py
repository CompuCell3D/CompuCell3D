from SimulationLauncher import SimulationLauncher

class CLISimulationLauncher(SimulationLauncher):

    def executeSimulation(self):
        pass

    def __setupSimulationObject(self):
        """
        This function creates and initializes the core simulation objects. Sim and SimThread.
        :return:
        """
        pass


    def getCoreSimulationObjectsNewPlayer(_parseOnlyFlag=False, _cmlOnly=False):
        import sys
        from os import environ
        import string

        import SystemUtils
        SystemUtils.setSwigPaths()
        SystemUtils.initializeSystemResources()
        # this dummy library was necessary to get restarting of the Python interpreter from C++ to work with SWIG generated libraries
        import Example

        import CompuCell
        CompuCell.initializePlugins()
        simthread = None
        sim = None

        global simulationPaths
        if not _parseOnlyFlag:
            sim = CompuCell.Simulator()
            sim.setNewPlayerFlag(True)
            sim.setBasePath(simulationPaths.basePath)
            if simthread is not None:
                simthread.setSimulator(sim)


            # here I will append path to search paths based on the paths to XML file and Python script paths
            global appendedPaths
            if simulationPaths.playerSimulationPythonScriptPath != "":
                sys.path.insert(0, simulationPaths.playerSimulationPythonScriptPath)
                appendedPaths.append(simulationPaths.playerSimulationPythonScriptPath)

            if simulationPaths.pathToPythonScriptNameFromXML != "":
                sys.path.insert(0, simulationPaths.pathToPythonScriptNameFromXML)
                appendedPaths.append(simulationPaths.pathToPythonScriptNameFromXML)

            if simulationPaths.playerSimulationXMLFilePath != "":
                sys.path.insert(0, simulationPaths.playerSimulationXMLFilePath)
                appendedPaths.append(simulationPaths.playerSimulationXMLFilePath)

            if simulationPaths.pathToxmlFileNameFromPython != "":
                sys.path.insert(0, simulationPaths.pathToxmlFileNameFromPython)
                appendedPaths.append(simulationPaths.pathToxmlFileNameFromPython)

            # initModules(sim)#extracts Plugins, Steppables and Potts XML elements and passes it to the simulator


            global simulationObjectsCreated
            simulationObjectsCreated = True

        global cmlFieldHandler
        # import CMLFieldHandler
        # cmlFieldHandler=CMLFieldHandler.CMLFieldHandler()
        # cmlFieldHandler.sim=sim
        createCMLFileHandler(sim)
        return sim, cmlFieldHandler

    def initializeSimulationObjects(sim, simthread):
        pass

    def mainLoopCLI(sim, simthread, steppableRegistry=None, _screenUpdateFrequency=None):
        print '\n### Staring main Loop CLI'
        global cmlFieldHandler  # rwh2
        global globalSteppableRegistry  # rwh2
        globalSteppableRegistry = steppableRegistry
        import ProjectFileStore

        extraInitSimulationObjects(sim, simthread)

        global customScreenshotDirectoryName
        global cc3dSimulationDataHandler
        global simulationPaths
        if customScreenshotDirectoryName:
            makeCustomSimDir(customScreenshotDirectoryName)
            if cc3dSimulationDataHandler is not None:
                cc3dSimulationDataHandler.copySimulationDataFiles(customScreenshotDirectoryName)
                simulationPaths.setSimulationResultStorageDirectoryDirect(customScreenshotDirectoryName)

        runFinishFlag = True

        if not steppableRegistry is None:
            steppableRegistry.init(sim)
            steppableRegistry.start()
        # init fieldWriter
        if cmlFieldHandler:
            cmlFieldHandler.fieldWriter.init(sim)
            cmlFieldHandler.getInfoAboutFields()
            cmlFieldHandler.outputFrequency = ProjectFileStore.outputFrequency
            cmlFieldHandler.outputFileCoreName = "output"

            cmlFieldHandler.prepareSimulationStorageDir(
                os.path.join(ProjectFileStore.outputDirectoryPath, "LatticeData"))
            cmlFieldHandler.setMaxNumberOfSteps(
                sim.getNumSteps())  # will determine the length text field  of the step number suffix
            cmlFieldHandler.writeXMLDescriptionFile()  # initialization of the cmlFieldHandler is done - we can write XML description file

            # self.simulationXMLFileName=""
            # self.simulationPythonScriptName=""

            print "simulationPaths XML=", simulationPaths.simulationXMLFileName
            print "simulationPaths PYTHON=", simulationPaths.simulationPythonScriptName

        global current_step
        current_step = 0

        # when num steps are declared at the CML
        # they have higher precedence than the number of MCS than declared in the simulation file

        # for current_step in range(sim.getNumSteps()):
        while True:
            # calling Python steppables which are suppose to run before MCS - e.g. secretion steppable
            if userStopSimulationFlag:
                runFinishFlag = False;
                break

            if not steppableRegistry is None:
                steppableRegistry.stepRunBeforeMCSSteppables(current_step)

            sim.step(current_step)  # steering using steppables

            if sim.getRecentErrorMessage() != "":
                raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

            if not steppableRegistry is None:
                steppableRegistry.step(current_step)

            if cmlFieldHandler.outputFrequency and not (current_step % cmlFieldHandler.outputFrequency):
                #            print MYMODULENAME,' mainLoopCML: cmlFieldHandler.writeFields(i), i=',i
                cmlFieldHandler.writeFields(current_step)
                # cmlFieldHandler.fieldWriter.addCellFieldForOutput()
                # cmlFieldHandler.fieldWriter.writeFields(cmlFieldHandler.outputFileCoreName+str(i)+".vtk")
                # cmlFieldHandler.fieldWriter.clear()

            # steer application will only update modules that uses requested using updateCC3DModule function from simulator
            sim.steer()
            if sim.getRecentErrorMessage() != "":
                raise CC3DCPlusPlusError(sim.getRecentErrorMessage())

            current_step += 1
            # print 'sim.getNumSteps()=',sim.getNumSteps()
            if current_step >= sim.getNumSteps():
                break

        print "END OF SIMULATION  "
        if runFinishFlag:
            sim.finish()
            steppableRegistry.finish()
            sim.cleanAfterSimulation()
        else:
            sim.cleanAfterSimulation()
            print "CALLING UNLOAD MODULES"


            # In exception handlers you have to call sim.finish to unload the plugins .
            # We may need to introduce new funuction name (e.g. unload) because finish does more than unloading
