try:
    from xml.parsers.expat import ExpatError
    import sys
    from os import environ
    import string
    import traceback

    python_module_path=os.environ["PYTHON_MODULE_PATH"]
    appended=sys.path.count(python_module_path)
    if not appended:
        sys.path.append(python_module_path)    
    # sys.path.append(environ["PYTHON_MODULE_PATH"])        
    import CompuCellSetup

    sim,simthread = CompuCellSetup.getCoreSimulationObjects(True)

    if CompuCellSetup.simulationPaths.simulationPythonScriptName != "":
        execfile(CompuCellSetup.simulationPaths.simulationPythonScriptName)
    else:
        sim,simthread = CompuCellSetup.getCoreSimulationObjects()
        import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()
        #import CompuCellSetup

        CompuCellSetup.initializeSimulationObjects(sim,simthread)
        steppableRegistry = CompuCellSetup.getSteppableRegistry()
        CompuCellSetup.mainLoop(sim,simthread,steppableRegistry) # main loop - simulation is invoked inside this function

except IndentationError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    simthread=PlayerPython.getSimthreadBasePtr()
    simthread.handleErrorMessage("Python Indentation Error",traceback_message)
except SyntaxError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    simthread=PlayerPython.getSimthreadBasePtr()
    simthread.handleErrorMessage("Python Syntax Error",traceback_message)
except IOError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    simthread=PlayerPython.getSimthreadBasePtr()
    simthread.handleErrorMessage("Python IO Error",traceback_message)
except ImportError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    simthread=PlayerPython.getSimthreadBasePtr()
    simthread.handleErrorMessage("Python Import Error",traceback_message)
except ExpatError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    import PlayerPython
    simthread=PlayerPython.getSimthreadBasePtr()
    xmlFileName=CompuCellSetup.simulationPaths.simulationXMLFileName
    print "Error in XML File","File:\n "+xmlFileName+"\nhas the following problem\n"+e.message
    simthread.handleErrorMessage("Error in XML File","File:\n "+xmlFileName+"\nhas the following problem\n"+e.message)

except AssertionError,e:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    import PlayerPython
    simthread=PlayerPython.getSimthreadBasePtr()
    print "Assertion Error: ",e.message
    simthread.handleErrorMessage("Assertion Error",e.message)
except:
    if CompuCellSetup.simulationObjectsCreated:
        sim.finish()
    traceback_message=traceback.format_exc()
    import PlayerPython
    simthread=PlayerPython.getSimthreadBasePtr()
    print "Unexpected Error:",traceback_message
    simthread.handleErrorMessage("Unexpected Error",traceback_message)
