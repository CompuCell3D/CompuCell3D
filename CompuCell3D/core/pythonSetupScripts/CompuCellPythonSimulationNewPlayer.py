def pythonTracebackFormatter():
    import string
    import re
    import traceback
    import sys
    lineNumberExtractRegex=re.compile('^[\s\S]*line[\s]*([0-9]*)')
    # fileNameExtractRegex=re.compile('^[\s]*File[\s]*[\"]?([\s\S]*)[\"]?line[\s]*([0-9]*)')
    fileNameExtractRegex=re.compile('^[\s]*File[\s]*[\"]?([\s\S]*)["]')
    columnNumberExtractRegex=re.compile('^[\s\S]*column[\s]*([0-9]*)')      
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    
    formatted_lines = traceback.format_exc().splitlines()   
    errorDescriptionLine=formatted_lines[-1]
  
    
    tracebackText="Error: "+errorDescriptionLine+"\n"
    
    fileNameFound=False
    
    
    lineNumber=0
    colNumber=0
    lineHeaderLength=0
    
    skipLine=False
    
    for idx in  range(len(formatted_lines)-1):
    
        line = formatted_lines[idx]
        fileNameGroups=fileNameExtractRegex.search(str(line))
        fileName=""
        if skipLine:
            skipLine=False
            continue
        
        
        try: 
            if fileNameGroups:
                fileName=fileNameGroups.group(1)
                fileNameFound=True
            else:
                # tracebackText+="    "+line+"\n"    
                
                if lineNumber>0:                    
                    
                    
                    try:    
                        colNumber=string.index(formatted_lines[idx+1],'^')    
                    except ValueError, e:    
                        pass    
                    

                    
                    lineHeader="    "+"Line: "+str(lineNumber)+" Col: "+str(colNumber)+" "
                    lineHeaderLength=len(lineHeader)                    
                    tracebackText+=lineHeader+line+"\n"
                    print "lineHeaderLength=",lineHeaderLength
                    
                    if colNumber>0:
                        tracebackText+=" "*lineHeaderLength+formatted_lines[idx+1]+"\n"
                        print lineHeader+line+"\n"
                        print " "*lineHeaderLength+formatted_lines[idx+1]+"\n"
                        
                        skipLine=True
                    else:
                        tracebackText+="    "+line+"\n"
                    lineNumber=0
                    colNumber=0
                    
                else:
                    
                    tracebackText+="    "+line+"\n"                    
                
                continue
                # dbgMsg("Searched text at line=",lineNumber)
                
        except IndexError,e:
            tracebackText+="    "+line+"\n"    
            continue
    
        lineNumberGroups=lineNumberExtractRegex.search(str(line))
        try: 
            if lineNumberGroups:
                lineNumber=int(lineNumberGroups.group(1))
                # dbgMsg("Searched text at line=",lineNumber)
                
        except IndexError,e:
                
            print "\n\n\n\n Could not find line number"        
        # try:    
            # colNumber=string.index(line,'^')    
        # except ValueError, e:    
            # pass
        
        
        
        # tracebackText+="    "+"Line: "+str(lineNumber)+" col: "+str(colNumber)+" "+line+"\n"
        if fileNameFound:
            
            tracebackText+="  File: "+fileName+"\n"
            
                 
            
        
            
    return tracebackText


def pythonErrorFormatter():
    import traceback, sys
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    # print "******************************************************** print_tb:"
    # traceback.print_tb(exc_traceback, limit=2, file=sys.stdout)
    # print "******************************************************** print_tb:"    
    
    # print "*** ********************extract_tb:"
    # print repr(traceback.extract_tb(exc_traceback))
    # print "**************************** extract_tb:"
    # print
    
    
    
    print "****** format_exc, first and last line:"
    
    
    
    formatted_lines = traceback.format_exc().splitlines()    
    errorDescriptionLine=formatted_lines[-1]
    fileLine=formatted_lines[-4]
    codeLine=formatted_lines[-3]
    columnLine=formatted_lines[-2]
    
    print errorDescriptionLine
    print fileLine
    print codeLine
    print columnLine
    
    
    print "****** format_exc, first and last line:"
    
    import re
    lineNumberExtractRegex=re.compile('^[\s\S]*line[\s]*([0-9]*)')
    
    fileNameExtractRegex=re.compile('^[\s]*File[\s]*([\s\S]*)line[\s]*([0-9]*)')
    
    columnNumberExtractRegex=re.compile('^[\s\S]*column[\s]*([0-9]*)')
    
    
    
    
    lineNumberGroups=lineNumberExtractRegex.search(str(fileLine))
    
    print "\n\n\n lineNumberGroups=",lineNumberGroups
    lineNumber=-1
    colNumber=-1
    try: 
        if lineNumberGroups:
            lineNumber=int(lineNumberGroups.group(1))
            # dbgMsg("Searched text at line=",lineNumber)
            
    except IndexError,e:
            
        print "\n\n\n\n Could not find line number"
    
    print "ERROR IN LINE:",lineNumber
    
    fileNameGroups=fileNameExtractRegex.search(str(fileLine))
    print "\n\n\n fileNameGroups=",fileNameGroups
    fileName=""
    try: 
        if fileNameGroups:
            fileName=fileNameGroups.group(1)
            
            # dbgMsg("Searched text at line=",lineNumber)
            
    except IndexError,e:
            
        print "\n\n\n\n Could not find file name"
        
    print "\n\n\n fileName:",fileName
    # formatted_lines = traceback.format_exc().splitlines()

    # print "formatted_lines:",formatted_lines
    import string
    colNumber=string.index(columnLine,'^')
    
    print "COLUMN NUMBER:",colNumber
    return errorDescriptionLine,fileName,lineNumber,colNumber,codeLine


# # # class Demo:
    # # # def __init__(self):
        # # # # self.array=[1.0 for i in xrange(50000000)]
        
        # # # from xml.parsers.expat import ExpatError
        # # # import sys
        # # # from os import environ
        # # # import string
        # # # import traceback

        # # # python_module_path=os.environ["PYTHON_MODULE_PATH"]
        # # # appended=sys.path.count(python_module_path)
        # # # if not appended:
            # # # sys.path.append(python_module_path)    
        
        # # # import CompuCell
        # # # CompuCell.initializePlugins()
        # # # # self.simthread = None
        # # # # self.sim = None
        
        
        # # # # sys.path.append(environ["PYTHON_MODULE_PATH"])
        
        
        
        # # # import SystemUtils
        # # # SystemUtils.setSwigPaths()
        # # # SystemUtils.initializeSystemResources()
        
        # # # # # # self.simulator=CompuCell.Simulator()
        # # # import CompuCellSetup
        
        # # # sim,simthread = CompuCellSetup.getCoreSimulationObjects(True) # this only parses xml to extract initial info. No CC3D object is created at this point
        
        # # # self.simulationFileName='d:\\Program Files (x86)\\ps\\Demos\\Models\\cellsort\\cellsort_2D\\Simulation\\cellsort_2D.xml'
        # # # self.cc3dXML2ObjConverter=None
        # # # import XMLUtils
        # # # self.cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
        # # # self.root_element = self.cc3dXML2ObjConverter.Parse(self.simulationFileName)
        # # # print 'root_element=',self.root_element.name


        # # # self.simulator,self.simthread = CompuCellSetup.getCoreSimulationObjects()         
        # # # CompuCellSetup.initializeSimulationObjects(self.simulator,self.simthread)
        
        # # # steppableRegistry = CompuCellSetup.getSteppableRegistry()
        
        # # # CompuCellSetup.mainLoop(self.simulator,self.simthread,steppableRegistry) # main loop - simulation is invoked inside this function
        # # # CompuCellSetup.simulationThreadObject=None
        
        # # # # import weakref
        # # # # sim=CompuCell.Simulator()
        # # # # self.simulator=weakref.ref(sim)
        # # # # # self.simulator=CompuCell.Simulator()
        
        # # # # # CompuCellSetup.initModules(self.simulator,self.cc3dXML2ObjConverter)#extracts Plugins, Steppables and Potts XML elements and passes it to the simulator
        # # # # # self.simulator.initializeCC3D()
        # # # # # self.simulator.extraInit()
        # # # # # self.simulator.start()
        
        
        # # # # self.simArray=[CompuCell.Point3D() for i in xrange(1000000)]    
        # # # # self.simArray=[CompuCell.Simulator() for i in xrange(10000)]    
        
        
# # # demo=Demo()
# # # print 'DEMO CREATED'
# # # import time
# # # time.sleep(3)
# # # # print 'array_len=',len(demo.array)

# # # # demo.simArray=None
# # # demo.simulator.unloadModules()
# # # # # # demo.simulator=None
# # # # demo.simthread.restartManager=None # using weakref takers care of it
# # # demo=None

# # # print 'DEMO DESTROYED'
# # # import time
# # # time.sleep(3)



from xml.parsers.expat import ExpatError
import sys
from os import environ
import string
import traceback

python_module_path=os.environ["PYTHON_MODULE_PATH"]
appended=sys.path.count(python_module_path)
if not appended:
    sys.path.append(python_module_path)    

import CompuCell
CompuCell.initializePlugins()
# simthread = None
# sim = None


# sys.path.append(environ["PYTHON_MODULE_PATH"])



# # # # # # import SystemUtils
# # # # # # SystemUtils.setSwigPaths()
# # # # # # SystemUtils.initializeSystemResources()

# # # # # # # # # simulator=CompuCell.Simulator()
# # # # # # import CompuCellSetup

# # # # # # sim,simthread = CompuCellSetup.getCoreSimulationObjects(True) # this only parses xml to extract initial info. No CC3D object is created at this point

# # # # # # simulationFileName='d:\\Program Files (x86)\\ps\\Demos\\Models\\cellsort\\cellsort_2D\\Simulation\\cellsort_2D.xml'
# # # # # # cc3dXML2ObjConverter=None
# # # # # # import XMLUtils
# # # # # # cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
# # # # # # root_element = cc3dXML2ObjConverter.Parse(simulationFileName)
# # # # # # print 'root_element=',root_element.name


# # # # # # simulator,simthread = CompuCellSetup.getCoreSimulationObjects()         
# # # # # # CompuCellSetup.initializeSimulationObjects(simulator,simthread)

# # # # # # steppableRegistry = CompuCellSetup.getSteppableRegistry()

# # # # # # CompuCellSetup.mainLoop(simulator,simthread,steppableRegistry) # main loop - simulation is invoked inside this function
# # # # # # CompuCellSetup.simulationThreadObject=None

# # # # # # simulator.unloadModules()


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
        
    sim,simthread = CompuCellSetup.getCoreSimulationObjects(True) # this only parses xml to extract initial info. No CC3D object is created at this point
    
    

    if CompuCellSetup.simulationPaths.simulationPythonScriptName != "":
        # fileObj=file(CompuCellSetup.simulationPaths.simulationPythonScriptName,"r")
        # exec fileObj
        # fileObj.close()        
        print "INSIDE IF CompuCellPythonSimulationNewPlayer \n\n\n"
        
        
        execfile(CompuCellSetup.simulationPaths.simulationPythonScriptName)
        print "COMPLETED execfile in CompuCellPythonSimulationNewPlayer \n\n\n"
    else:
        print "INSIDE ELSE CompuCellPythonSimulationNewPlayer \n\n\n"        
        sim,simthread = CompuCellSetup.getCoreSimulationObjects() # here , once initial info has been extracted we starrt creating CC3D objects - e.g. Simulator is created in this Fcn call
        import CompuCell #notice importing CompuCell to main script has to be done after call to getCoreSimulationObjects()
        #import CompuCellSetup
        
        
        CompuCellSetup.initializeSimulationObjects(sim,simthread)
                
        steppableRegistry = CompuCellSetup.getSteppableRegistry()
        
        CompuCellSetup.mainLoop(sim,simthread,steppableRegistry) # main loop - simulation is invoked inside this function
        
        # # # sim,simthread=None,None
        
        # # # print '\n\n\n\n GOT HERE AFTER MAIN LOOP'
        

except IndentationError,e:
    print "CompuCellSetup.simulationObjectsCreated=",CompuCellSetup.simulationObjectsCreated
    if CompuCellSetup.simulationObjectsCreated:        
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Indentation Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except AttributeError,e:
    print "CompuCellSetup.simulationObjectsCreated=",CompuCellSetup.simulationObjectsCreated
    if CompuCellSetup.simulationObjectsCreated:        
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Attribute Error",traceback_message)    
    simthread.handleErrorFormatted(pythonTracebackFormatter())
except SyntaxError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
    # simthread.handleErrorMessage("Python Syntax Error",traceback_message)
except ValueError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Value Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except TypeError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    print 'e.message=',e
    print 'traceback=',traceback
        
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Type Error",traceback_message)    
    simthread.handleErrorFormatted(pythonTracebackFormatter())

except KeyError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Key Error",traceback_message) 
    simthread.handleErrorFormatted(pythonTracebackFormatter())    
    
except IndexError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Index Error",traceback_message)  
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except LookupError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Lookup Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
  
except NameError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Name Error",traceback_message)    
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except FloatingPointError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Floating Point Error",traceback_message) 
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except ZeroDivisionError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Zero Division Error",traceback_message) 
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except OverflowError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Overflow Error",traceback_message)    
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except ArithmeticError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Arithmetic Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())

except EOFError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python EOF Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except IOError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python IO Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())

except OSError,e:
    if CompuCellSetup.simulationObjectsCreated:
        #sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python OS Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except ImportError,e:
    if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # stack=traceback.extract_stack()
    # print "THISIS STACK: ",stack
    # simthread.handleErrorMessage("Python Import Error",traceback_message)
    # errorDescriptionLine,fileName,lineNumber,colNumber,codeLine=pythonErrorFormatter()    
    # simthread.handleErrorMessageDetailed(errorDescriptionLine,fileName,lineNumber,colNumber,codeLine)
    simthread.handleErrorFormatted(pythonTracebackFormatter())

# except BufferError,e:
    # if CompuCellSetup.simulationObjectsCreated:
        # # sim.finish()
        # sim.cleanAfterSimulation()
    # traceback_message=traceback.format_exc()
    # print traceback_message
    # import PlayerPython
    # # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Buffer Error",traceback_message)    

except MemoryError,e:
    if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Memory Error",traceback_message) 
    simthread.handleErrorFormatted(pythonTracebackFormatter())    

except ReferenceError,e:
    if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Reference Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())

except RuntimeError,e:
    if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python Runtime Error",traceback_message)  
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except SystemError,e:
    if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()
        sim.cleanAfterSimulation()
    traceback_message=traceback.format_exc()
    print traceback_message
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    # simthread.handleErrorMessage("Python System Error",traceback_message)  
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except ExpatError,e:
    if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()
        sim.cleanAfterSimulation()
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    xmlFileName=CompuCellSetup.simulationPaths.simulationXMLFileName
    print "Error in XML File","File:\n "+xmlFileName+"\nhas the following problem\n"+e.message
    
    import re
    lineNumberExtractRegex=re.compile('^[\s\S]*line[\s]*([0-9]*)')
    columnNumberExtractRegex=re.compile('^[\s\S]*column[\s]*([0-9]*)')
    
    # simthread.handleErrorMessage("Error in XML File","File:\n "+xmlFileName+"\nhas the following problem\n"+e.message)
    print "SEARCHING MESSAGE=",e.message
    lineNumberGroups=lineNumberExtractRegex.search(str(e.message))
    
    print "\n\n\n lineNumberGroups=",lineNumberGroups
    lineNumber=-1
    colNumber=-1
    try: 
        if lineNumberGroups:
            lineNumber=int(lineNumberGroups.group(1))
            # dbgMsg("Searched text at line=",lineNumber)
            
    except IndexError,e:
            
        print "\n\n\n\n Could not find line number"
    
    columnNumberGroups=columnNumberExtractRegex.search(str(e.message))    
    print "\n\n\n columnNumberGroups=",columnNumberGroups
    try: 
        if columnNumberGroups:
            colNumber=int(columnNumberGroups.group(1))
            
            # dbgMsg("Searched text at line=",lineNumber)
            
    except IndexError,e:
            
        print "\n\n\n\n Could not find column number"
    
    # simthread.handleErrorMessageDetailed("Error in XML File",xmlFileName,lineNumber,colNumber,e.message)
    tracebackMessage="Error: Error in XML File"+"\n"
    tracebackMessage+="  File: "+xmlFileName+"\n"    
    tracebackMessage+="    Line: "+str(lineNumber)+" Col: "+str(colNumber)+" "+e.message
    simthread.handleErrorFormatted(tracebackMessage)

except AssertionError,e:
    if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()
        sim.cleanAfterSimulation()
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    print "Assertion Error: ",e.message
    # simthread.handleErrorMessage("Assertion Error",e.message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())
    
except CompuCellSetup.CC3DCPlusPlusError,e:
    # for C++ exceptions we do not call sim.finish() because modules are unloaded immediately in the exception handling call in the C++
    # print "CompuCellSetup.simulationObjectsCreated=",CompuCellSetup.simulationObjectsCreated
    # if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()    
    print "RUNTIME ERROR IN C++ CODE: ",e.message
    simthread.handleErrorMessage("RUNTIME ERROR IN C++ CODE",e.message)
    # simthread.handleErrorFormatted(pythonTracebackFormatter())

except:
    # if CompuCellSetup.simulationObjectsCreated and CompuCellSetup.playerType=="new":
        # sim.finish()
    if CompuCellSetup.simulationObjectsCreated:
        # sim.finish()       
        sim.cleanAfterSimulation()        
    traceback_message=traceback.format_exc()
    import PlayerPython
    # simthread=PlayerPython.getSimthreadBasePtr()
    print "Unexpected Error:",traceback_message
    # simthread.handleErrorMessage("Unexpected Error",traceback_message)
    simthread.handleErrorFormatted(pythonTracebackFormatter())
