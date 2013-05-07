##@Module rrPython
#This module allows access to the rr_c_api.dll from python"""

import sys
import os
from ctypes import *
from numpy import *

os.chdir(os.path.dirname(__file__))
sharedLib=''
rrLib=None
libHandle=None
if sys.platform.startswith('win32'):
    rrInstallFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bin'))
    os.environ['PATH'] = rrInstallFolder + ';' + "c:\\Python27" + ';' + "c:\\Python27\\Lib\\site-packages" + ';' + os.environ['PATH']
    sharedLib=os.path.join(rrInstallFolder, 'rr_c_api.dll')
    libHandle=windll.kernel32.LoadLibraryA(sharedLib)
    rrLib = WinDLL (None, handle=libHandle)

else:
    rrInstallFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
    sharedLib=os.path.join(rrInstallFolder, 'librr_c_api.so')
    rrLib=cdll.LoadLibrary(sharedLib)


##\mainpage notitle
#\section Introduction
#RoadRunner is a high performance and portable simulation engine for systems and synthetic biology. To run a simple SBML model and generate time series data we would call:
#
#@code
#
#import rrPython
#
#rrPython.loadSBMLFromFile('C:\\Models\\mymodel.xml')
#
#rrPython.simulate()
#@endcode
#
#\section Setup
#In order to import the python module, the python folder within the roadRunner install folder must be in the system's python path. To make sure it is, do the following in Windows:
#
#Open the control panel and click on 'System'
#The following will appear; click on 'Advanced System Settings'
#\image html http://i.imgur.com/bvn9c.jpg
#
#Click on the 'Environment Variables' button highlighted in the image below
#\image html http://i.imgur.com/jBCfn.jpg
#
#Highlight the python path entry and click edit. The prompt shown below will appear. Enter the location of the python folder within the install folder with a semicolon between any other entries.
#\image html http://i.imgur.com/oLC32.jpg

##\defgroup initialization Library initialization and termination methods
# \brief Initialize library and terminate library instance
#
# \defgroup loadsave Read and Write models
# \brief Read and write models to files or strings. Support for SBML formats.
#
# \defgroup utility Utility functions
# \brief Various miscellaneous routines that return useful information about the library
#
# \defgroup errorfunctions Error handling functions
# \brief Error handling routines
#
# \defgroup state Current state of system
# \brief Compute derivatives, fluxes, and other values of the system at the current state
#
# \defgroup steadystate Steady State Routines
# \brief Compute and obtain basic information about the steady state
#
# \defgroup reaction Reaction group
# \brief Get information about reaction rates
#
# \defgroup rateOfChange Rates of change group
# \brief Get information about rates of change
#
# \defgroup boundary Boundary species group
# \brief Get information about boundary species
#
# \defgroup floating Floating species group
# \brief Get information about floating species
#
# \defgroup initialConditions Initial conditions group
# \brief Set or get initial conditions
#
# \defgroup parameters Parameter group
# \brief Set and get global and local parameters
#
# \defgroup compartment Compartment group
# \brief Set and Get information on compartments
#
# \defgroup simulation Time-course simulation
# \brief Deterministic, stochastic, and hybrid simulation algorithms
#
# \defgroup mca Metabolic Control Analysis
# \brief Calculate control coefficients and sensitivities
#
# \defgroup stoich Stoichiometry analysis
# \brief Linear algebra based methods for analyzing a reaction network
#
# \defgroup helperRoutines Helper Routines
# \brief Helper routines for acessing the various C API types, eg lists and arrays
#
# \defgroup toString ToString Routines
# \brief Render various result data types as strings
#
# \defgroup freeRoutines Free memory routines
# \brief Routines that should be used to free various data structures generated during the course of using the library


#=======================rr_c_api=======================#
charptr = POINTER(c_char)

rrLib.createRRInstance.restype = c_void_p

#===== The Python API allocate an internal global handle to ONE instance of the Roadrunner API
gHandle = rrLib.createRRInstance()

# Utility and informational methods
rrLib.getInfo.restype = c_char_p
rrLib.getVersion.restype = c_char_p
rrLib.getBuildDate.restype = c_char_p
rrLib.getBuildTime.restype = c_char_p
rrLib.getBuildDateTime.restype = c_char_p
rrLib.getCopyright.restype = c_char_p
rrLib.setTempFolder.restype = c_bool
rrLib.getTempFolder.restype = c_char_p

rrLib.getStringElement.restype = c_void_p
rrLib.getNumberOfStringElements.restype = c_int
rrLib.getNumberOfStringElements.argtype = [c_void_p]
rrLib.pause.restype = None
#rrLib.getStringElement.argtypes = [POINTER(POINTER(c_ubyte)), c_int]

# More Utility Methods
rrLib.setCapabilities.restype = c_bool
rrLib.getCapabilities.restype = c_char_p
rrLib.setTimeStart.restype = c_bool
rrLib.setTimeEnd.restype = c_bool
rrLib.setNumPoints.restype = c_bool
rrLib.setTimeCourseSelectionList.restype = c_bool
rrLib.oneStep.restype = c_bool
rrLib.getTimeStart.restype = c_bool
rrLib.getTimeEnd.restype = c_bool
rrLib.getNumPoints.restype = c_bool
rrLib.reset.restype = c_bool
rrLib.freeText.restype = c_bool
#rrLib.freeText.argtypes = [c_char_p]

# Set and get family of methods
rrLib.setBoundarySpeciesByIndex.restype = c_bool
rrLib.setFloatingSpeciesByIndex.restype = c_bool
rrLib.setGlobalParameterByIndex.restype = c_bool
rrLib.getBoundarySpeciesByIndex.restype = c_bool
rrLib.getFloatingSpeciesByIndex.restype = c_bool
rrLib.getGlobalParameterByIndex.restype = c_bool
rrLib.getCompartmentByIndex.restype = c_bool
rrLib.setCompartmentByIndex.restype = c_bool
rrLib.getSimulationResult.restype = c_void_p

# Logging
#rrLib.enableLogging.restype = c_bool
rrLib.setLogLevel.restype = c_bool
rrLib.getLogLevel.restype = c_char_p
rrLib.getLogFileName.restype = c_char_p
rrLib.hasError.restype = c_bool
rrLib.getLastError.restype = c_char_p
rrLib.freeRRInstance.restype = c_bool

# Load SBML methods
rrLib.loadSBML.restype = c_bool
rrLib.loadSBMLFromFile.restype = c_bool
rrLib.loadSBMLFromFileJob.restype = c_void_p
rrLib.getCurrentSBML.restype = c_char_p
rrLib.getSBML.restype = c_char_p

# Initial condition methods
rrLib.setFloatingSpeciesInitialConcentrations.restype = c_bool

# Helper routines
rrLib.getVectorLength.restype = c_int
rrLib.getVectorElement.restype = c_bool
rrLib.setVectorElement.restype = c_bool
rrLib.setVectorElement.argtypes = [c_int, c_int, c_double]

rrLib.getMatrixNumRows.restype = c_int
rrLib.getMatrixNumCols.restype = c_int
rrLib.getMatrixElement.restype = c_bool
rrLib.setMatrixElement.restype = c_bool
rrLib.setMatrixElement.argtypes = [c_int, c_int, c_int, c_double]
rrLib.getResultNumRows.restype = c_int
rrLib.getResultNumCols.restype = c_int
rrLib.getResultElement.restype = c_bool
rrLib.getResultColumnLabel.restype = c_char_p
rrLib.getCCodeHeader.restype = c_char_p
rrLib.getCCodeSource.restype = c_char_p

rrLib.isListItemInteger.resType = c_bool
rrLib.isListItemDouble.resType = c_bool
rrLib.isListItemString.resType = c_bool
rrLib.isListItemList.resType = c_bool

# Flags/Options
rrLib.setComputeAndAssignConservationLaws.restype = c_bool

# Steady state methods
rrLib.steadyState.restype = c_bool
rrLib.setSteadyStateSelectionList.restype = c_bool

# State Methods
rrLib.getValue.restype = c_bool
rrLib.setValue.restype = c_bool

# MCA
rrLib.getuCC.restype = c_bool
#rrLib.getuCC.argtypes = [c_void_p, c_char_p, c_char_p, c_double)]
rrLib.getCC.restype = c_bool
rrLib.getEE.restype = c_bool
rrLib.getuEE.restype = c_bool
rrLib.getScaledFloatingSpeciesElasticity.restype = c_bool

# Free memory functions

# Print/format functions
rrLib.resultToString.restype = c_char_p
rrLib.matrixToString.restype = c_char_p
rrLib.vectorToString.restype = c_char_p
rrLib.stringArrayToString.restype = c_char_p
rrLib.listToString.restype = c_char_p
rrLib.getNumberOfStringElementsrestype = c_int

# SBML utility methods
rrLib.getParamPromotedSBML.restype = c_char_p

# Reaction rates
rrLib.getNumberOfReactions.restype = c_int
rrLib.getReactionRate.restype = c_bool

# NOM lib forwarded functions
rrLib.getNumberOfRules.restype = c_int

rrLib.getEigenvalueIds.restype = charptr
rrLib.computeSteadyStateValues.restype = c_void_p
rrLib.getSteadyStateSelectionList.restype = c_void_p
rrLib.getTimeCourseSelectionListrestype = c_void_p
rrLib.simulate.restype = c_void_p
rrLib.simulateJob.restype = c_void_p
rrLib.simulateEx.restype = c_void_p
rrLib.getFloatingSpeciesConcentrations.restype = c_void_p
rrLib.getBoundarySpeciesConcentrations.restype = c_void_p
rrLib.getGlobalParameterValues.restype = c_void_p
rrLib.getFullJacobian.restype = c_void_p
rrLib.getReducedJacobian.restype = c_void_p
rrLib.getEigenvalues.restype = c_void_p
rrLib.getEigenvaluesMatrix.restype = c_int
#rrLib.getEigenvaluesMatrix.argtypes = [c_int]

rrLib.getStoichiometryMatrix.restype = c_void_p
rrLib.getLinkMatrix.restype = c_void_p
rrLib.getNrMatrix.restype = c_void_p
rrLib.getL0Matrix.restype = c_void_p
rrLib.getConservationMatrix.restype = c_void_p
rrLib.getFloatingSpeciesInitialConcentrations.restype = c_void_p
rrLib.getFloatingSpeciesInitialConditionIds.restype = c_void_p
rrLib.getReactionRates.restype = c_void_p
rrLib.getReactionRatesEx.restype = c_void_p
rrLib.getRatesOfChange.restype = c_void_p
rrLib.getRatesOfChangeIds.restype = c_void_p
rrLib.getRatesOfChangeEx.restype = c_void_p
rrLib.getReactionIds.restype = c_void_p
rrLib.getBoundarySpeciesIds.restype = c_void_p
rrLib.getFloatingSpeciesIds.restype = c_void_p
rrLib.getGlobalParameterIds.restype = c_void_p
rrLib.getCompartmentIds.restype = c_void_p
rrLib.getAvailableTimeCourseSymbols.restype = c_void_p
rrLib.getAvailableSteadyStateSymbols.restype = c_void_p
rrLib.getElasticityCoefficientIds.restype = c_void_p
rrLib.getUnscaledFluxControlCoefficientIds.restype = c_void_p
rrLib.getFluxControlCoefficientIds.restype = c_void_p
rrLib.getUnscaledConcentrationControlCoefficientIds.restype = c_void_p
rrLib.getConcentrationControlCoefficientIds.restype = c_void_p
rrLib.getUnscaledElasticityMatrix.restype = c_void_p
rrLib.getScaledElasticityMatrix.restype = c_void_p
rrLib.getUnscaledConcentrationControlCoefficientMatrix.restype = c_void_p
rrLib.getScaledConcentrationControlCoefficientMatrix.restype = c_void_p
rrLib.getUnscaledFluxControlCoefficientMatrix.restype = c_void_p
rrLib.getScaledFluxControlCoefficientMatrix.restype = c_void_p

rrLib.createVector.restype = c_void_p
rrLib.createRRList.restype = c_void_p
rrLib.createIntegerItem.restype = c_void_p
rrLib.createDoubleItem.restype = c_void_p
rrLib.createStringItem.restype = c_void_p
rrLib.createListItem.restype = c_void_p
rrLib.getListItem.restype = c_void_p
rrLib.getList.restype = c_void_p

#Rates of change
rrLib.getRateOfChange.restype = c_bool
rrLib.evalModel.restype = c_bool

#Plugin functionality
rrLib.loadPlugins.restype = c_bool
rrLib.unLoadPlugins.restype = c_bool
rrLib.getNumberOfPlugins.restype = c_int
rrLib.getPluginInfo.restype = c_char_p
rrLib.executePlugin.restype = c_bool

#Job functions
rrLib.isJobFinished.restype = c_bool
rrLib.areJobsFinished.restype = c_bool

#Debugging functions
rrLib.compileSource.restype = c_bool


#Unload roadrunner dll from python
def unloadAPI():
    del gHandle
    return windll.kernel32.FreeLibrary(libHandle)

def freeAPI():
    return windll.kernel32.FreeLibrary(libHandle)

##\ingroup utility
#@{

##\brief Retrieve "general" information about current state of the currently allocated RoadRunner instance, e.g. model
#\return char* info - Returns null if it fails, otherwise it returns the information
def getInfo():
    theInfo = rrLib.getInfo(gHandle)
    theInfo = filter(None, theInfo.split('\n'))
    return theInfo

##\brief Retrieve the current version number of the library
#\return char* version - Returns null if it fails, otherwise it returns the version number of the library
def getVersion():
    return rrLib.getVersion(gHandle)

##\brief Retrieve the current build date of the library
#\return Returns null if it fails, otherwise it returns the build date
def getBuildDate():
    return rrLib.getBuildDate()

##\brief Retrieve the current build time of the library
#\return Returns null if it fails, otherwise it returns the build Time
def getBuildTime():
    return rrLib.getBuildTime()

##\brief Retrieve the current build date + time of the library
#\return Returns null if it fails, otherwise it returns the build date + time
def getBuildDateTime():
    return rrLib.getBuildDateTime()

##\brief Retrieve the current copyright notice for the library
#\return Returns null if it fails, otherwise it returns the copyright string
def getCopyright(aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.getCopyright(aHandle)

##\brief Sets the write location for the temporary file
#
#When cRoadRunner is run in C generation mode its uses a temporary folder to store the
#generated C source code. This method can be used to set the temporary folder path if necessary.
#
#\return Returns true if succcessful
def setTempFolder(folder, aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.setTempFolder(aHandle, folder)

##\brief Returns the full path of the temporary folder
#
#When cRoadRunner is run in C generation mode its uses a temporary folder to store the
#generate C source code. This method can be used to get the current value
#for the the temporary folder path.
#
#\return Returns null if it fails, otherwise it returns the path
def getTempFolder(aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.getTempFolder(aHandle)

##\brief Retrieve a rrLib for the C code structure, RRCCode
#
#When cRoadRunner is run in C generation mode its uses a temporary folder to store the
#generate C source code. This method can be used to obtain the header and main source
#code after a model has been loaded by using the helper routines (getCCodeSource and getCCodeHeader())
#
#\return Returns null if it fails, otherwise it returns a pointer to the RRCode structure
def getCCode():
    return rrLib.getCCode(gHandle)

##@}

##\ingroup errorfunctions
#@{

##\brief Enables logging
#\return Returns true if succesful
def enableLogging():
    return rrLib.enableLoggingToFile(gHandle)

##\brief Set the logging status level
#The logging level is determined by the following strings
#
#"ANY", "DEBUG5", "DEBUG4", "DEBUG3", "DEBUG2", "DEBUG1",
#"DEBUG", "INFO", "WARNING", "ERROR"
#
#Example: setLogLevel ("DEBUG4")
#
#\param lvl The logging level string
#\return Returns true if succesful
def setLogLevel(lvl):
    return rrLib.setLogLevel(lvl)

##\brief Returns the log level as a string
#The logging level can be one of the following strings
#
#"ANY", "DEBUG5", "DEBUG4", "DEBUG3", "DEBUG2", "DEBUG1",
#"DEBUG", "INFO", "WARNING", "ERROR"
#
#Example: str = rrPython.getLogLevel ()
#\return Returns False is it fails else returns the logging string
def getLogLevel():
    return rrLib.getLogLevel()

##\brief Returns the name of the log file
#\return Returns False if it fails else returns the full path to the logging file name
def getLogFileName():
    return rrLib.getLogFileName()

##\brief Check if there is an error string to retrieve
#
#Example: status = rrPython.hasError()
#
#\return status - Returns true if there is an error waiting to be retrieved
def hasError():
    return rrLib.hasError()

##\brief Returns the last error
#
#Example: str = rrPython.getLastError()
#
#\return Returns false if it fails, otherwise returns the error string
def getLastError():
    return rrLib.getLastError()

def getInstanceCount(rrHandles):
    return rrLib.getInstanceCount(rrHandles)

def getRRHandle(rrHandles, index):
    return rrLib.getRRHandle(rrHandles, c_int(index))

##\brief Initialize the roadRunner library and returns a new RoadRunner instance
#\return Returns an instance of the library, returns false if it fails
def createRRInstance():
    return rrLib.createRRInstance()

def createRRInstances(nrOfInstances):
    return rrLib.createRRInstances(c_int(nrOfInstances))

#    aList = []
    #Put each instance into a python list???
#    if rrs is not None:
#        for i in range(nrOfInstances):
#            aList.append(getRRHandle(rrs, i))
#    return aList

##\brief Free the roadRunner instance
#\param rrLib Free the roadRunner instance given in the argument
def freeRRInstance(iHandle):
    return rrLib.freeRRInstance(iHandle)

##\brief Free roadRunner instances
#\param rrLib Frees roadRunner instances
def freeRRInstances(rrInstances):
    return rrLib.freeRRInstances(rrInstances)

##@}

##\ingroup utility
#@{

##\brief Enable/disable conservation analysis
#\param OnOrOff Set to 1 to switch on conservation analysis, 0 to switch it off
#\return Returns True if successful
def setComputeAndAssignConservationLaws(OnOrOff, aHandle=None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.setComputeAndAssignConservationLaws(aHandle,  c_bool(OnOrOff))

##@}

##\ingroup loadsave
#@{

##\brief Create a model from an SBML string
#\param[in] sbml string
#\return Returns true if successful
def loadSBML(sbml, aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.loadSBML(aHandle, sbml)

##\brief Loads SBML model from a file
#\param fileName file name
#\return Returns true if successful
def loadSBMLFromFile(fileName, aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.loadSBMLFromFile(aHandle, fileName)

##\brief Loads SBML model from a file in a thread
#\param fileName file name
#\return Returns true if successful
def loadSBMLFromFileJob(fileName, aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.loadSBMLFromFileJob(aHandle, fileName)

def waitForJob(jobHandle):
    return rrLib.waitForJob(jobHandle)

def waitForJobs(jobsHandle):
    return rrLib.waitForJobs(jobsHandle)

##\brief Loads SBML model from a file into a list of roadrunner instances, using a thread pool
#\param fileName file name
#\return Returns true if successful
def loadSBMLFromFileJobs(rrs, fileName, threadCount = 4):
    return rrLib.loadSBMLFromFileJobs(rrs, fileName, threadCount)

##\brief Return the current state of the model in the form of an SBML string
#\return Returns False if it fails or no model is loaded, otherwise returns the SBML string.
def getCurrentSBML(aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.getCurrentSBML(aHandle)

##\brief Retrieve the last SBML model that was loaded
#\return Returns False if it fails or no model is loaded, otherwise returns the SBML string
def getSBML(aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.getSBML(aHandle)

##@}


##\ingroup parameters
#@{

##\brief Promote any local parameters to global status
#\param sArg The string containing SBML model to promote
#\return Returns False if it fails, otherwise it returns the promoted SBML string
def getParamPromotedSBML(sArg, aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    value = c_char(sArg)
    if rrLib.getParamPromotedSBML(aHandle, pointer(value)) == True:
        return value.value
    else:
        raise RuntimeError('Index out of range')

##@}


##\ingroup simulation
#@{

##\brief Sets simulator capabilities
#\param[out] caps An XML string that specifies the simulators capabilities
#\return Returns true if successful
def setCapabilities(caps):
    return rrLib.setCapabilities(gHandle, caps)

##\brief Returns simulator capabilities
#\return Returns False if it fails, otherwise returns the simulator's capabilities in the form of an XML string
def getCapabilities():
    return rrLib.getCapabilities(gHandle)

##\brief Sets the start time for the simulation
#\param timeStart
#\return Returns True if successful
def setTimeStart(timeStart, rrHandle = None):
    if rrHandle is None:
        rrHandle = gHandle
    return rrLib.setTimeStart (rrHandle, c_double(timeStart))

##\brief Sets the end time for the simulation
#\param timeEnd
#\return Returns True if successful
def setTimeEnd(timeEnd, rrHandle = None):
    if rrHandle is None:
        rrHandle = gHandle
    return rrLib.setTimeEnd(rrHandle, c_double(timeEnd))

##\brief Set the number of points to generate in a simulation
#\param numPoints Number of points to generate
#\return Returns True if successful
def setNumPoints(numPoints, rrHandle = None):
    if rrHandle is None:
        rrHandle = gHandle
    return rrLib.setNumPoints(rrHandle, numPoints)

##\brief Sets the list of variables returned by simulate() or simulateEx()
#
#Example: rrPython.setTimeCourseSelectionList ("Time, S1, J1, J2")
#
#\param list A string of Ids separated by spaces or comma characters
#\return Returns True if successful
def setTimeCourseSelectionList(myList, rrHandle = None):
    if rrHandle is None:
        rrHandle = gHandle

    if type (myList) == str:
       return rrLib.setTimeCourseSelectionList(rrHandle, myList)
    if type (myList) == list:
        strList = ''
        for i in range (len(myList)):
            strList = strList + myList[i] + ' '
        return rrLib.setTimeCourseSelectionList(rrHandle, strList)
    raise RuntimeError('Expecting string or list in setTimeCourseSelectionList')


##\brief Returns the list of variables returned by simulate() or simulateEx()
#\return A list of symbol IDs indicating the currect selection list
def getTimeCourseSelectionList():
    value = rrLib.getTimeCourseSelectionList(gHandle)
    return stringArrayToList (value)

##\brief Carry out a time-course simulation, use setTimeStart etc to set
#characteristics
#\return Returns a string containing the results of the simulation organized in rows and columns
def simulate(aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    result = rrLib.simulate(aHandle)
    #TODO: Check result
    rowCount = rrLib.getResultNumRows(result)
    colCount = rrLib.getResultNumCols(result)
    resultArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                rvalue = m
                cvalue = n
                value = c_double()
                if rrLib.getResultElement(result, rvalue, cvalue, pointer(value)) == True:
                    resultArray[m, n] = value.value
    rrLib.freeResult(result)
    return resultArray

##\brief Carry out a time-course simulation in a thread, use setTimeStart etc to set
#characteristics
#\return Returns a handle to the thread. Use this handle to see when the thread has finished
def simulateJob(aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.simulateJob(aHandle)


##\brief Check if a job is done
#characteristics
#\return Returns true/false indicating if a job has finsished
def isJobFinished(aHandle = None):
    if aHandle is None:
        aHandle = gHandle
    return rrLib.isJobFinished(aHandle)

##\brief Carry out a time-course simulation for a thread pool
#characteristics
#\return Returns a handle to Jobs. Use this handle to see when the jobs have finished
def simulateJobs(rrsHandle, nrOfThreads):
    return rrLib.simulateJobs(rrsHandle, nrOfThreads)

def writeRRData(outFile, rrInstanceList=None):
    if rrInstanceList is not None:
        rrLib.writeMultipleRRData(rrInstanceList, outFile)
    else:
        rrLib.writeRRData(gHandle, outFile)


def getSimulationResult(aHandle = None):
    if aHandle is None:
        aHandle = gHandle

    result = rrLib.getSimulationResult(aHandle)

    #TODO: Check result
    rowCount = rrLib.getResultNumRows(result)
    colCount = rrLib.getResultNumCols(result)
    resultArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                rvalue = m
                cvalue = n
                value = c_double()
                if rrLib.getResultElement(result, rvalue, cvalue, pointer(value)) == True:
                    resultArray[m, n] = value.value
    rrLib.freeResult(result)
    return resultArray

#use getResultElement and other helper routines to build array that can be used in numpy to plot with matplotlib
#get num cols, get num rows, create array, fill array with two loops


##\brief Carry out a time-course simulation based on the given arguments
#
#Example: m = rrPython.simulateEx(0, 25, 200)
#
#\return Returns a string containing the results of the simulation organized in rows and columns
def simulateEx(timeStart, timeEnd, numberOfPoints):
    startValue = c_double(timeStart)
    endValue = c_double(timeEnd)
    nrPoints = c_int(numberOfPoints)
    result = rrLib.simulateEx(gHandle, startValue, endValue, nrPoints)
    #TODO: Check result
    rowCount = rrLib.getResultNumRows(result)
    colCount = rrLib.getResultNumCols(result)
    resultArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getResultElement(result, rvalue, cvalue, pointer(value)) == True:
                    resultArray[m, n] = value.value
    rrLib.freeResult(result)
    return resultArray;

##\brief Carry out a one step integration of the model
#
#Example: status = rrPython.oneStep(currentTime, stepSize)
#
#\param[in] currentTime The current time in the simulation
#\param[in] stepSize The step size to use in the integration
#\param[in] value The new time (currentTime + stepSize)
#Takes (double, double) as an argument
#\return
def oneStep (currentTime, stepSize):                             #test this
    curtime = c_double(currentTime)
    stepValue = c_double(stepSize)
    value = c_double()
    if rrLib.oneStep(gHandle, (curtime), (stepValue), pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##\brief Returns the simulation start time
#
#Example: status = rrPython.getTimeStart()
#
#\return Returns the simulation start time as a float
def getTimeStart():
    value = c_double()
    if rrLib.getTimeStart(gHandle, pointer(value)) == True:
        return value.value
    else:
        return ('Index out of Range')

##\brief Returns the simulation end time
#
#Example: status = rrPython.getTimeEnd()
#
#\return Returns the simulation end Time as a float
def getTimeEnd():
    value = c_double()
    if rrLib.getTimeEnd(gHandle, pointer(value)) == True:
        return value.value
    else:
        return ('Index out of Range')

##\brief Returns the value of the current number of points
#
#Example: status = rrPython.getNumPoints()
#
#\return Returns the value of the number of points
def getNumPoints():
    value = c_int()
    if rrLib.getNumPoints(gHandle, pointer(value)) == True:
        return value.value
    else:
        return ('Index out of Range')

##\brief Reset all floating species concentrations to their intial conditions
#
#Example: status = rrPython.reset()
#
#\return Returns True if successful
def reset():
    return rrLib.reset(gHandle)

##@}

##\ingroup steadystate
#@{

##\brief Computes the steady state of the loaded model
#
#Example: status = rrPython.steadyState()
#
#\return Returns a value that is set during the call that indicates how close the solution is to the steady state. The smaller the value, the better.
def steadyState():
    value = c_double()
    if rrLib.steadyState(gHandle, pointer(value)) == True:
        return value.value
    else:
        return (GetLastError(gHandle))

##\brief A convenient method for returning a vector of the steady state species concentrations
#
#Example: values = rrPython.computeSteadyStateValues()
#
#\return Returns the vector of steady state values or NONE if an error occurred.
def computeSteadyStateValues():
    values = rrLib.computeSteadyStateValues(gHandle)
    if values == None:
       raise RuntimeError(getLastError())
    return rrVectorToPythonArray (values)

##\brief Set the selection list of the steady state analysis
#
#param[in] list The string argument should be a space-separated list of symbols in the selection list
#
#\return Returns True if successful
def setSteadyStateSelectionList(aList):
    if type (aList) == str:
       value = c_char_p(aList)
       return rrLib.setSteadyStateSelectionList(gHandle, value)
    if type (aList) == list:
        strList = ''
        for i in range (len(aList)):
            strList = strList + aList[i] + ' '
        value = c_char_p (strList)
        return rrLib.setSteadyStateSelectionList(gHandle, strList)
    raise RuntimeError('Expecting string or list in setTimeCourseSelectionList')



##\brief Get the selection list for the steady state analysis
#\return Returns False if it fails, otherwise it returns a list of strings representing symbols in the selection list
def getSteadyStateSelectionList():
    value = rrLib.getSteadyStateSelectionList(gHandle)
    return stringArrayToList (value)


##@}

##\ingroup state
#@{

##\brief Get the value for a given symbol, use getAvailableSymbols() for a list of symbols
#
#Example: status = rrPython.getValue("S1")
#
#\param symbolId The symbol that we wish to obtain the value for
#\return Returns the value if successful, otherwise returns False
def getValue(symbolId, rrHandle = None):
    if rrHandle is None:
        rrHandle = gHandle

    value = c_double()
    if rrLib.getValue(rrHandle, symbolId, pointer(value)) == True:
        return value.value
    else:
        raise RuntimeError(getLastError() + ': ' + symbolId)

##\brief Set the value for a given symbol, use getAvailableSymbols() for a list of symbols
#
#Example: status = rrPython.setValue("S1", 0.5)
#
#\param symbolId The symbol that we wish to set the value for
#\param value The value that the symbol will be set to
#\return Returns True if successful
def setValue(symbolId, value, rrHandle = None):
    if rrHandle is None:
        rrHandle = gHandle

    if rrLib.setValue(rrHandle, symbolId, c_double(value)) == True:
        return True
    else:
        raise RuntimeError('Index out of range')

##@}

##\ingroup floating
#@{

##\brief Retrieve a string containing concentrations for all the floating species
#
#Example: values = rrPython.getFloatingSpeciesConcentrations()
#
#\return Returns a string of floating species concentrations or None if an error occured
def getFloatingSpeciesConcentrations():
    values = rrLib.getFloatingSpeciesConcentrations(gHandle)
    return rrVectorToPythonArray (values)

##\brief Sets the concentration for a floating species by its index. Species are indexed starting at 0.
#
#Example: rrPython.setFloatingSpeciesByIndex(0, .5)
#
#\param index The index to the floating species (corresponds to position in getFloatingSpeciesIds()) starting at 0
#\param value The concentration of the species to set
#\return Returns True if successful
def setFloatingSpeciesByIndex(index, value):
    return rrLib.setFloatingSpeciesByIndex(gHandle, c_int(index), c_double(value))

##\brief Returns the concentration of a floating species by its index. Species are indexed starting at 0.
#
#Example: value = rrPython.getFloatingSpeciesByIndex()
#
#\param index The index to the floating species (corresponds to position in getFloatingSpeciesIds()) starting at 0
#\return Returns the concentration of the species if successful
def getFloatingSpeciesByIndex(index):
    value = c_double()
    if rrLib.getFloatingSpeciesByIndex(gHandle, c_int(index), pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##\brief Set the floating species concentration to the vector
#
#Example: status = rrPython.setFloatingSpeciesConcentrations (myArray)
#
#\param myArray A numPy array of floating species concentrations
#\return Returns True if successful
def setFloatingSpeciesConcentrations(myArray):
    return rrLib.setFloatingSpeciesConcentrations(gHandle, PythonArrayTorrVector (myArray))

##@}

##\ingroup boundary
#@{

##\brief Sets the concentration for a Boundary species by its index. Species are indexed starting at 0.
#
#Example: rrPython.setBoundarySpeciesByIndex(0, .5)
#
#\param index The index to the boundary species (corresponds to position in getBoundarySpeciesIds()) starting at 0
#\param value The concentration of the species to set
#\return Returns True if successful
def setBoundarySpeciesByIndex(index, value):
    if rrLib.setBoundarySpeciesByIndex(gHandle, c_int(index), c_double(value)) == True:
        return True
    else:
        raise RuntimeError('Index out of range')

##\brief Returns the concentration of a boundary species by its index. Species are indexed starting at 0.
#
#Example: value = rrPython.getBoundarySpeciesByIndex()
#
#\param index The index to the Boundary species (corresponds to position in getBoundarySpeciesIds()) starting at 0
#\return Returns the concentration of the species if successful
def getBoundarySpeciesByIndex(index):
    value = c_double()
    if rrLib.getBoundarySpeciesByIndex(gHandle, (index), pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##\brief Set the boundary species concentration to the vector
#\param vec A vector of boundary species concentrations
#\return Returns True if successful
def setBoundarySpeciesConcentrations(vector):
    return rrLib.setBoundarySpeciesConcentrations(gHandle, vector)

##\brief Returns a string with boundary species concentrations
#\return Returns the concentration of species if successful
def getBoundarySpeciesConcentrations():
    values = rrLib.getBoundarySpeciesConcentrations(gHandle)
    return rrVectorToPythonArray (values)

##@}

##\ingroup parameters
#@{

##\brief Retrieve a string containing values for all global parameters
#
#Example: values = rrPython.getGlobalParameterValues()
#
#\return Returns a string of global parameter values or None if an error occured
def getGlobalParameterValues():
    values = rrLib.getGlobalParameterValues(gHandle)
    return rrVectorToPythonArray (values)

##\brief Sets the value for a global parameter by its index. Parameters are indexed starting at 0.
#
#Example: rrPython.setGlobalParameterByIndex(0, .5)
#
#\param index The index to the global parameter (corresponds to position in getGlobalParameterIds()) starting at 0
#\param value The value of the global parameter to set
#\return Returns True if successful
def setGlobalParameterByIndex(index, value):
    if rrLib.setGlobalParameterByIndex(gHandle, c_int(index), c_double(value)) == True:
        return True;
    else:
        raise RuntimeError('Index out of range')

##\brief Returns the concentration of a global parameter by its index. Parameters are indexed starting at 0.
#
#Example: value = rrPython.getGlobalParameterByIndex()
#
#\param index The index to the global parameter (corresponds to position in getGlobalParameterIds()) starting at 0
#\return Returns the value of the global parameter if successful
def getGlobalParameterByIndex(index):
    value = c_double()
    if rrLib.getGlobalParameterByIndex(gHandle, c_int(index), pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of Range')

##@}

##/ingroup compartment
#@{

##\brief Returns the volume of a compartment by its index. Compartments are indexed starting at 0.
#
#Example: value = rrPython.getCompartmentByIndex()
#
#\param index The index to the compartment (corresponds to position in getCompartmentIds()) starting at 0
#\return Returns the volume of the compartment if successful
def getCompartmentByIndex(index):
    value = c_double()
    if rrLib.getCompartmentByIndex(gHandle, c_int(index), pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of Range')

##\brief Sets the value for a compartment by its index. Compartments are indexed starting at 0.
#
#Example: rrPython.setCompartmentByIndex(0, .5)
#
#\param index The index to the compartment (corresponds to position in getCompartmentIds()) starting at 0
#\param value The volume of the compartment to set
#\return Returns True if Successful
def setCompartmentByIndex(index, value):
    if rrLib.setCompartmentByIndex(gHandle, c_int(index), c_double(value)) == True:
        return True
    else:
        raise RuntimeError('Index out of range')

##@}

##\ingroup stoich
#@{

##\brief Retrieve the full Jacobian for the current model
#\return Returns the full Jacobian matrix
def getFullJacobian():
    matrix = rrLib.getFullJacobian(gHandle)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
    result = rrLib.matrixToString(matrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
            value = c_double()
            rvalue = m
            cvalue = n
            if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
               matrixArray[m, n] = value.value
    rrLib.freeMatrix(matrix)
    return matrixArray

##\brief Retreive the reduced Jacobian for the current model
#\return Returns the reduced Jacobian matrix
def getReducedJacobian():
    matrix = rrLib.getReducedJacobian(gHandle)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
                    matrixArray[m, n] = value.value
    return matrixArray

def getEigenvaluesMatrix (m):
    rrm = createRRMatrix (m)
    matrix = rrLib.getEigenvaluesMatrix (rrm)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
                    matrixArray[m, n] = value.value
    return matrixArray


##\brief Retreive the eigenvalue matrix for the current model
#\return Returns a matrix of eigenvalues. The first column will contain the real values and te second column will contain the imaginary values.
def getEigenvalues():
    matrix = rrLib.getEigenvalues(gHandle)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
#    result = rrLib.matrixToString(matrix)
#    print c_char_p(result).value
#    rrLib.freeText(result)

    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
                    matrixArray[m, n] = value.value

    rrLib.freeMatrix(matrix)
    return matrixArray

##\brief Retreive the stoichiometry matrix for the current model
#\return Returns the stoichiometry matrix
def getStoichiometryMatrix():
    matrix = rrLib.getStoichiometryMatrix(gHandle)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
                    matrixArray[m, n] = value.value
    rrLib.freeMatrix(matrix)
    return matrixArray


##\brief Retreive the Link matrix for the current model
#\return Returns the Link matrix
def getLinkMatrix():
    matrix = rrLib.getLinkMatrix(gHandle)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
    #result = rrLib.matrixToString(matrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
                    matrixArray[m, n] = value.value
    rrLib.freeMatrix(matrix)
    return matrixArray

##\brief Retrieve the reduced stoichiometry matrix for the current model
#\return Returns the reduced stoichiometry matrix
def getNrMatrix():
    matrix = rrLib.getNrMatrix(gHandle)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
    result = rrLib.matrixToString(matrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
                    matrixArray[m, n] = value.value
    rrLib.freeMatrix(result)
    return matrixArray


##\brief Retrieve the L0 matrix for the current model
#\return Returns the L0 matrix
def getL0Matrix():
    matrix = rrLib.getL0Matrix(gHandle)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
    result = rrLib.matrixToString(matrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
                    matrixArray[m, n] = value.value
    rrLib.freeMatrix(result)
    return matrixArray

##\brief Retrieve the conservation matrix for the current model
#\return Returns the conservation matrix
def getConservationMatrix():
    matrix = rrLib.getConservationMatrix(gHandle)
    if matrix == 0:
       return 0
    rowCount = rrLib.getMatrixNumRows(matrix)
    colCount = rrLib.getMatrixNumCols(matrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
                value = c_double()
                rvalue = m
                cvalue = n
                if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
                    matrixArray[m, n] = value.value
    rrLib.freeMatrix(matrix)
    return matrixArray

##@}


##\addtogroup initialConditions
#@{

##\brief Set the initial floating species concentrations
#
#Example: status = rrPython.setFloatingSpeciesInitialConcentrations(myArray)
#
#\param myArray A numPy array of species concentrations: order given by getFloatingSpeciesIds
#\return Returns True if successful
def setFloatingSpeciesInitialConcentrations(myArray):
    return rrLib.setFloatingSpeciesInitialConcentrations (gHandle, PythonArrayTorrVector (myArray))


##\brief Get the initial floating species concentrations
#
#Example: vec = rrPython.getFloatingSpeciesInitialConcentrations()
#
#\return Returns a string containing the intial concentrations
def getFloatingSpeciesInitialConcentrations():
    values = rrLib.getFloatingSpeciesInitialConcentrations(gHandle)
    return rrVectorToPythonArray (values)

##\brief Get the initial floating species Ids
#
#Example: vec = rrPython.getFloatingSpeciesInitialConditionIds()
#
#\return Returns a string containing the initial conditions
def getFloatingSpeciesInitialConditionIds():
    values = rrLib.getFloatingSpeciesInitialConditionIds(gHandle)
    return stringArrayToList (values)

##@}


##\ingroup reaction
#@{

##\brief Obtain the number of reactions in the loaded model
#
#Example: number = rrPython.getNumberOfReactions()
#
#\return Returns -1 if it fails, returns 0 or more if it is successful (indicating the number of reactions)
def getNumberOfReactions():
    return rrLib.getNumberOfReactions(gHandle)

##\brief Returns the reaction rate by index
#\return Returns the reaction rate
def getReactionRate(index):
    value = c_double()
    if rrLib.getReactionRate(gHandle, c_int(index), pointer(value)) == True:
        return value.value
    else:
        raise RuntimeError(getLastError())

##\brief Returns a string containing the current reaction rates
#\return Returns a string containing the current reaction rates
def getReactionRates():
    values = rrLib.getReactionRates(gHandle)
    return rrVectorToPythonArray (values)


##\brief Retrieve a numPy array containing the reaction rates given a vector of species concentrations
#
#\param myArray The numPy array of floating species concentrations
#\return Returns a numPy array containing reaction rates
def getReactionRatesEx(myArray):
    values = rrLib.getReactionRatesEx(gHandle, PythonArrayTorrVector (myArray))
    return rrVectorToPythonArray (values)

##@}

##\ingroup rateOfChange
#@{

##\brief Returns the rates of change in a string
#
#Example: values = rrPython.getRatesOfChange
#
#\return Returns a numPy array containing rates of change values
def getRatesOfChange():
    values = rrLib.getRatesOfChange(gHandle)
    return rrVectorToPythonArray (values)

##\brief Retrieve the string list of rates of change Ids
#
#Example: Ids = rrPython.getRatesOfChangeIds
#
#\return Returns a list of rates of change Ids
def getRatesOfChangeIds():
    values = rrLib.getRatesOfChangeIds(gHandle)
    return stringArrayToList (value)

##\brief Retrieve the rate of change for a given floating species by its index. Species are indexed starting at 0
#
#Example: value = rrPython.getRateOfChange(0)
#
#\param Index to the rate of change item
#\return Returns the rate of change of the ith floating species, otherwise it raises an exception
def getRateOfChange(index):
    value = c_double()
    if rrLib.getRateOfChange(gHandle, c_int(index), pointer(value)) == True:
        return value.value
    else:
        raise RuntimeError(getlastError())


##\brief Retrieve the vector of rates of change in a string given a vector of floating species concentrations
#
#Example: values = rrPython.getRatesOfChangeEx (myArray)
#
#\param myArray A numPy array of species concentrations: order given by getFloatingSpeciesIds
#\return Returns a string containing a vector with the rates of change
def getRatesOfChangeEx(myArray):
    values = rrLib.getRatesOfChangeEx(gHandle, PythonArrayTorrVector (myArray))
    return rrVectorToPythonArray (values)

##@}

##\ingroup state
#@{

##\brief Evaluate the current model, which updates all assignments and rates of change
#\return Returns False if it fails
def evalModel():
    return rrLib.evalModel(gHandle)

##@}

#Get number family
rrLib.getNumberOfCompartments.restype = c_int
rrLib.getNumberOfBoundarySpecies.restype = c_int
rrLib.getNumberOfFloatingSpecies.restype = c_int
rrLib.getNumberOfGlobalParameters.restype = c_int
rrLib.getNumberOfDependentSpecies.restype = c_int
rrLib.getNumberOfIndependentSpecies.restype = c_int

##\ingroup floating
#@{

##\brief Returns the number of floating species in the model
#\return Returns the number of floating species in the model
def getNumberOfFloatingSpecies():
    return rrLib.getNumberOfFloatingSpecies(gHandle)

##\brief Returns the number of dependent species in the model
#\return Returns the number of dependent species in the model
def getNumberOfDependentSpecies():
    return rrLib.getNumberOfDependentSpecies(gHandle)

##\brief Returns the number of independent species in the model
#\return Returns the number of independent species in the model
def getNumberOfIndependentSpecies():
    return rrLib.getNumberOfIndependentSpecies(gHandle)

##@}

##\ingroup compartment
#@{

##\brief Returns the number of compartments in the model
#\return Returns the number of compartments in the model
def getNumberOfCompartments():
    return rrLib.getNumberOfCompartments(gHandle)

##@}

##\ingroup boundary

##\brief Returns the number of boundary species in the model
#\return Returns the number of boundary species in the model
def getNumberOfBoundarySpecies():
    return rrLib.getNumberOfBoundarySpecies(gHandle)

##@}

##\ingroup parameters
#@{

##\brief Returns the number of global parameters in the model
#\return Returns the number of global parameters in the model
def getNumberOfGlobalParameters():
    return rrLib.getNumberOfGlobalParameters(gHandle)

##@}


#Get Ids family

##\addtogroup reaction
#@{

##\brief Returns a list of reaction Ids
#\return Returns a string containing a list of reaction Ids
def getReactionIds():
    value = rrLib.getReactionIds(gHandle)
    return stringArrayToList (value)

##\brief Returns a string containing the list of rate of change Ids
#\return Returns a string containing the list of rate of change Ids
def getRatesOfChangeIds():
    value = rrLib.getRatesOfChangeIds(gHandle)
    return stringArrayToList (value)
##@}

##\ingroup compartment
#@{

##\brief Gets the list of compartment Ids
#\return Returns -1 if it fails, otherwise returns a string containing the list of compartment Ids
def getCompartmentIds():
    value = rrLib.getCompartmentIds(gHandle)
    return stringArrayToList (value)

##@}

##\ingroup boundary
#@{

##\brief Gets the list of boundary species Ids
#\return Returns a string containing the list of boundary species Ids
def getBoundarySpeciesIds():
    value = rrLib.getBoundarySpeciesIds(gHandle)
    return stringArrayToList (value)

##@}

##\ingroup floating
#@{

##\brief Gets the list of floating species Ids
#\return Returns a string containing the list of floating species Ids
def getFloatingSpeciesIds():
    value = rrLib.getFloatingSpeciesIds(gHandle)
    return stringArrayToList (value)

##@}

##\ingroup parameters
#@{

##\brief Gets the list of global parameter Ids
#\return Returns a string containing the list of global parameter Ids
def getGlobalParameterIds():
    value = rrLib.getGlobalParameterIds(gHandle)
    return stringArrayToList (value)

##@}

##\ingroup state
#@{

##\brief Returns the Ids of all floating species eigenvalues
#\return Returns a string containing the list of all floating species eigenvalues
def getEigenvalueIds():
    values = rrLib.getEigenvalueIds(gHandle)
    return stringArrayToList (values)

##\brief Returns a string containing the list of all steady state simulation variables
#\return Returns a string containing the list of all steady state simulation variables
def getAvailableSteadyStateSymbols():
    value = rrLib.getAvailableSteadyStateSymbols(gHandle)
    result = rrLib.listToString(value)
    return result

##\brief Returns a string containing the list of all time course simulation variables
#\return Returns a string containing the list of all time course simulation variables
def getAvailableTimeCourseSymbols():
    value = rrLib.getAvailableTimeCourseSymbols(gHandle)
    result = rrLib.listToString(value)
    return result

##@}

#Get MCA methods

##\addtogroup mca
#@{

##\brief Returns the Ids of all elasticity coefficients
#\return Returns a string containing the list of elasticity coefficient Ids
def getElasticityCoefficientIds():
    value = rrLib.getElasticityCoefficientIds(gHandle)
    result = rrLib.listToString(value)
    return result

##\brief Returns the Ids of all unscaled flux control coefficients
#\return Returns a string containing the list of all unscaled flux control coefficient Ids
def getUnscaledFluxControlCoefficientIds():
    value = rrLib.getUnscaledFluxControlCoefficientIds(gHandle)
    result = rrLib.listToString(value)
    return result

##\brief Returns the Ids of all flux control coefficients
#\return Returns a string containing the list of all flux control coefficient Ids
def getFluxControlCoefficientIds():
    value = rrLib.getFluxControlCoefficientIds(gHandle)
    result = rrLib.listToString(value)
    return result

##\brief Returns the Ids of all unscaled concentration control coefficients
#\return Returns a string containing the list of all unscaled concentration coefficient Ids
def getUnscaledConcentrationControlCoefficientIds():
    value = rrLib.getUnscaledConcentrationControlCoefficientIds(gHandle)
    result = rrLib.listToString(value)
    return result

##\brief Returns the Ids of all concentration control coefficients
#\return Returns a string containing the list of all concentration control coefficient Ids
def getConcentrationControlCoefficientIds():
    value = rrLib.getConcentrationControlCoefficientIds(gHandle)
    result = rrLib.listToString(value)
    return result

##\brief  Retrieve the unscaled elasticity matrix for the current model
#\return Returns a string containing the matrix of unscaled elasticities. The first column will contain the
#real values and the second column the imaginary values.
def getUnscaledElasticityMatrix():
    value = rrLib.getUnscaledElasticityMatrix(gHandle)
    m = createMatrix (value)
    rrLib.freeMatrix(value)
    return m

##\brief Retrieve the scaled elasticity matrix for the current model
#\return Returns a string containing the matrix of scaled elasticities. The first column will contain
#real values and the second column the imaginary values.
def getScaledElasticityMatrix():
    value = rrLib.getScaledElasticityMatrix(gHandle)
    m = createMatrix (value)
    rrLib.freeMatrix(value)
    return m

##\brief Retrieve the unscaled concentration control coefficient matrix for the current model
#\return Returns a string containing the matrix of unscaled concentration control coefficients. The first column will contain
#real values and the second column the imaginary values.
def getUnscaledConcentrationControlCoefficientMatrix():
    value = rrLib.getUnscaledConcentrationControlCoefficientMatrix(gHandle)
    m = createMatrix (value)
    rrLib.freeMatrix(value)
    return m

##\brief Retrieve the scaled concentration control coefficient matrix for the current model
#\return Returns a string containing the matrix of scaled concentration control coefficients. The first column will contain
#real values and the second column the imaginary values.
def getScaledConcentrationControlCoefficientMatrix():
    value = rrLib.getScaledConcentrationControlCoefficientMatrix(gHandle)
    m = createMatrix (value)
    rrLib.freeMatrix(value)
    return m

##\brief Retrieve the unscaled flux control coefficient matrix for the current model
#\return Returns a string containing the matrix of unscaled flux control coefficients. The first column will contain
#real values and the second column the imaginary values.
def getUnscaledFluxControlCoefficientMatrix():
    value = rrLib.getUnscaledFluxControlCoefficientMatrix(gHandle)
    m = createMatrix (value)
    rrLib.freeMatrix(value)
    return m

##\brief Retrieve the scaled flux control coefficient matrix for the current model
#\return Returns a string containing the matrix of scaled flux control coefficients. The first column will contain
#real values and the second column the imaginary values.
def getScaledFluxControlCoefficientMatrix():
    value = rrLib.getScaledFluxControlCoefficientMatrix(gHandle)
    m = createMatrix (value)
    rrLib.freeMatrix(value)
    return m

##\brief Get unscaled control coefficient with respect to a global parameter
#
#Takes (variableName, parameterName) as an argument, where both arguments are strings
#\return
def getuCC(variable, parameter):
    variable = c_char_p(variable)
    parameter = c_char_p(parameter)
    value = c_double()
    if rrLib.getuCC(gHandle, variable, parameter, pointer(value)) == True:
        return value.value;
    else:
        errStr = getLastError()
        raise RuntimeError(errStr)

##\brief Retireve a single control coefficient
#
#\param[in] variable This is the dependent variable of the coefficient, for example a flux or species concentration
#\param[in] parameter This is the independent parameter, for example a kinetic constant or boundary species
#\param[out] value This is the value of the control coefficeint returned to the caller
#
#\return Returns a single control coefficient if successful
def getCC(variable, parameter):
    value = c_double()
    if rrLib.getCC(gHandle, variable, parameter, pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##\brief Retireve a single elasticity coefficient
#
#\param[in] variable This is the dependent variable of the coefficient, for example a flux or species concentration
#\param[in] parameter This is the independent parameter, for example a kinetic constant or boundary species
#\param[out] value This is the value of the control coefficeint returned to the caller
#\return Returns a single elasticity coefficient if successful
def getEE(variable, parameter):
    value = c_double()
    if rrLib.getEE(gHandle, variable, parameter,  pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##\brief Retrieve a single unscaled elasticity coefficient
#
#\param[in] name This is the reaction variable for the unscaled elasticity
#\param[in] species This is the independent parameter, for example a floating of boundary species
#\param[out] value This is the value of the unscaled elasticity coefficient returned to the caller
#Takes (reactionName, parameterName) as an argument, where both arguments are strings
#\return
def getuEE(name, species):
    value = c_double()
    if rrLib.getuEE(gHandle, name, species, pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##\brief Compute the scaled elasticity for a given reaction and given species
#
#Takes (reactionName, parameterName) as an argument, where both arguments are strings
#\return
def getScaledFloatingSpeciesElasticity(reactionName, speciesName):
    value = c_double()
    if rrLib.getScaledFloatingSpeciesElasticity(gHandle, reactionName, speciesName,  pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##@}

##\ingroup NOM functions
#@{

##\brief Returns the number of rules in the current model
#\return Returns an integer larger or equal to 0 if succesful, or -1 on failure
def getNumberOfRules():
    return rrLib.getNumberOfRules(gHandle)

##@}


##\ingroup toString
#@{

##\brief Returns a result struct in string form.
#\return Returns a result struct as a string
def resultToString(result):
    return rrLib.resultToString(result)

##\brief Returns a matrix in string form.
#\return Returns a matrix as a string
def matrixToString(matrix):
    return rrLib.matrixToString(matrix)

##\brief Returns a vector in string form.
#\return Returns a vector as a string
def vectorToString(vector):
    return rrLib.vectorToString(vector)

##\brief Returns a string list in string form.
#\return Returns a string list as a string
def stringArrayToString(aList):
    return rrLib.stringArrayToString(aList)

##\brief Returns a list in string form
#\return Returns a string array as a string
def listToString(aList):
    return rrLib.listToString(aList)

##@}

# ----------------------------------------------------------------------------
#Free memory functions

##\ingroup freeRoutines
#@{


##\brief Free the result structure rrLib returned by simulate() and simulateEx()
def freeResult(rrLib):
    return rrLib.freeresult(rrLib)

##\brief Free char* generated by the C library routines
def freeText(text):
    return rrLib.freeText(text)

##\brief Free RRStringArrayHandle structures
def freeStringArray(sl):
    return rrLib.freeStringArray(sl)

##\brief Free RRVectorHandle structures
def freeVector(vector):
    return rrLib.freeVector(vector)

##\brief Free RRMatrixHandle structures
def freeMatrix(matrix):
    return rrLib.freeMatrix(matrix)

##\brief Free RRCCodeHandle structures
def freeCCode(code):
    return rrLib.freeCCode(code)

##@}
##\brief Pause
#\return void
def pause():
    return rrLib.pause()

#------------------------------------------------------------------------------
##\ingroup helperRoutines
#@{

##\brief Get the number of elements in a vector type
#
#Example: count = rrPython.getVectorLength(myVector)
#
#\param vector A vector rrLib
#\return Returns -1 if it fails, otherwise returns the number of elements in the vector
def getVectorLength(vector):
    return rrLib.getVectorLength(vector)
##\brief
#
#Example: myVector = rrPython.createVector(10)
#
#\param size The number of elements in the new vector
#\return Returns NONE if it fails, otherwise returns a vector rrLib
def createVector(size):
#    value = c_int(size)
    return rrLib.createVector(size)


##\brief Get a particular element from a vector
#
#Example: status = rrPython.getVectorElement (myVector, 10, &value);
#
#\param vector A pointer to the vector variable type
#\param index An integer indicating the ith element to retrieve (indexing is from zero)
#\param value A pointer to the retrieved double value
#\return Returns the vector element if successful
def getVectorElement(vector, index):
    value = c_double()
    if rrLib.getVectorElement(vector, c_int(index), pointer(value)) == True:
        return value.value
    else:
        raise RuntimeError(getLastError())

##\brief Get a particular element from a vector
#
#Example: status = rrPython.setVectorElement (myVector, 10, 3.1415);
#
#\param vector A vector rrLib
#\param index An integer indicating the ith element to set (indexing is from zero)
#\param value The value to store in the vector at the given index position
#\return Returns true if succesful
def setVectorElement(vector, index, value):
    value = c_double(value)
    if rrLib.setVectorElement(vector, c_int(index),  value) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

#def getStringListElement(stringList, index):
#    value = c_int()
#    if rrLib.getStringListElement(stringList, index, pointer(value)) == True:
#        return value.value
#    else:
#        raise RuntimeError("Index out of range")


##\brief Retrieve the number of rows in the given matrix
#
#Example: nRows = rrPython.getMatrixNumRows(matrix)
#
#\param matrix A matrix rrLib
#\return Returns -1 if fails, otherwise returns the number of rows
def getMatrixNumRows(matrix):
    return rrLib.getMatrixNumRows(matrix)

##\brief Retrieve the number of columns in the given matrix
#
#Example: nRows = rrPython.getMatrixNumCols(matrix)
#
#\param matrix A matrix rrLib
#\return Returns -1 if fails, otherwise returns the number of columns
def getMatrixNumCols(matrix):
    return rrLib.getMatrixNumCols(matrix)

##\brief Retrieves an element at a given row and column from a matrix type variable
#
#Example: status = rrPython.getMatrixElement (matrix, 2, 4)
#
#\param matrix A matrix rrLib
#\param row The row index to the matrix
#\param column The column index to the matrix
#\return Returns the value of the element if successful
def getMatrixElement(matrix, row, column):
    value = c_double()
    rvalue = c_int(row)
    cvalue = c_int(column)
    if rrLib.getMatrixElement(matrix, rvalue, cvalue, pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##\brief Retrieve the number of rows in the given result data
#
#Example: nRows = rrPython.getResultNumRows(result)
#
#\param result A result rrLib
#\return Returns -1 if fails, otherwise returns the number of rows
def getResultNumRows(result):
    return rrLib.getResultNumRows(result)

##\brief Retrieve the number of columns in the given result data
#
#Example: nRows = rrPython.getResultNumCols(result);
#
#\param result A result rrLib
#\return Returns -1 if fails, otherwise returns the number of columns
def getResultNumCols(result):
    return rrLib.getResultNumCols(result)

##\brief Retrieves an element at a given row and column from a result type variable
#
#Example: status = rrPython.getResultElement(result, 2, 4);
#
#\param result A result rrLib
#\param row The row index to the result data
#\param column The column index to the result data
#\return Returns true if succesful
def getResultElement(result, row, column):
    value = c_double()
    rvalue = c_int(row)
    cvalue = c_int(column)
    if rrLib.getMatrixElement(result, rvalue, cvalue, pointer(value)) == True:
        return value.value;
    else:
        raise RuntimeError('Index out of range')

##\brief Retrieves an element at a given row and column from a result type variable
#
#Example: str = getResultColumnLabel (result, 2, 4);
#
#\param result A result rrLib
#\param column The column index for the result data (indexing from zero)
#\return Returns NONE if fails, otherwise returns the string column label
def getResultColumnLabel(result, column):
    cvalue = c_int(column)
    return rrLib.getResultColumnLabel(result, cvalue)


##\brief Retrieve the header file code for the current model (if applicable)
#
#
#Example:   CCode = rrPython.getCCode()
#           header = rrPython.getCCodeHeader(CCode)
#
#\param code A rrLib for a string that stores the C code
#\return Returns the header for the C code rrLib used as an argument
def getCCodeHeader(codeHandle):
    return rrLib.getCCodeHeader(codeHandle)

##\brief Retrieve the source file code for the current model (if applicable)
#
#
#Example:   CCode = rrPython.getCCode()
#           header = rrPython.getCCodeSource(CCode)
#
#\param code A rrLib for a string that stores the C code
#\return Returns the source for the C code rrLib used as an argument
def getCCodeSource(codeHandle):
    return rrLib.getCCodeSource(codeHandle)

##\brief Returns the number of elements in a string array
#
#
#Example:  num = rrPython.getNumberOfStringElements(myStringArray)
#
#\param code A rrLib to the string array
#\return Returns the number of elements in the string array, -1 if there was an error
def getNumberOfStringElements(myArray):
    return rrLib.getNumberOfStringElements(myArray)

##\brief Utility function to return the indexth element from a string array
#
#
#Example:  num = rrPython.getStringElement (stringArray, 3)
#
#\param stringArray A rrLib to the string array
#\param index The indexth element to access (indexing from zero)
#\return Returns the string or raises exception if fails
def getStringElement (stringArray, index):
    element = rrLib.getStringElement (stringArray, index)
    if element == None:
       raise RuntimeError(getLastError())
    return element

##@}

# ----------------------------------------------------------------------------
# Utility function for converting a roadRunner stringarray into a Python List
def stringArrayToList (stringArray):
    result = []
    n = rrLib.getNumberOfStringElements (stringArray)
    for i in range (n):
        element = rrLib.getStringElement (stringArray, i)
        if element == False:
           raise RuntimeError(getLastError())
        val = c_char_p(element).value
        result.append (val)
        rrLib.freeText(element)
    return result

def rrVectorToPythonArray (vector):
    n = rrLib.getVectorLength(vector)
    if n == -1:
        raise RuntimeError ('vector is NULL in rrVectorToPythonArray')
    pythonArray = zeros(n)
    for i in range(n):
        pythonArray[i] = getVectorElement(vector, i)
    return pythonArray

def PythonArrayTorrVector (myArray):
    v = rrLib.createVector (len(myArray))
    for i in range (len(myArray)):
        value = myArray[i]
        rrLib.setVectorElement (v, i, value)
    return v


def rrListToPythonList (values):
    n = rrLib.getListLength (values)
    result = []
    for i in range (n):
        item = rrLib.getListItem (values, i)
        result.append (rrLib.getStringListItem (item))
    return result


def createMatrix (rrMatrix):
    rowCount = rrLib.getMatrixNumRows(rrMatrix)
    colCount = rrLib.getMatrixNumCols(rrMatrix)
    matrixArray = zeros((rowCount, colCount))
    for m in range(rowCount):
        for n in range(colCount):
            value = c_double()
            rvalue = m
            cvalue = n
            if rrLib.getMatrixElement(rrMatrix, rvalue, cvalue, pointer(value)) == True:
               matrixArray[m, n] = value.value
    return matrixArray

def createRRMatrix (marray):
    r = marray.shape[0]
    c = marray.shape[1]
    rrm = rrLib.createRRMatrix (r, c)
    for i in range (c):
        for j in range (r):
            rrLib.setMatrixElement (rrm, i, j, marray[i, j])
    return rrm

# ---------------------------------------------------------------------------------

#Plugin functionality
#rrLib.loadPlugins.restyp = c_bool
#rrLib.unLoadPlugins.restyp = c_bool
#rrLib.getNumberOfPlugins.restyp = c_int
#rrLib.getPluginInfo.restyp = c_char_p
def loadPlugins():
    return rrLib.loadPlugins(gHandle)

def unLoadPlugins():
    return rrLib.unLoadPlugins(gHandle)

def getNumberOfPlugins():
    return rrLib.getNumberOfPlugins(gHandle)

def getPluginInfo(pluginName):
    return rrLib.getPluginInfo(gHandle, pluginName)

def executePlugin(pluginName):
    return rrLib.executePlugin(gHandle, pluginName)


def compileSource(sourceFileName, rrHandle = None):
    if rrHandle is None:
        rrHandle = gHandle
    return rrLib.compileSource(rrHandle, sourceFileName)

