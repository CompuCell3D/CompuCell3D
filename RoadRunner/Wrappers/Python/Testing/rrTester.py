# Test Module for RoadRunner
#
# Usage:
# import rrTester
# runTest ('mymodel')


#-------------------------------------------------------------
# Tests for steady state and stoichiometric calculations in
# roadRunner. Herbert M Sauro November 2012
#-------------------------------------------------------------

#------------------------------------------
# Change this line for different test files
#nameOfResultsFile = 'results_roadRunnerTest_1.txt'


import sys
import random
import string
import rrPython
from numpy import *

# Module wide file handle
fHandle = ''
rpadding = 45
sbmlStr = ''
JarnacStr = ''

def expectApproximately (a, b, tol):
     diff = a - b
     return abs(diff) < tol

def passMsg (errorFlag):
    if errorFlag:
        return "*****FAIL*****"
    else:
        return "PASS"

# Empty lines are ignored
# Lines starting with # are also ignored

def readLine ():
    line = fHandle.readline()
    while line == '\n':
       line = fHandle.readline()
    while line == '':
        line = fHandle.readline()

    while (line[0] == '#') or (line == '') or (line[0] == '\n'):
          if line == '':
             return line
          line = fHandle.readline();
    return line.strip('\n')

def jumpToNextTest():
     line = readLine()
     #line = ''
     #while line == '':
     #      line = fHandle.readline().strip ('\n')
     while line[0] != '[':
        line = readLine()
     return line

def getSBMLStr ():
    sbmlStr = ''
    line = fHandle.readline()
    while (line != '[END_MODEL]' + '\n'):
          sbmlStr = sbmlStr + line
          line = fHandle.readline()
    return sbmlStr

def getJarnacStr ():
    JarnacStr = ''
    line = fHandle.readline()
    while (line != '[END_MODEL]' + '\n'):
          JarnacStr = JarnacStr + line
          line = fHandle.readline()
    return JarnacStr

def loadSBMLModelFromTestFile ():
    testId = jumpToNextTest()
    if testId == '[SBML]':
       return getSBMLStr ()


def loadJarnacModelFromTestFile ():
    testId = jumpToNextTest ()
    if testId == '[JARNAC]':
       return getJarnacStr ()

# ------------------------------------------------------------------------
def setConservationLaw(testId):
    line = readLine ()
    if line == 'True':
        rrPython.setComputeAndAssignConservationLaws(1)
    else:
        rrPython.setComputeAndAssignConservationLaws(0)

def mySetSteadyStateSelectionList(testId):
    line = readLine ()
    rrPython.setSteadyStateSelectionList (line)

def myComputeSteadyState(testId):
    line = readLine ()
    if line == "True":
       print "Compute Steady State, distance to SteadyState:", rrPython.steadyState()


def checkSpeciesConcentrations(testId):
   words = []
   species = []
   m = rrPython.getNumberOfFloatingSpecies()
   for i in range (0,m):
       line = readLine ()
       words = line.split()
       words.append (rrPython.getValue(words[0]))
       species.append (words)

   # Steady State Concentrations
   print string.ljust ("Check " + testId, rpadding),
   errorFlag = False
   for i in range (0,m):
       expectedValue =  float (species[i][1])
       if expectApproximately (expectedValue, species[i][2], 1E-6) == False:
          errorFlag = True
          break
   print passMsg (errorFlag)

def checkFluxes(testId):
   words = []
   fluxes = []
   # Steady State Fluxes
   print string.ljust ("Check " + testId, rpadding),
   errorFlag = False
   m = rrPython.getNumberOfFloatingSpecies()
   n = rrPython.getNumberOfReactions();
   for i in range (0,n):
       line = readLine ()
       words = line.split()
       words.append (rrPython.getValue(words[0]))
       fluxes.append (words)

   for i in range (0,n):
       expectedValue = float (fluxes[i][1])
       if expectApproximately (expectedValue, fluxes[i][2], 1E-6) == False:
          errorFlag = True
          break
   print passMsg (errorFlag)

def checkFullJacobian(testId):
   # Jacobian
   print string.ljust ("Check " + testId, rpadding),
   errorFlag = False
   m = rrPython.getNumberOfFloatingSpecies()
   Jacobian = rrPython.getFullJacobian()
   for i in range(0,m):
       line = readLine ()
       words = line.split()
       for j in range(0,m):
           expectedValue = float(words[j])
           if expectApproximately (expectedValue, Jacobian[i,j], 1E-6) == False:
              errorFlag = True
              break
   print passMsg (errorFlag)


def checkIndividualEigenvalues(testId):
   # Eigenvalues
   print string.ljust ("Check " + testId, rpadding),
   errorFlag = False
   m = rrPython.getNumberOfFloatingSpecies()
   try:
     for i in range(0,m):
        line = readLine ()
        words = line.split()
        eigenvalueName = words[0]
        realPart = rrPython.getValue ('eigen_' + eigenvalueName)
        realPart = float (realPart)
        if expectApproximately (realPart, float(words[1]), 1E-6) == False:
            errorFlag = True
            break
     print passMsg (errorFlag)
   except:
      print 'Unexpected error in checkIndividualEigenvalues:', sys.exc_info()[0]


def checkEigenvalueMatrix(testId):
   # Eigenvalues
   print string.ljust ("Check " + testId, rpadding),
   errorFlag = False
   m = rrPython.getNumberOfFloatingSpecies()
   eigenvalues = rrPython.getEigenvalues()
   for i in range(0,m):
      line = readLine ()
      words = line.split()
      realPart = float (words[0])
      # Check if there is an imaginary part
      if len (words) == 1:
         imagPart = 0
      else:
         imagPart= float (words[1])
      if (expectApproximately (realPart, eigenvalues[i,0], 1E-6) == False) or (expectApproximately (imagPart, eigenvalues[i,1], 1E-6)) == False:
          errorFlag = True
          break
   print passMsg (errorFlag)


def checkStoichiometryMatrix(testId):
   # Stoichiometry matrix
   print string.ljust ("Check " + testId, rpadding),
   errorFlag = False
   m = rrPython.getNumberOfFloatingSpecies()
   n = rrPython.getNumberOfReactions();
   st = rrPython.getStoichiometryMatrix()
   for i in range(0,m):
       line = readLine ()
       words = line.split()
       for j in range(0,n):
           if expectApproximately(float (words[j]), st[i,j], 1E-6) == False:
              errorFlag = True
              break
   print passMsg (errorFlag)

def checkLinkMatrix(testId):
   # Link matrix
   print string.ljust ("Check " + testId, rpadding),
   errorFlag = False
   m = rrPython.getNumberOfFloatingSpecies()
   st = rrPython.getLinkMatrix()
   for i in range(0,m):
       words = readLine ().split()
       for j in range(0,m):
           if expectApproximately(float (words[j]), st[i,j], 1E-6) == False:
              errorFlag = True
              break
   print passMsg (errorFlag)

def checkUnscaledConcentrationControlMatrix(testId):
   # Unscaled Concentration Control matrix
   print string.ljust ("Check " + testId, rpadding),
   words = []
   errorFlag = False
   m = rrPython.getNumberOfFloatingSpecies()
   n = rrPython.getNumberOfReactions();
   st = rrPython.getUnscaledConcentrationControlCoefficientMatrix();
   for i in range(0,m):
        words = readLine ().split()
        for j in range(0,n):
            if expectApproximately(float (words[j]), st[i,j], 1E-6) == False:
               errorFlag = True
               break
   print passMsg (errorFlag)


def checkScaledConcentrationControlMatrix(testId):
   # Unscaled Concentration Control matrix
   print string.ljust ("Check " + testId, rpadding),
   words = []
   errorFlag = False
   m = rrPython.getNumberOfFloatingSpecies()
   n = rrPython.getNumberOfReactions();
   st = rrPython.getScaledConcentrationControlCoefficientMatrix();
   for i in range(0,m):
        words = readLine ().split()
        for j in range(0,n):
            if expectApproximately(float (words[j]), st[i,j], 1E-6) == False:
               errorFlag = True
               break
   print passMsg (errorFlag)


def checkUnscaledFluxControlCoefficientMatrix(testId):
   # Unscaled Flux Control matrix
   print string.ljust ("Check " + testId, rpadding),
   words = []
   errorFlag = False
   n = rrPython.getNumberOfReactions();
   st = rrPython.getUnscaledFluxControlCoefficientMatrix();
   for i in range(0,n):
        words = readLine ().split()
        for j in range(0,n):
            if expectApproximately(float (words[j]), st[i,j], 1E-6) == False:
               errorFlag = True
               break
   print passMsg (errorFlag)


def checkScaledFluxControlCoefficientMatrix(testId):
   # Unscaled Flux Control matrix
   print string.ljust ("Check " + testId, rpadding),
   words = []
   errorFlag = False
   n = rrPython.getNumberOfReactions();
   st = rrPython.getScaledFluxControlCoefficientMatrix()
   for i in range(0,n):
        words = readLine ().split()
        for j in range(0,n):
            if expectApproximately(float (words[j]), st[i,j], 1E-6) == False:
               errorFlag = True
               break
   print passMsg (errorFlag)


def checkUnscaledElasticityMatrix(testId):
  # Jacobian
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  m = rrPython.getNumberOfFloatingSpecies()
  uee = rrPython.getUnscaledElasticityMatrix()
  for i in range(0,m):
      line = readLine ()
      words = line.split()
      for j in range(0,m):
          expectedValue = float(words[j])
          if expectApproximately (expectedValue, uee[i,j], 1E-6) == False:
             errorFlag = True
             break
  print passMsg (errorFlag)

def checkScaledElasticityMatrix(testId):
  # Jacobian
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  m = rrPython.getNumberOfFloatingSpecies()
  ee = rrPython.getScaledElasticityMatrix()
  for i in range(0,m):
      line = readLine ()
      words = line.split()
      for j in range(0,m):
          expectedValue = float(words[j])
          if expectApproximately (expectedValue, ee[i,j], 1E-6) == False:
             errorFlag = True
             break
  print passMsg (errorFlag)


def checkGetFloatingSpeciesIds(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  expected = rrPython.getFloatingSpeciesIds()
  m = rrPython.getNumberOfFloatingSpecies()
  for i in range(0,m):
      if words[i] != expected[i]:
             errorFlag = True
             break
  print passMsg (errorFlag)

def checkGetBoundarySpeciesIds(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  expected = rrPython.getBoundarySpeciesIds()
  m = rrPython.getNumberOfBoundarySpecies()
  for i in range(0,m):
      if words[i] != expected[i]:
             errorFlag = True
             break
  print passMsg (errorFlag)


def checkGetGlobalParameterIds (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  expected = rrPython.getGlobalParameterIds()
  m = rrPython.getNumberOfGlobalParameters()
  for i in range(0,m):
      if words[i] != expected[i]:
             errorFlag = True
             break
  print passMsg (errorFlag)


def checkGetCompartmentIds (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  expected = rrPython.getCompartmentIds()
  m = rrPython.getNumberOfCompartments()
  for i in range(0,m):
      if words[i] != expected[i]:
             errorFlag = True
             break
  print passMsg (errorFlag)


def checkReactionIds (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  expected = rrPython.getReactionIds()
  m = rrPython.getNumberOfReactions();
  for i in range(0,m):
      if words[i] != expected[i]:
             errorFlag = True
             break
  print passMsg (errorFlag)


def checkFloatingSpeciesInitialConditionIds (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  expected = rrPython.getFloatingSpeciesInitialConditionIds()
  m = rrPython.getNumberOfFloatingSpecies()
  for i in range(0,m):
      if words[i] != expected[i]:
             errorFlag = True
             break
  print passMsg (errorFlag)


def checkEigenValueIds (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  expected = rrPython.getEigenvalueIds()
  m = rrPython.getNumberOfFloatingSpecies()
  for i in range(0,m):
      if words[i] != expected[i]:
             errorFlag = True
             break
  print passMsg (errorFlag)


def checkGetRatesOfChangeIds (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  expected = rrPython.getRatesOfChangeIds()
  n = rrPython.getNumberOfFloatingSpecies()
  for i in range(0,n):
      if words[i] != expected[i]:
             errorFlag = True
             break
  print passMsg (errorFlag)


def checkSetSteadyStateSelectionList(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  result = rrPython.setSteadyStateSelectionList (line)
  if result == False:
     errorFlag = True
  print passMsg (errorFlag)


def checkGetSteadyStateSelectionList(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  result = rrPython.getSteadyStateSelectionList ()
  for i in range (len (words)):
      if words[i] != result[i]:
         errorFlag = True
         break
  print passMsg (errorFlag)


def checkSetTimeCourseSelectionList(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  result = rrPython.setTimeCourseSelectionList (line)
  if result == False:
     errorFlag = True
  print passMsg (errorFlag)


def checkGetTimeCourseSelectionList(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  line = readLine ()
  words = line.split()
  result = rrPython.getTimeCourseSelectionList()
  for i in range (len (words)):
      if words[i] != result[i]:
         errorFlag = True
         break
  print passMsg (errorFlag)


def checkComputeSteadyStateValues(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  list = rrPython.getSteadyStateSelectionList()
  ss = rrPython.computeSteadyStateValues()
  words = readLine().split()
  for i in range (len (list)):
      if expectApproximately(float (words[i]), ss[i], 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def checkFloatingSpeciesConcentrations(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  ss = rrPython.getFloatingSpeciesConcentrations()
  words = readLine().split()
  for i in range (len (ss)):
      if expectApproximately(float (words[i]), ss[i], 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def checkBoundarySpeciesConcentrations(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  ss = rrPython.getBoundarySpeciesConcentrations()
  words = readLine().split()
  for i in range (len (ss)):
       if expectApproximately(float (words[i]), ss[i], 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def checkGlobalParameterValues(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  ss = rrPython.getGlobalParameterValues()
  words = readLine().split()
  for i in range (len (ss)):
      if expectApproximately(float (words[i]), ss[i], 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def checkInitalFloatingSpeciesConcentations(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  ss = rrPython.getFloatingSpeciesInitialConcentrations()
  words = readLine().split()
  for i in range (len (ss)):
      if expectApproximately(float (words[i]), ss[i], 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def checkReactionRates(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  ss = rrPython.getReactionRates()
  words = readLine().split()
  for i in range (len (ss)):
      if expectApproximately(float (words[i]), ss[i], 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def checkGetReactionRatesByIndex(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  words = readLine().split()
  n = rrPython.getNumberOfReactions()
  for i in range (n):
      value = rrPython.getReactionRate (i)
      if expectApproximately(float (words[i]), value, 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def checkNumberOfDependentSpecies(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  value = int (readLine())
  n = rrPython.getNumberOfDependentSpecies()
  if n != value:
    errorFlag = True
  print passMsg (errorFlag)


def checkNumberOfIndependentSpecies(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  value = int (readLine())
  n = rrPython.getNumberOfIndependentSpecies()
  if n != value:
    errorFlag = True
  print passMsg (errorFlag)


def checkInitialConditions(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  words = readLine().split()
  values = rrPython.getFloatingSpeciesInitialConcentrations()
  for i in range(len(words)):
      if expectApproximately (float (words[i]), values[i], 1E-6) == False:
        errorFlag = True
  print passMsg (errorFlag)


def checkNumberOfRules(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  value = int (readLine())
  if rrPython.getNumberOfRules() != value:
    errorFlag = True
  print passMsg (errorFlag)


def checkGetRatesOfChange(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  words = readLine().split()
  values = rrPython.getRatesOfChange()
  for i in range (len(words)):
      if expectApproximately (float (words[i]), values[i], 1E-6) == False:
        errorFlag = True
  print passMsg (errorFlag)


def checkGetReactionRatesEx (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  inputConcs = asarray (readLine().split(), dtype=float64)
  values = rrPython.getReactionRatesEx (inputConcs)
  outputRates = asarray (readLine().split(), dtype=float64)
  if not allclose (values, outputRates):
    errorFlag = True
  print passMsg (errorFlag)


def checkGetRatesOfChangeEx (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  inputConcs = asarray (readLine().split(), dtype=float64)
  values = rrPython.getRatesOfChangeEx (inputConcs)
  outputRates = asarray (readLine().split(), dtype=float64)
  if not allclose (values, outputRates):
    errorFlag = True
  print passMsg (errorFlag)


def checkRateRateOfChangeByIndex(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  inputConcs = asarray (readLine().split(), dtype=float64)
  outputRates = asarray (readLine().split(), dtype=float64)
  rrPython.setFloatingSpeciesConcentrations (inputConcs)
  for i in range (len (inputConcs)):
      value = rrPython.getRateOfChange (i)
      if expectApproximately (value, outputRates[i], 1E-6) == False:
        errorFlag = True
        break
  print passMsg (errorFlag)

# ---------------------------------------------------------------------------

def setGetValues(IdList, testId):
    print string.ljust ("Check " + testId, rpadding),
    errorFlag = False
    for i in range (len(IdList)):
        value = random.random()*10
        rrPython.setValue (IdList[i], value)
        if expectApproximately (rrPython.getValue (IdList[i]), value, 1E-6) == False:
            errorFlag = True
            break
    print passMsg (errorFlag)


def setGetTimeStart(testId):
    print string.ljust ("Check " + testId, rpadding),
    errorFlag = False
    value = random.random ()*10
    rrPython.setTimeStart (value)
    if expectApproximately (rrPython.getTimeStart (), value, 1E-6) == False:
            errorFlag = True
    print passMsg (errorFlag)


def setGetTimeEnd(testId):
    print string.ljust ("Check " + testId, rpadding),
    errorFlag = False
    value = random.random ()*10
    rrPython.setTimeEnd (value)
    if expectApproximately (rrPython.getTimeEnd (), value, 1E-6) == False:
            errorFlag = True
    print passMsg (errorFlag)


def setGetNumberOfPoints(testId):
    print string.ljust ("Check " + testId, rpadding),
    errorFlag = False
    value = random.randint (1, 100)
    rrPython.setNumPoints (value)
    if rrPython.getNumPoints () != value:
            errorFlag = True
    print passMsg (errorFlag)


def setGetTimeCourseSelectionList(testId):
    print string.ljust ("Check " + testId, rpadding),
    errorFlag = False
    myList = rrPython.getFloatingSpeciesIds()
    newList = list (myList)
    random.shuffle (newList)
    rrPython.setTimeCourseSelectionList (newList)
    if rrPython.getTimeCourseSelectionList() != newList:
        errorFlag = True
    print passMsg (errorFlag)


def setGetSteadyStateSelectionList(testId):
    print string.ljust ("Check " + testId, rpadding),
    errorFlag = False
    myList = rrPython.getFloatingSpeciesIds()
    newList = list (myList)
    while newList == myList:
         random.shuffle (newList)
    rrPython.setSteadyStateSelectionList (newList)
    getList = rrPython.getSteadyStateSelectionList()
    if getList != newList:
        errorFlag = True
    print passMsg (errorFlag)


def setGetFloatingSpeciesByIndex(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  n = rrPython.getNumberOfFloatingSpecies()
  for i in range (n):
      value = random.random()*10
      rrPython.setFloatingSpeciesByIndex (i, value)
      if expectApproximately(rrPython.getFloatingSpeciesByIndex (i), value, 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def setGetBoundarySpeciesByIndex(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  n = rrPython.getNumberOfBoundarySpecies()
  for i in range (n):
      value = random.random()*10
      rrPython.setBoundarySpeciesByIndex (i, value)
      if expectApproximately(rrPython.getBoundarySpeciesByIndex (i), value, 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def setGetCompartmentByIndex(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  n = rrPython.getNumberOfCompartments()
  for i in range (n):
      value = random.random()*10
      rrPython.setCompartmentByIndex (i, value)
      if expectApproximately(rrPython.getCompartmentByIndex (i), value, 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def setGetGlobalParameterByIndex (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  n = rrPython.getNumberOfGlobalParameters()
  for i in range (n):
      value = random.random()*10
      rrPython.setGlobalParameterByIndex (i, value)
      if expectApproximately(rrPython.getGlobalParameterByIndex (i), value, 1E-6) == False:
          errorFlag = True
          break;
  print passMsg (errorFlag)


def setGetFloatingSpeciesConcentrations (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  getArray = rrPython.getFloatingSpeciesConcentrations()
  setArray = zeros(len(getArray))
  for i in range(len(getArray)):
      value = random.random()*10
      setArray[i] = value
  rrPython.setFloatingSpeciesConcentrations (setArray)
  if (setArray != rrPython.getFloatingSpeciesConcentrations()).all():
      errorFlag = True
  print passMsg (errorFlag)


def setGetBoundarySpeciesConcentrations (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  getArray = rrPython.getBoundarySpeciesConcentrations()
  setArray = zeros(len(getArray))
  for i in range(len(getArray)):
      value = random.random()*10
      setArray[i] = value
  rrPython.setBoundarySpeciesConcentrations (rrPython.PythonArrayTorrVector (setArray))
  if (setArray != rrPython.getBoundarySpeciesConcentrations()).all():
      errorFlag = True
  print passMsg (errorFlag)


def setGetInitialFloatingSpeciesConcentrations (testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  getArray = rrPython.getFloatingSpeciesInitialConcentrations ()
  setArray = zeros(len(getArray))
  for i in range(len(getArray)):
      value = random.random()*10
      setArray[i] = value
  rrPython.setFloatingSpeciesInitialConcentrations (setArray)
  if (setArray != rrPython.getFloatingSpeciesInitialConcentrations()).all():
      errorFlag = True
  print passMsg (errorFlag)


def setGetReset(testId):
  print string.ljust ("Check " + testId, rpadding),
  errorFlag = False
  values = zeros (rrPython.getNumberOfFloatingSpecies())
  for i in range (len (values)):
    values[i] = random.random()*10
  initial = rrPython.getFloatingSpeciesInitialConcentrations()
  rrPython.setFloatingSpeciesConcentrations (values)
  # Should reset the floats by to the current initial condition
  rrPython.reset()
  values = rrPython.getFloatingSpeciesConcentrations()
  if(values != initial).all():
    errorFlag = True
  print passMsg (errorFlag)


def scriptTests():
    print
    print "Testing Set and Get Functions"
    print "-----------------------------"
    setGetValues(rrPython.getFloatingSpeciesIds(), 'Set/Get Value (Floats)')
    setGetValues(rrPython.getBoundarySpeciesIds(), 'Set/Get Value (Boundary)')
    setGetValues(rrPython.getGlobalParameterIds(), 'Set/Get Value (Global Parameters)')
    setGetValues(rrPython.getCompartmentIds(), 'Set/Get Value (Compartments)')
    setGetTimeStart('Set/Get TimeStart')
    setGetTimeEnd ('Set/Get TimeEnd')
    setGetNumberOfPoints ('Set/Get Number Of Points')
    setGetTimeCourseSelectionList ('Set/Get Time Course Selection List')
    setGetSteadyStateSelectionList ('Set/Get Steady State Selection List')
    setGetFloatingSpeciesByIndex ('Set/Get Floating Species by Index')
    setGetBoundarySpeciesByIndex ('Set/Get Boundary Species by Index')
    setGetCompartmentByIndex ('Set/Get Compartment by Index')
    setGetGlobalParameterByIndex ('Set/Get Global Parameter bu Index')
    setGetBoundarySpeciesConcentrations ('Set/Get Boundary Species Concs')
    setGetFloatingSpeciesConcentrations ('Set/Get Floating Species Concs')
    setGetInitialFloatingSpeciesConcentrations ('Set/Get Initial Concs')
    setGetReset ('Set/Get Reset Method')


# ------------------------------------------------------------------------
# List of tests
functions = {'[Compute Steady State]': myComputeSteadyState,
             '[Set Steady State Selection List]': mySetSteadyStateSelectionList,
             '[Conservation Laws]': setConservationLaw,
             '[Species Concentrations]': checkSpeciesConcentrations,
             '[Fluxes]': checkFluxes,
             '[Full Jacobian]': checkFullJacobian,
             '[Eigenvalue Matrix]': checkEigenvalueMatrix,
             '[Individual Eigenvalues]': checkIndividualEigenvalues,
             '[Stoichiometry Matrix]': checkStoichiometryMatrix,
             '[Link Matrix]': checkLinkMatrix,
             '[Unscaled Elasticity Matrix]': checkUnscaledElasticityMatrix,
             '[Scaled Elasticity Matrix]': checkScaledElasticityMatrix,
             '[Unscaled Concentration Control Matrix]': checkUnscaledConcentrationControlMatrix,
             '[Unscaled Flux Control Matrix]': checkUnscaledFluxControlCoefficientMatrix,
             '[Scaled Concentration Control Matrix]': checkScaledConcentrationControlMatrix,
             '[Scaled Flux Control Matrix]': checkScaledFluxControlCoefficientMatrix,
             '[Floating Species Ids]': checkGetFloatingSpeciesIds,
             '[Boundary Species Ids]': checkGetBoundarySpeciesIds,
             '[Global Parameter Ids]': checkGetGlobalParameterIds,
             '[Compartment Ids]': checkGetCompartmentIds,
             '[Reaction Ids]': checkReactionIds,
             '[Species Initial Condition Ids]': checkFloatingSpeciesInitialConditionIds,
             '[Get Eigenvalue Ids]': checkEigenValueIds,
             '[Get Rates Of Change Ids]': checkGetRatesOfChangeIds,
             '[Set Steady State Selection List]': checkSetSteadyStateSelectionList,
             '[Get Steady State Selection List]': checkGetSteadyStateSelectionList,
             '[Set Time Course Selection List]': checkSetTimeCourseSelectionList,
             '[Get Time Course Selection List]': checkGetTimeCourseSelectionList,
             '[Compute Steady State Values]': checkComputeSteadyStateValues,
             '[Floating Species Concentrations]': checkFloatingSpeciesConcentrations,
             '[Boundary Species Concentrations]': checkBoundarySpeciesConcentrations,
             '[Get Global Parameter Values]': checkGlobalParameterValues,
             '[Get Initial Floating Species Concs]': checkInitalFloatingSpeciesConcentations,
             '[Get Reaction Rates]': checkReactionRates,
             '[Get Reaction Rate By Index]': checkGetReactionRatesByIndex,
             '[Number of Dependent Species]': checkNumberOfDependentSpecies,
             '[Number of Independent Species]': checkNumberOfIndependentSpecies,
             '[Get Initial Floating Species Concs]': checkInitialConditions,
             '[Number Of Rules]': checkNumberOfRules,
             '[Get Rates Of Change]': checkGetRatesOfChange,
             '[Get Reaction Rates Ex]': checkGetReactionRatesEx,
             '[Get Rates of Change Ex]': checkGetRatesOfChangeEx,
             '[Get Rate of Change by Index]': checkRateRateOfChangeByIndex,
              }

def runTester (pathToModels, testModel):
    global fHandle
    global sbmlStr
    global JarnacStr

    print "Starting Tester on ", testModel

    sys.path.append (pathToModels)
    #print 'Version: ', rrPython.getVersion()
    #print 'Build Date: ', rrPython.getBuildDate()
    #print 'Copyright Message: ', rrPython.getCopyright()

    fHandle = open (pathToModels + testModel, 'r')

    sbmlStr = loadSBMLModelFromTestFile ()
    JarnacStr = loadJarnacModelFromTestFile ()

    print "\n", "Info:"+ "\n"

    rrPython.enableLogging()
    info = rrPython.getInfo()
    for item in info:
        print item
    print

    # Load any initialization actions
    testId = jumpToNextTest()
    if testId == '[INITIALIZATION]':
        testId = jumpToNextTest ()
        while testId != '[END_INITIALIZATION]':
            if functions.has_key(testId):
               func = functions[testId]
               func(testId)
            else:
               print 'No initialization function found for ' + testId
            testId = jumpToNextTest()

    # Load the model into RoadRunner
    if rrPython.loadSBML(sbmlStr) == False:
        print 'Failed to load model'
        print rrPython.getLastError()
        sys.exit(-1)

    print rrPython.getuCC ('J1', 'k1')
    print rrPython.getuCC ('J1', 'k_1')
    print rrPython.getuCC ('J1', 'k2')
    print rrPython.getuCC ('J1', 'k_2')
    print rrPython.getuCC ('J1', 'k3')
    print rrPython.getuCC ('J1', 'k_3')
    print rrPython.getuCC ('J1', 'k4')
    print rrPython.getuCC ('J1', 'k_4')

    # Now start the tests proper
    testId = jumpToNextTest()
    if testId == '[START_TESTS]':
        testId = jumpToNextTest()
        while testId != '[END_TESTS]':
           if functions.has_key(testId):
              func = functions[testId]
              func(testId)
           else:
              print string.ljust (testId, rpadding), 'NO TEST'
           testId = jumpToNextTest()
    else:
      print 'No Tests found'

    scriptTests()

    #print
    #print 'Unscaled Flux Control Ids: ', rrPython.getUnscaledFluxControlCoefficientIds(), "\n"
    #print 'Scaled Flux Control Ids: ', rrPython.getFluxControlCoefficientIds(), "\n"
    #print 'Unscaled Concentration Control Ids: ', rrPython.getUnscaledConcentrationControlCoefficientIds(), "\n"
    #print 'Scaled Concentration Control Ids', rrPython.getConcentrationControlCoefficientIds(), "\n"

    #print 'Steady State Symbols: ', rrPython.getAvailableSteadyStateSymbols(), "\n"
    #print 'Time Course Symbols: ', rrPython.getAvailableTimeCourseSymbols(), "\n"

# --------------------------------------------------------------








