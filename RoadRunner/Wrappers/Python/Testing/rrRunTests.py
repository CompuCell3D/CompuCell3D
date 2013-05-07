
import sys
import rrTester
import rrPython
import numpy

pathToModels = 'C:\\vs\\trunk\\Wrappers\\Python\\Testing\\'
#pathToModels = 'R:\\installs\\vs_2012\\RelWDebInfo\\testing\\'
# runTester takes two arguments:
#  1) The path to the results and model file
#  2) The name of the results and model file

rrTester.runTester (pathToModels, 'Test_1.txt')
