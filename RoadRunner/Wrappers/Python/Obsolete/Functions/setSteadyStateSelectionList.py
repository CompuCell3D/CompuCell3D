import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'setSteadyStateSelectionList'
rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')

try:
    list = rrPython.setSteadyStateSelectionList('S1')
    if str(list) is not False:
        result = 'True'
    else:
        result = 'False'
except:
    result = 'False'


PythonTestResults = open('C:\\RoadRunner\\PythonTestResults.csv','a')
writer = csv.writer(PythonTestResults)
writevar = function + '=' + result
writer.writerow([writevar])
PythonTestResults.close()