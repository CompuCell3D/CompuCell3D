import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'getResultElement'
rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')
rrPython.simulate()

try:
    element = rrPython.getResultElement()
    if str(element) is not False:
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