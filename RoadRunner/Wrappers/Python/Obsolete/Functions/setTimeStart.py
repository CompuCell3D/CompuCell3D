import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'setTimeStart'
rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')

try:
    start = rrPython.setTimeStart(0.0)
    if str(start) is not False:
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