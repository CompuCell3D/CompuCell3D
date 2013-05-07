import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'getRateOfChangeIds'
rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')

try:
    names = rrPython.getRateOfChangeIds(10)
    if str(names) is not False:
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