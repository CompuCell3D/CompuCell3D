import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'setCompartmentByIndex'
rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')

index = 0
value = 1.0
try:
    specs = rrPython.setCompartmentByIndex(index, value)
    if str(specs) is not False:
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