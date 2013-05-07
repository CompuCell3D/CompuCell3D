import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'getBoundarySpeciesByIndex'
rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')

index = 0

try:
    specs = rrPython.getBoundarySpeciesByIndex(index)
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