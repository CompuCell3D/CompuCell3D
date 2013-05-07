import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'getRatesOfChangeEx'
rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')
rrPython.simulate()

SpeciesArray = [1.0,.9,.8,.6]

try:
    rates = rrPython.getRatesOfChangeEx(SpeciesArray)
    if str(rates) is not False:
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