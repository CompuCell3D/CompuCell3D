import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'setValue'
rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')

species = 'S1'
val = 1.0

try:
    concentration = rrPython.setValue(species,val)
    if str(concentration) == str(val):
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