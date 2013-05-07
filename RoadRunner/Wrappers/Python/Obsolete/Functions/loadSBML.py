import rrPython
import os
import csv
os.chdir('C:\\RoadRunner\\bin')

function = 'loadSBML'
file = open('C:\\RoadRunner\\Models\\feedback.xml','r').read()
#rrPython.loadSBML(file)

try:
    sbml = rrPython.loadSBML(file)
    if str(sbml) is not False:
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