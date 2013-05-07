import rrPython
import os
rrPython.enableLogging()
rrPython.setTempFolder('C:\\RRTemp')
os.chdir('C:\\RoadRunner\\bin')

function = 'loadSBMLFromFile'

try:
    sbml = rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')
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