import rrPython
import os
rrPython.enableLogging()
rrPython.setTempFolder('C:\\RRTemp')
os.chdir('C:\\RoadRunner\\bin')

result = rrPython.loadSBMLFromFile('C:\\RoadRunner\\Models\\feedback.xml')
print result

simulation = rrPython.simulate()
print simulation

#insert known results, compare

PythonTestResults = open('C:\\RoadRunner\\PythonTestResults.csv','a')
writer = csv.writer(PythonTestResults)
writevar = function + '=' + result
writer.writerow([writevar])
PythonTestResults.close()