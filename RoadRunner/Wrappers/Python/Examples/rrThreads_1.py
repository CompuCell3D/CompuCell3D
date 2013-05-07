import sys
from rrPython import *

if hasError():
    print 'There was an error: '
    print getLastError();
    print 'Exiting...'
    exit(0)

modelFile=''
if sys.platform.startswith('win32'):
    modelFile ="r:/models/feedback.xml"
    setTempFolder('r:/temp')
else:
    modelFile = "../models/test_1.xml"
    setTempFolder('../temp')

print getLastError()

print 'TempFolder is :' + getTempFolder()

jobHandle = loadSBMLFromFileJob(modelFile)
waitForJob(jobHandle)

print 'Loading SBML errors:' + getLastError()

jobHandle = simulateJob()

#waitForJob(jobHandle)
while(isJobFinished(jobHandle) == False):
    print 'The job is not done'

print  getSimulationResult()

print getLastError()
print "RRPython is done"
