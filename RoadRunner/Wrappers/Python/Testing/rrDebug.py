import sys
import rrPython
rr = rrPython


modelFile=''
if sys.platform.startswith('win32'):
    modelFile ="r:/models/feedback.xml"
    if not rr.setTempFolder('r:/rrTemp/python') == True:
        print "Failed to set temp folder"
else:
    modelFile ="/home/totte/rrInstall/models/feedback.xml"
    rr.setTempFolder('/home/totte/rrTemp/python')

print rr.getLastError()
print 'RoadRunner Version: ' + rr.getVersion()
print 'RoadRunner Build DateTime: ' + rr.getBuildDateTime()
print 'Copyright: ' + rr.getCopyright()

tempFolder=rr.getTempFolder()

print 'TempFolder is :' + tempFolder
info = rr.getInfo()

print info;


rr.setComputeAndAssignConservationLaws(True)
result = rr.loadSBMLFromFile(modelFile)

print 'Result of loading sbml: '
print result;
print rr.getEigenvalues()
print rr.getEigenvalueIds()
print rr.getTimeCourseSelectionList()
rr.setNumPoints(50)
rr.setTimeEnd(5)
#print 'Unloading shared library'
#print rr.Unload(rr.handle)

rr.loadPlugins()
print rr.getPluginInfo("TestPlugin")

simResult = rr.simulate()
print simResult

print "done"
