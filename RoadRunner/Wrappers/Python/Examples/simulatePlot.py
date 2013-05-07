import rrPython
import numpy
import matplotlib.pyplot as plot

model = open('C:\\roadRunner\\models\\feedback.xml', 'r').read()
rrPython.loadSBML(model)
timeStart = 0.0
timeEnd = 10.0
numPoints = 50
results = rrPython.simulateEx(timeStart, timeEnd, numPoints)
print results

S1 = results[:,2]
S2 = results[:,3]
S3 = results[:,4]
x = numpy.arange(timeStart, timeEnd, (timeEnd - timeStart)/numPoints)
plot.plot(x, S1, label="S1")
plot.plot(x, S2, label="S2")
plot.plot(x, S3, label="S3")
plot.legend(bbox_to_anchor=(1.05, 1), loc=5, borderaxespad=0.)
plot.ylabel('Concentration (moles/L)')
plot.xlabel('time (s)')

plot.show()
