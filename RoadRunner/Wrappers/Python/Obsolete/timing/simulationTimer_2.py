import rrPython
import timeit
import cmd

print rrPython.getCopyright()
modelPath = 'R:\\roadrunnerwork\\Models\\BorisEJB.xml'
rrPython.setTempFolder('R:\\rrTemp')
rrPython.enableLogging()
print 'Temp folder is:' + rrPython.getTempFolder()

rrPython.setLogLevel('Info')
level = rrPython.getLogLevel()
print 'Log level is ' + str(level)
rrPython.loadSBMLFromFile(modelPath)

simulations = 10
start_Time = 0
end_Time = 2
number_of_points = 1000

rrPython.setTimeStart(start_Time)
rrPython.setTimeEnd(end_Time)
rrPython.setNumPoints(number_of_points)

t = timeit.Timer('rrPython.simulate()','import rrPython')
totalTime = t.timeit(number = simulations)
meanTime = totalTime/simulations

print 'Average simulation time: ' + str(meanTime) + ' seconds'
