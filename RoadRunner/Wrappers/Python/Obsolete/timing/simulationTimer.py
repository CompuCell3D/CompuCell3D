import rrPython
import timeit
import cmd

modelPath = raw_input('Model location: (C:\\RoadRunner\\Models\\feedback.xml, etc.)')

rrPython.loadSBMLFromFile(modelPath)

simulations = raw_input('Number of simulations:')
start_Time = raw_input('Start time:')
start_Time = float(start_Time)
end_Time = raw_input('End time:')
end_Time = float(end_Time)
number_of_points = raw_input('Number of points:')
number_of_points = int(number_of_points)
simulations = int(simulations)

rrPython.setTimeStart(start_Time)
rrPython.setTimeEnd(end_Time)
rrPython.setNumPoints(number_of_points)

t = timeit.Timer('rrPython.simulate()','import rrPython')

totalTime = t.timeit(number = simulations)

meanTime = totalTime/simulations

print 'Average simulation time: ' + str(meanTime) + ' seconds'