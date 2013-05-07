import rrPython
import os

modelPath = ('C:\\roadRunner\\models\\feedback.xml')

sbml = rrPython.loadSBMLFromFile(modelPath)
rrPython.setTimeStart(0.0)
rrPython.setTimeEnd(3.0)
rrPython.setNumPoints(20)
rrPython.setSteadyStateSelectionList("time S1 S2 S3 S4")
results = rrPython.simulate()

print results