import rrPython

rrPython.loadSBMLFromFile('C:\\roadRunner\\models\\simple.xml')
rrPython.setSteadyStateSelectionList('time S1 S2')
results = rrPython.simulateEx(0.0,2.0,20)

print results