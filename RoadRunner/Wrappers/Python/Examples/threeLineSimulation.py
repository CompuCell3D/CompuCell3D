import rrPython
rrPython.loadSBMLFromFile('C:\\roadRunner\\models\\feedback.xml')
results = rrPython.simulateEx(0.0,2.0,20)

print results
