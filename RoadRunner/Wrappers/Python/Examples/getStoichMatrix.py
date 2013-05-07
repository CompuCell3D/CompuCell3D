import rrPython

modelPath = ('C:\\RoadRunner\\Models\\feedback.xml')
rrPython.loadSBMLFromFile(modelPath)
matrix = rrPython.getStoichiometryMatrix()

print matrix