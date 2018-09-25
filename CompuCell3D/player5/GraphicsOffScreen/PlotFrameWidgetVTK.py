import sys
#from PyQt4 import Qt
from PyQt4 import QtCore, QtGui
import vtk
import math

if sys.platform=='darwin':    
    from Utilities.QVTKRenderWindowInteractor_mac import QVTKRenderWindowInteractor
else:    
    from Utilities.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class PlotWidget(QtGui.QWidget):   # rf. QVTKRenderWindowInteractor
    def __init__(self, parent=None, wflags=QtCore.Qt.WindowFlags(), **kw):
        QtGui.QWidget.__init__(self, parent, wflags|QtCore.Qt.MSWindowsOwnDC)

    def setTitle(self):
        pass
        

class PlotFrameWidget(QtGui.QFrame):
    def __init__(self, parent=None):
#        QtGui.QFrame.__init__(self, parent)
#
##        self.plotWidget = CartesianPlot()
#        self.plotWidget = PlotWidget()
#        
#        self.parentWidget = parent
#        layout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
#        layout.addWidget(self.plotWidget)
#        self.setLayout(layout)
#        self.resize(400, 400)
#        self.setMinimumSize(200, 200) #needs to be defined to resize smaller than 400x400


        QtGui.QFrame.__init__(self, parent)
        self.qvtkWidget = QVTKRenderWindowInteractor(self)   # a QWidget
        self.parentWidget = parent
        
#        self.lineEdit = QtGui.QLineEdit()
        
#        self.__initCrossSectionActions()
#        self.cstb = self.initCrossSectionToolbar()        
        
        layout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
#        layout.addWidget(self.cstb)
        layout.addWidget(self.qvtkWidget)
        self.setLayout(layout)
        
        self.qvtkWidget.Initialize()
        self.qvtkWidget.Start()
        
#        self.ren = vtk.vtkRenderer()
#        self.renWin = self.qvtkWidget.GetRenderWindow()
#        self.renWin.AddRenderer(self.ren)
#        self.resize(300, 300)


        self.chart = vtk.vtkChartXY()
        self.view = vtk.vtkContextView()
        self.ren = self.view.GetRenderer()
        self.renWin = self.qvtkWidget.GetRenderWindow()
        self.renWin.AddRenderer(self.ren)

        # Create a table with some points in it
        table = vtk.vtkTable()

        arrX = vtk.vtkFloatArray()
        arrX.SetName("X Axis")

        arrC = vtk.vtkFloatArray()
        arrC.SetName("Cosine")

        numPoints = 20
        inc = 7.5 / (numPoints - 1)

        for i in range(0,numPoints):
            arrX.InsertNextValue(i*inc)
            arrC.InsertNextValue(math.cos(i * inc) + 0.0)

        table.AddColumn(arrX)
        table.AddColumn(arrC)

        # Now add the line plots with appropriate colors
        line = self.chart.AddPlot(0)
        line.SetInput(table,0,1)
        line.SetColor(0,0,255,255)
        line.SetWidth(2.0)


#        self.view.GetRenderer().SetBackground([0.6,0.6,0.1])
#        self.view.GetScene().AddItem(self.chart)

        
    # def __getattr__(self, attr):
        # """Makes the object behave like a DrawBase"""
        # if not self.draw3DFlag:
            # if hasattr(self.draw2D, attr):            
                # return getattr(self.draw2D, attr)
            # else:
                # raise AttributeError, self.__class__.__name__ + \
                      # " has no attribute named " + attr
        # else:
            # if hasattr(self.draw3D, attr):            
                # return getattr(self.draw3D, attr)
            # else:
                # raise AttributeError, self.__class__.__name__ + \
                      # " has no attribute named " + attr
            
        # # print "FINDING ATTRIBUTE ", attr
        # # self.getattrFcn(attr)


    # # note that if you close widget using X button this slot is not called
    # # we need to reimplement closeEvent
    # # def close(self):           
    def closeEvent(self,ev):
        self.parentWidget.closeActiveSubWindowSlot()
