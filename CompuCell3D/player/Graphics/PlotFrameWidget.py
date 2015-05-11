import sys
from PyQt4 import Qt
import PyQt4.Qwt5 as Qwt
from PyQt4.Qwt5.anynumpy import *


class CartesianAxis(Qwt.QwtPlotItem):
    """Supports a coordinate system similar to 
    http://en.wikipedia.org/wiki/Image:Cartesian-coordinate-system.svg
    """

    def __init__(self, masterAxis, slaveAxis):
        """Valid input values for masterAxis and slaveAxis are QwtPlot.yLeft,
        QwtPlot.yRight, QwtPlot.xBottom, and QwtPlot.xTop. When masterAxis is
        an x-axis, slaveAxis must be an y-axis; and vice versa.
        """
        Qwt.QwtPlotItem.__init__(self)
        self.__axis = masterAxis
        if masterAxis in (Qwt.QwtPlot.yLeft, Qwt.QwtPlot.yRight):
            self.setAxis(slaveAxis, masterAxis)
        else:
            self.setAxis(masterAxis, slaveAxis)
        self.scaleDraw = Qwt.QwtScaleDraw()
        self.scaleDraw.setAlignment((Qwt.QwtScaleDraw.LeftScale,
                                     Qwt.QwtScaleDraw.RightScale,
                                     Qwt.QwtScaleDraw.BottomScale,
                                     Qwt.QwtScaleDraw.TopScale)[masterAxis])

    # __init__()

    def draw(self, painter, xMap, yMap, rect):
        """Draw an axis on the plot canvas
        """
        if self.__axis in (Qwt.QwtPlot.yLeft, Qwt.QwtPlot.yRight):
            self.scaleDraw.move(round(xMap.xTransform(0.0)), yMap.p2())
            self.scaleDraw.setLength(yMap.p1()-yMap.p2())
        elif self.__axis in (Qwt.QwtPlot.xBottom, Qwt.QwtPlot.xTop):
            self.scaleDraw.move(xMap.p1(), round(yMap.xTransform(0.0)))
            self.scaleDraw.setLength(xMap.p2()-xMap.p1())
        self.scaleDraw.setScaleDiv(self.plot().axisScaleDiv(self.__axis))
        self.scaleDraw.draw(painter, self.plot().palette())

    # draw()

# class CartesianAxis


class CartesianPlot(Qwt.QwtPlot):
    """Creates a coordinate system similar system 
    http://en.wikipedia.org/wiki/Image:Cartesian-coordinate-system.svg
    """

    def __init__(self, *args):        
        Qwt.QwtPlot.__init__(self, *args)
        self.setTitle('Cartesian Coordinate System Demo')

        # create a plot with a white canvas
        self.setCanvasBackground(Qt.white)
        # legend = Qwt.QwtLegend()
        # legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
        # legend.setItemMode(Qwt.QwtLegend.ClickableItem)
        # self.insertLegend(legend, Qwt.QwtPlot.BottomLegend)
        # self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))
        super(CartesianPlot,self).setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))
        
        self.replot()
        self.sizeHint =  QtCore.QSize(300, 300)
        
    def sizeHint(self):
        return self.sizeHint
        # return QtCore.QSize(300, 300)
        
    def minimumSizeHint(self):
        return QtCore.QSize(100, 100)

    # def resizeEvent(self, ev):
        # print "resize event"
        
        # w = self.width()
        # h = self.height()

        # print 'w,h = ', (w,h)
        # # import time
        # # time.sleep(2)
        
        # self.sizeHint =  QtCore.QSize(w+200, h+200)
        # super(CartesianPlot,self).resizeEvent(ev)
        # # self.resize(self.sizeHint)

        
        
import sys
import os
import string


from PyQt4 import QtCore, QtGui,QtOpenGL
import vtk
from PyQt4.QtCore import *
from PyQt4.QtGui import *




class PlotFrameWidget(QtGui.QFrame):

    def __init__(self, parent=None):
        QtGui.QFrame.__init__(self, parent)

        self.plotWidget=CartesianPlot()
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))

        self.plotInterface = None

        self.parentWidget=parent
        layout=QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
        layout.addWidget(self.plotWidget)
        self.setLayout(layout)
        # self.resize(600, 600)
        self.setMinimumSize(100, 100) #needs to be defined to resize smaller than 400x400
        
        # self.resizePlot(400, 300)
        # self.resizePlot(600, 600)
        
    # def resizeEvent (self,ev):
        # print 'RESIZE EVENT PLOT FRAM WIDGET=',ev
        
    def resizePlot(self,x,y):    
        self.plotWidget.sizeHint = QtCore.QSize(x, y)
        self.plotWidget.resize(self.plotWidget.sizeHint)
        self.resize(self.plotWidget.sizeHint)
        
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
    def closeEvent(self, ev):
        pass
        # self.parentWidget.closeActiveSubWindowSlot()
        
        
    
