from PySteppables import *
import CompuCell
import CompuCellSetup
import sys


class ExtraPlotSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def start(self):

        self.pW=self.addNewPlotWindow(_title='Average Volume And Surface',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Variables', _xScaleType='linear',_yScaleType='linear',_grid=False)
		
        self.pW.addPlot("MVol", _style='Lines',_color='red')
        self.pW.addPlot("MSur", _style='Dots',_color='green')
        
        # adding automatically generated legend
        # default possition is at the bottom of the plot but here we put it at the top
        self.pW.addAutoLegend("top")

        self.clearFlag = False

    def step(self, mcs):
        if not self.pW:
            print "To get scientific plots working you need extra packages installed:"
            print "Windows/OSX Users: Make sure you have numpy installed. For instructions please visit www.compucell3d.org/Downloads"
            print "Linux Users: Make sure you have numpy and PyQwt installed. Please consult your linux distributioun manual pages on how to best install those packages"
            return
            # self.pW.addDataPoint("MCS",mcs,mcs)

        # self.pW.addDataPoint("MCS1",mcs,-2*mcs)
        # this is totall non optimized code. It is for illustrative purposes only. 
        meanSurface = 0.0
        meanVolume = 0.0
        numberOfCells = 0
        for cell in self.cellList:
            meanVolume += cell.volume
            meanSurface += cell.surface
            numberOfCells += 1
        meanVolume /= float(numberOfCells)
        meanSurface /= float(numberOfCells)



        if mcs > 100 and mcs < 200:
            self.pW.eraseAllData()
        else:
            self.pW.addDataPoint("MVol", mcs, meanVolume)
            self.pW.addDataPoint("MSur", mcs, meanSurface)
            if mcs >= 200:
                print "Adding meanVolume=", meanVolume
                print "plotData=", self.pW.plotData["MVol"]


        self.pW.showAllPlots()

        # Saving plots as PNG's
        if mcs < 50:
            qwtPlotWidget = self.pW.getQWTPLotWidget()
            qwtPlotWidgetSize = qwtPlotWidget.size()
            # print "pW.size=",self.pW.size()
            fileName = "ExtraPlots_" + str(mcs) + ".png"
            self.pW.savePlotAsPNG(fileName, 550, 550)  # here we specify size of the image saved - default is 400 x 400


class ExtraMultiPlotSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def start(self):

        self.pWVol=self.addNewPlotWindow(_title='Average Volume ',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Volume', _xScaleType='linear',_yScaleType='linear')
        
        self.pWVol.addPlot("MVol", _style='Dots',_color='blue')
        
        if not self.pWVol:
            return
        
        # adding automatically generated legend
        self.pWVol.addAutoLegend()
       
        self.pWSur=self.addNewPlotWindow(_title='Average Surface',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Surface', _xScaleType='linear',_yScaleType='linear')
        self.pWSur.addPlot("MSur", _style='Dots', _color='red')


    def step(self, mcs):
        # this is totally non optimized code. It is for illustrative purposes only. 
        meanSurface = 0.0
        meanVolume = 0.0
        numberOfCells = 0
        for cell in self.cellList:
            meanVolume += cell.volume
            meanSurface += cell.surface
            numberOfCells += 1
        meanVolume /= float(numberOfCells)
        meanSurface /= float(numberOfCells)

        self.pWVol.addDataPoint("MVol", mcs, meanVolume)
        self.pWSur.addDataPoint("MSur", mcs, meanSurface)
        print "meanVolume=", meanVolume, "meanSurface=", meanSurface

        # self.pW.showPlot("MCS1")
        self.pWVol.showAllPlots()
        self.pWSur.showAllPlots()
