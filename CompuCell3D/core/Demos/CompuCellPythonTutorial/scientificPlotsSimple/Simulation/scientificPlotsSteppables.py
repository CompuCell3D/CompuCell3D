from PySteppables import *
import CompuCell
import CompuCellSetup
import time
import sys


class ExtraPlotSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def start(self):
        _config_options = {'background': 'white', 'legend': True}

        self.pW = self.addNewPlotWindow(_title='Average Volume And Surface', _xAxisTitle='MonteCarlo Step (MCS)',
                                        _yAxisTitle='Variables', _xScaleType='linear', _yScaleType='linear',
                                        _config_options=_config_options)
        self.pW.addPlot('MVol', _style='Dots', _color='red', _size=5)
        self.pW.addPlot('MSur', _style='Bars', _size=0.2)

    def step(self, mcs):

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
            self.pW.addDataPoint('MVol', mcs, meanVolume)
            self.pW.addDataPoint('MSur', mcs, meanSurface)
            if mcs >= 200:
                print 'Adding meanVolume=', meanVolume
                print 'plotData=', self.pW.plotData['MVol']

        # Saving plots as PNG's
        if mcs < 50:
            fileName = 'ExtraPlots_' + str(mcs) + '.png'
            self.pW.savePlotAsPNG(fileName, 1000,
                                  1000)  # here we specify size of the image saved - default is 400 x 400
            self.pW.savePlotAsData(fileName + '.txt', CSV_FORMAT)


class ExtraMultiPlotSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def start(self):
        _config_options_1 = {'background': 'white111', 'legend': False}
        _config_options_2 = {'background': 'green', 'legend': True}

        self.pWVol = self.addNewPlotWindow(_title='Average Volume', _xAxisTitle='MonteCarlo Step (MCS)',
                                           _yAxisTitle='Average Volume', _config_options=_config_options_1)
        self.pWVol.addPlot(_plotName='MVol', _style='Dots', _color='red', _size=5)
        self.pWSur = self.addNewPlotWindow(_title='Average Surface', _xAxisTitle='MonteCarlo Step (MCS)',
                                           _yAxisTitle='Average Surface', _config_options=_config_options_2)
        self.pWSur.addPlot(_plotName='MSur')

    def step(self, mcs):
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
