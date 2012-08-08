from PySteppables import *
import CompuCell
import sys




            
class ExtraPlotSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def start(self):
        # import CompuCellSetup  
        # print "CompuCellSetup.viewManager=",CompuCellSetup.viewManager    
        # CompuCellSetup.viewManager.plotManager.addNewPlotWindow()

        import CompuCellSetup  
        #self.pW=CompuCellSetup.addNewPlotWindow()
        # CompuCellSetup.viewManager.plotManager.emitNewPlotWindow()
        
        
        # this is an example of plot customization. you do not have to include all those modifications in your script. Default settings will work fine. 
        # If you need more customization options please contact us and we will gladly add them. 
        # If you are expert user you can alway retwieve underlying QwtPlot class and using PyQwt manual implement way more cutom modification than it is currently possible with PlotWindowInterface
        # Example:
        # qwtWidget=self.pW.getQWTPLotWidget()
        # this gives you access to QwtPlot widget which is a qt object and you can manipulate directly all the properties of it using API provided here 
        # http://qwt.sourceforge.net/class_qwt_plot.html
        
        self.pW=CompuCellSetup.viewManager.plotManager.getNewPlotWindow()
        if not self.pW:
            return
        #Plot Title - properties           
        self.pW.setTitle("Average Volume And Surface")
        self.pW.setTitleSize(12)
        self.pW.setTitleColor("Green")
        
        #plot background
        self.pW.setPlotBackgroundColor("orange")
        
        # properties of x axis
        self.pW.setXAxisTitle("MonteCarlo Step (MCS)")
        self.pW.setXAxisTitleSize(10)      
        self.pW.setXAxisTitleColor("blue")              
        
        # properties of y axis
        self.pW.setYAxisTitle("Variables")        
        self.pW.setYAxisLogScale()
        self.pW.setYAxisTitleSize(10)        
        self.pW.setYAxisTitleColor("red")                      
        
        # choices for style are NoCurve,Lines,Sticks,Steps,Dots
        self.pW.addPlot("MVol",_style='Dots')
        self.pW.addPlot("MSur",_style='Steps')
        
        # plot MCS
        self.pW.changePlotProperty("MVol","LineWidth",5)
        self.pW.changePlotProperty("MVol","LineColor","red")     
        # plot MCS1
        self.pW.changePlotProperty("MSur","LineWidth",1)
        self.pW.changePlotProperty("MSur","LineColor","green")         
        
        self.pW.addGrid()
        #adding automatically generated legend
        # default possition is at the bottom of the plot but here we put it at the top
        self.pW.addAutoLegend("top")
        
        # print "\n\n\n ADDED NEW PLOT WINDOWS"
        
        self.clearFlag=False
    
    
    def step(self,mcs):
        if not self.pW:
            print "To get scientific plots working you need extra packages installed:"
            print "Windows/OSX Users: Make sure you have numpy installed. For instructions please visit www.compucell3d.org/Downloads"
            print "Linux Users: Make sure you have numpy and PyQwt installed. Please consult your linux distributioun manual pages on how to best install those packages"
            return        
        # self.pW.addDataPoint("MCS",mcs,mcs)
        
        # self.pW.addDataPoint("MCS1",mcs,-2*mcs)
        # this is totall non optimized code. It is for illustrative purposes only. 
        meanSurface=0.0
        meanVolume=0.0
        numberOfCells=0
        for cell  in  self.cellList:
            meanVolume+=cell.volume
            meanSurface+=cell.surface
            numberOfCells+=1
        meanVolume/=float(numberOfCells)
        meanSurface/=float(numberOfCells)
        
        # self.pW.addDataPoint("MVol",mcs,meanVolume)
        # self.pW.addDataPoint("MSur",mcs,meanSurface)
        # print "meanVolume=",meanVolume,"meanSurface=",meanSurface
        
        if  mcs >100 and mcs <200:
            self.pW.eraseAllData()
        else:
            self.pW.addDataPoint("MVol",mcs,meanVolume)
            self.pW.addDataPoint("MSur",mcs,meanSurface)
            if mcs>=200:
                print "Adding meanVolume=",meanVolume
                print "plotData=",self.pW.plotData["MVol"]
            

        
        
        # if  mcs >100 and mcs <200:
            # # self.pW.getQWTPLotWidget().plotWidget().clear()
            # self.pW.eraseAllData()
            # # self.pW.eraseData("MVol")
            # # self.pW.eraseData("MSur")
            # # self.pW.addDataPoint("MVol",0,0)
            # # self.pW.addDataPoint("MSur",0,0)
            # # self.clearFlag=True
            
            # # self.pW.replot()
        # else:
            # # print "ADDING meanVolume=",meanVolume
            # # print "ADDING meanSurface=",meanSurface
            
            # # if self.clearFlag:
                # # self.pW.eraseData("MVol")
                # # self.pW.eraseData("MSur")
                # # self.clearFlag=False
                
            # self.pW.addDataPoint("MVol",mcs,meanVolume)
            # self.pW.addDataPoint("MSur",mcs,meanSurface)
            
            # self.pW.clear()

        # self.pW.addDataPoint("MVol",mcs,meanVolume)
        # self.pW.addDataPoint("MSur",mcs,meanSurface)
            
        self.pW.showAllPlots()
        
        # or you can individually show/update particular curves of the plot
        # self.pW.showPlot("MVol")
        # self.pW.showPlot("MSur")
        
        #Saving plots as PNG's
        if mcs<50:            
            qwtPlotWidget=self.pW.getQWTPLotWidget()
            qwtPlotWidgetSize=qwtPlotWidget.size()
            # print "pW.size=",self.pW.size()
            fileName="ExtraPlots_"+str(mcs)+".png"
            self.pW.savePlotAsPNG(fileName,550,550) # here we specify size of the image saved - default is 400 x 400
        
        
class ExtraMultiPlotSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def start(self):
        # import CompuCellSetup  
        # print "CompuCellSetup.viewManager=",CompuCellSetup.viewManager    
        # CompuCellSetup.viewManager.plotManager.addNewPlotWindow()

        import CompuCellSetup  
        #self.pW=CompuCellSetup.addNewPlotWindow()
        # CompuCellSetup.viewManager.plotManager.emitNewPlotWindow()
        
        self.pWVol=CompuCellSetup.viewManager.plotManager.getNewPlotWindow()    
        if not self.pWVol:
            return         
        self.pWVol.setTitle("Average Volume")        
        self.pWVol.setXAxisTitle("MonteCarlo Step (MCS)")
        self.pWVol.setYAxisTitle("Average Volume")        
        self.pWVol.addPlot("MVol")
        self.pWVol.changePlotProperty("MVol","LineWidth",5)
        self.pWVol.changePlotProperty("MVol","LineColor","red")     
        self.pWVol.addGrid()
        #adding automatically generated legend
        self.pWVol.addAutoLegend()
        
        self.pWSur=CompuCellSetup.viewManager.plotManager.getNewPlotWindow()        
        self.pWSur.setTitle("Average Surface")        
        self.pWSur.setXAxisTitle("MonteCarlo Step (MCS)")        
        self.pWSur.setYAxisTitle("Average Surface")        
        self.pWSur.addPlot("MSur")
        self.pWSur.changePlotProperty("MSur","LineWidth",1)
        self.pWSur.changePlotProperty("MSur","LineColor","green")         
        self.pWSur.addGrid()
        
        
        

        
    def step(self,mcs):
        if not self.pWVol:
            return         
        
        # self.pW.addDataPoint("MCS",mcs,mcs)
        
        # self.pW.addDataPoint("MCS1",mcs,-2*mcs)
        # this is totall non optimized code. It is for illustrative purposes only. 
        meanSurface=0.0
        meanVolume=0.0
        numberOfCells=0
        for cell  in  self.cellList:
            meanVolume+=cell.volume
            meanSurface+=cell.surface
            numberOfCells+=1
        meanVolume/=float(numberOfCells)
        meanSurface/=float(numberOfCells)
        
        self.pWVol.addDataPoint("MVol",mcs,meanVolume)
        self.pWSur.addDataPoint("MSur",mcs,meanSurface)
        print "meanVolume=",meanVolume,"meanSurface=",meanSurface
        
        # self.pW.showPlot("MCS1")
        self.pWVol.showAllPlots()
        self.pWSur.showAllPlots()
