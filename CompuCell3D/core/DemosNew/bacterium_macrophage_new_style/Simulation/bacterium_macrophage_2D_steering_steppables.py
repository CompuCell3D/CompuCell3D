# from PySteppables import *
# import CompuCell
# import sys
# from XMLUtils import dictionaryToMapStrStr as d2mss

from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
import sys
import time



class InventoryCheckSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency=frequency)
            
    def start(self):        
        print("INSIDE START FUNCTION")
        print('view_manager = ', CompuCellSetup.persistent_globals.view_manager)

        self.pW=self.add_new_plot_window(title='Average Volume And Surface', xAxisTitle='MonteCarlo Step (MCS)', yAxisTitle='Variables', xScaleType='linear', yScaleType='linear', grid=False)
		
        self.pW.addPlot("MVol", _style='Lines',_color='red')
        self.pW.addPlot("MSur", _style='Dots',_color='green')
        
        # adding automatically generated legend
        # default possition is at the bottom of the plot but here we put it at the top
        self.pW.addAutoLegend("top")



    def step(self, mcs):

        if not self.pW:
            print("To get scientific plots working you need extra packages installed: numpy pyqtgraph")
            return


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
                print ("Adding meanVolume=", meanVolume)
                print ("plotData=", self.pW.plotData["MVol"])


        self.pW.showAllPlots()



        # print("running mcs=", mcs)
        # for i, cell in enumerate(self.cellList):
        #     if i > 3:
        #         break
        #     # print ('cell=', cell)
        #     print ('cell.id=', cell.id)

        if mcs ==300:
            # CompuCellSetup.stop_simulation()
            self.stop_simulation()

# class ChemotaxisSteering(SteppableBasePy):
#     def __init__(self,_simulator,_frequency=100):
#         SteppableBasePy.__init__(self,_simulator,_frequency)
        

#     def step(self,mcs):
#         if mcs>100 and not mcs%100:
        
#             attrVal=float(self.getXMLAttributeValue('Lambda',['Plugin','Name','Chemotaxis'],['ChemicalField','Name','ATTR'],['ChemotaxisByType','Type','Macrophage']))    
#             self.setXMLAttributeValue('Lambda',attrVal-3,['Plugin','Name','Chemotaxis'],['ChemicalField','Name','ATTR'],['ChemotaxisByType','Type','Macrophage'])
#             self.updateXML()
        
# #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
# # OLD STYLE STEERING

# class ChemotaxisSteeringOldStyle(SteppablePy):
#     def __init__(self,_simulator,_frequency=100):
#         SteppablePy.__init__(self,_frequency)
#         self.simulator=_simulator

#     def step(self,mcs):
#         if mcs>100 and not mcs%100:
#             # get <Plugin Name="Chemotaxis"> section of XML file 
#             chemotaxisXMLData=self.simulator.getCC3DModuleData("Plugin","Chemotaxis")
#             # check if we were able to successfully get the section from simulator
#             if chemotaxisXMLData:
#                 # get <ChemicalField Source="DiffusionSolverFE" Name="ATTR" > element  
#                 chemicalField=chemotaxisXMLData.getFirstElement("ChemicalField",d2mss({"Source":"DiffusionSolverFE", "Name":"ATTR"}))
#                 # check if the attempt was succesful
#                 if chemicalField:
#                     # get <ChemotaxisByType Type="Macrophage" Lambda="xxx"/> - notice we only specify "Type":"Macrophage" because Lambda is subject to change - i.e. this is steerable parameter
#                     chemotaxisByTypeMacrophageElement=chemicalField.getFirstElement("ChemotaxisByType",d2mss({"Type":"Macrophage"}))
#                     if chemotaxisByTypeMacrophageElement:
#                         # get value of Lambda from <ChemotaxisByType Type="Macrophage" Lambda="xxx"/>
#                         # notice that no conversion fro strin to float is necessary as getAttributeAsDouble returns floating point value
#                         lambdaVal=chemotaxisByTypeMacrophageElement.getAttributeAsDouble("Lambda")
#                         print "lambdaVal=",lambdaVal
#                         # decrease lambda by 0.2
#                         lambdaVal-=3
#                         # update attribute value of Lambda - but remember about float to string conversion
#                         chemotaxisByTypeMacrophageElement.updateElementAttributes(d2mss({"Lambda":str(lambdaVal)}))
#             self.simulator.updateCC3DModule(chemotaxisXMLData);
        

        