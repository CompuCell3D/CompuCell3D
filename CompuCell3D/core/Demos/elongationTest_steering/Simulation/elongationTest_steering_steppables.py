from PySteppables import *
import CompuCell
import sys
from XMLUtils import dictionaryToMapStrStr as d2mss



            
class LengthConstraintSteering(SteppablePy):
    def __init__(self,_simulator,_frequency=100):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator

    def step(self,mcs):
        if mcs>100 and not mcs%100:
            # get <Plugin Name="LengthConstraint"> section of XML file 
            lengthConstraintXMLData=self.simulator.getCC3DModuleData("Plugin","LengthConstraint")
            # check if we were able to successfully get the section from simulator
            if lengthConstraintXMLData:
                # get <LengthEnergyParameters CellType="Body1" TargetLength="xxx" LambdaLength="xxx" /> element  
                lengthEnergyParametersBody1=lengthConstraintXMLData.getFirstElement("LengthEnergyParameters",d2mss({"CellType":"Body1"}))
                # check if the attempt was succesful
                if lengthEnergyParametersBody1:
                    # get value of the TargetLength attribute from <LengthEnergyParameters CellType="Body1" TargetLength="xxx" LambdaLength="xxx" /> element  
                    targetLength=lengthEnergyParametersBody1.getAttributeAsDouble("TargetLength")
                    #increase targetLength by 1
                    targetLength+=0.5
                    # update <LengthEnergyParameters CellType="Body1" TargetLength="xxx" LambdaLength="xxx" /> element remembering abuot converting targetLength to string
                    lengthEnergyParametersBody1.updateElementAttributes(d2mss({"TargetLength":str(targetLength)}))
                # finally call simulator.updateCC3DModule(lengthConstraintXMLData) to tell simulator to update model parameters - this is actual steering        
                self.simulator.updateCC3DModule(lengthConstraintXMLData)
        if mcs>3000:
            # here we relax connectivity constraint at MCS>3000
            connectivityXMLData=self.simulator.getCC3DModuleData("Plugin","Connectivity")
            if connectivityXMLData:
                penaltyElement=connectivityXMLData.getFirstElement("Penalty")
                if penaltyElement:
                    penaltyElement.updateElementValue(str(0))
                 
                self.simulator.updateCC3DModule(connectivityXMLData)
            
        

