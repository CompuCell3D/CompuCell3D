from PySteppables import *
import CompuCell
import sys
from XMLUtils import dictionaryToMapStrStr as d2mss
            
class ChemotaxisSteering(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        

    def step(self,mcs):
        if mcs>100 and not mcs%100:
        
            attrVal=float(self.getXMLAttributeValue('Lambda',['Plugin','Name','Chemotaxis'],['ChemicalField','Name','ATTR'],['ChemotaxisByType','Type','Macrophage']))    
            self.setXMLAttributeValue('Lambda',attrVal-3,['Plugin','Name','Chemotaxis'],['ChemicalField','Name','ATTR'],['ChemotaxisByType','Type','Macrophage'])
            self.updateXML()
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
# OLD STYLE STEERING

class ChemotaxisSteeringOldStyle(SteppablePy):
    def __init__(self,_simulator,_frequency=100):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator

    def step(self,mcs):
        if mcs>100 and not mcs%100:
            # get <Plugin Name="Chemotaxis"> section of XML file 
            chemotaxisXMLData=self.simulator.getCC3DModuleData("Plugin","Chemotaxis")
            # check if we were able to successfully get the section from simulator
            if chemotaxisXMLData:
                # get <ChemicalField Source="DiffusionSolverFE" Name="ATTR" > element  
                chemicalField=chemotaxisXMLData.getFirstElement("ChemicalField",d2mss({"Source":"DiffusionSolverFE", "Name":"ATTR"}))
                # check if the attempt was succesful
                if chemicalField:
                    # get <ChemotaxisByType Type="Macrophage" Lambda="xxx"/> - notice we only specify "Type":"Macrophage" because Lambda is subject to change - i.e. this is steerable parameter
                    chemotaxisByTypeMacrophageElement=chemicalField.getFirstElement("ChemotaxisByType",d2mss({"Type":"Macrophage"}))
                    if chemotaxisByTypeMacrophageElement:
                        # get value of Lambda from <ChemotaxisByType Type="Macrophage" Lambda="xxx"/>
                        # notice that no conversion fro strin to float is necessary as getAttributeAsDouble returns floating point value
                        lambdaVal=chemotaxisByTypeMacrophageElement.getAttributeAsDouble("Lambda")
                        print "lambdaVal=",lambdaVal
                        # decrease lambda by 0.2
                        lambdaVal-=3
                        # update attribute value of Lambda - but remember about float to string conversion
                        chemotaxisByTypeMacrophageElement.updateElementAttributes(d2mss({"Lambda":str(lambdaVal)}))
            self.simulator.updateCC3DModule(chemotaxisXMLData);
        

        