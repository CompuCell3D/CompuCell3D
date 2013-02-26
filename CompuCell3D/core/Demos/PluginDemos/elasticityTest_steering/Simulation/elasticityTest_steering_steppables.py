from PySteppables import *
import CompuCell
import sys
from XMLUtils import dictionaryToMapStrStr as d2mss


class ElasticitySteering(SteppablePy):
    def __init__(self,_simulator,_frequency=1):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
    def step(self, mcs):
        if mcs>100 and not mcs%100 and mcs<2000:
            # get <Plugin Name="Elasticity"> section of XML file 
            elasticityXMLData=self.simulator.getCC3DModuleData("Plugin","Elasticity")
            # check if we were able to successfully get the section from simulator
            if elasticityXMLData:
                # get <TargetLengthElasticity> element
                targetLengthElasticityElement=elasticityXMLData.getFirstElement("TargetLengthElasticity")
                # check if the attempt was succesful
                if targetLengthElasticityElement:
                    # get value of <TargetLengthElasticity> element
                    targetLengthElasticity=float(targetLengthElasticityElement.getText())
                    #increase it by 1
                    targetLengthElasticity+=1.0
                    # update value of <TargetLengthElasticity> element - converting targetLengthElasticity to string
                    targetLengthElasticityElement.updateElementValue(str(targetLengthElasticity))
            # finally call simulator.updateCC3DModule(elasticityXMLData) to tell simulator to update model parameters - this is actual steering
            self.simulator.updateCC3DModule(elasticityXMLData)
 