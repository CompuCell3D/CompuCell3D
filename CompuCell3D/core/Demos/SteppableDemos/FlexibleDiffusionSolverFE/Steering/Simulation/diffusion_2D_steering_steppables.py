from PySteppables import *
import CompuCell
import sys
from XMLUtils import dictionaryToMapStrStr as d2mss
from XMLUtils import CC3DXMLListPy 



class DiffusionSolverSteering(SteppablePy):
    def __init__(self,_simulator,_frequency=100):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
    def step(self, mcs):
        if mcs>100:
            # get <Steppable Type="FlexibleDiffusionSolverFE"> section of XML file 
            fiexDiffXMLData=self.simulator.getCC3DModuleData("Steppable","FlexibleDiffusionSolverFE")
            # check if we were able to successfully get the section from simulator
            if fiexDiffXMLData:
                # get a list of <DiffusionField> elements  - notice how we use CC3DXMLListPy to construct Python iterable list
                diffusionFieldsElementVec=CC3DXMLListPy(fiexDiffXMLData.getElements("DiffusionField"))
                # go over all <DiffusionFields> elements - here we have only one such element but if you have more fields in the solver 
                # you will need chek all of them to find required  element
                for diffusionFieldElement in diffusionFieldsElementVec:
                    # here we assume that <DiffusionData>
                    #                              <FieldName>xxx</FieldName> element exists and pick the one whose value is "FGF"
                    if diffusionFieldElement.getFirstElement("DiffusionData").getFirstElement("FieldName").getText()=="FGF":
                        # we get diffusion constant element , again no checking if such element exists - we assume it does
                        diffConstElement=diffusionFieldElement.getFirstElement("DiffusionData").getFirstElement("DiffusionConstant")
                        # convert string value of <DiffusionConstant> element  to float
                        diffConst=float(diffConstElement.getText())
                        # increase diffusion constant
                        diffConst+=0.01
                        # update the value of the <DiffusionConstant> element - convert flot to string
                        diffConstElement.updateElementValue(str(diffConst))
                        # getting <SecretionData> section
                        if mcs>500:
                            # get  <SecretionData>
                            #          <Secretion Type="Bacterium">2</Secretion>
                            # notice we skip checking if the attemt was sucessful and assume it was
                            secretionElement=diffusionFieldElement.getFirstElement("SecretionData").getFirstElement("Secretion",d2mss({"Type":"Bacterium"}))
                            secretionConst=float(secretionElement.getText())
                            #increase secretion of Bacterium by 2
                            secretionConst+=2
                            print "secretionConst=",secretionConst
                            #update value of the <Secretion Type="Bacterium">
                            secretionElement.updateElementValue(str(secretionConst))
                # finally call simulator.updateCC3DModule(fiexDiffXMLData) to tell simulator to update model parameters - this is actual steering                            
                self.simulator.updateCC3DModule(fiexDiffXMLData)    
                    
                    
