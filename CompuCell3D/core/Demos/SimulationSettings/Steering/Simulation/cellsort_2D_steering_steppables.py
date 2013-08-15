from PySteppables import *
import CompuCell
import sys

            
class ContactSteeringAndTemperature(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def step(self,mcs):
        
        temp=float(self.getXMLElementValue(['Potts'],['Temperature']))
        self.setXMLElementValue(temp+1,['Potts'],['Temperature'])    
        
        val=float(self.getXMLElementValue(['Plugin','Name','Contact'],['Energy','Type1','NonCondensing','Type2','Condensing']))       
        
        self.setXMLElementValue(val+1,['Plugin','Name','Contact'],['Energy','Type1','NonCondensing','Type2','Condensing']) 
                
        self.updateXML()    
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
# OLD STYLE STEERING
class PottsSteeringOldStyle(SteppablePy):
    def __init__(self,_simulator,_frequency=1):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        
    def step(self, _mcs):
        # get Potts section of XML file 
        pottsXMLData=self.simulator.getCC3DModuleData("Potts")
        # check if we were able to successfully get the section from simulator
        if pottsXMLData:
            # get Temperature XML element
            temperatureElement=pottsXMLData.getFirstElement("Temperature")
            # check if the attempt was succesful
            if temperatureElement:
                # get value of the temperature and convert it to a floating point number - it is  important to remember abuot proper conversions
                temperature=float(temperatureElement.getText())
                # increase temperature by 1.0
                temperature+=1.0
                # update XML Temperature element by converting floating point number to string and calling updateElementValue function
                temperatureElement.updateElementValue(str(temperature))
            # finally call simulator.updateCC3DModule(pottsXMLData) to tell simulator to update model parameters - this is actual steering    
            self.simulator.updateCC3DModule(pottsXMLData)    
        
            
            
class ContactSteeringOldStyle(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator

    def step(self,mcs):
        # get <Plugin Name="Contact"> section of XML file 
        contactXMLData=self.simulator.getCC3DModuleData("Plugin","Contact")
        # check if we were able to successfully get the section from simulator
        if contactXMLData:
            # get <Energy Type1="NonCondensing" Type2="Condensing"> element  
            energyNonCondensingCondensingElement=contactXMLData.getFirstElement("Energy",d2mss({"Type1":"NonCondensing","Type2":"Condensing"}))
            # check if the attempt was succesful
            if energyNonCondensingCondensingElement:
                # get value of the <Energy Type1="NonCondensing" Type2="Condensing"> element  and convert it into float
                val=float(energyNonCondensingCondensingElement.getText())
                # increase the value by 1.0
                val+=1.0
                # update <Energy Type1="NonCondensing" Type2="Condensing"> element remembering abuot converting the value bask to string
                energyNonCondensingCondensingElement.updateElementValue(str(val))
            # finally call simulator.updateCC3DModule(contactXMLData) to tell simulator to update model parameters - this is actual steering        
            self.simulator.updateCC3DModule(contactXMLData);
        

