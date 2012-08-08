from PyPlugins import *

from PyPluginsExamples import MitosisPyPluginBase
class MitosisPyPlugin(MitosisPyPluginBase):
    def __init__(self , _simulator , _changeWatcherRegistry , _stepperRegistry):
        MitosisPyPluginBase.__init__(self,_simulator,_changeWatcherRegistry, _stepperRegistry)
        self.setDivisionAlongMajorAxis()
    def updateAttributes(self):
        self.parentCell.targetVolume=50.0
        self.childCell.targetVolume=self.parentCell.targetVolume
        self.childCell.lambdaVolume=self.parentCell.lambdaVolume
        print "self.childCell.id=",self.childCell.id
        print "self.parenCell.id=",self.parentCell.id
        
        if self.parentCell.type==1:
            self.childCell.type=3
        else:
            self.childCell.type=1
        print "self.childCell.type=",self.childCell.type
        print "self.parentCell.type=",self.parentCell.type

