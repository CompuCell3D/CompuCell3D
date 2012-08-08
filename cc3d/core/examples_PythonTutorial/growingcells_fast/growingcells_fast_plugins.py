from PyPlugins import *

from PyPluginsExamples import MitosisPyPluginBase
class MitosisPyPlugin(MitosisPyPluginBase):
    def __init__(self , _simulator , _changeWatcherRegistry , _stepperRegistry):
        MitosisPyPluginBase.__init__(self,_simulator,_changeWatcherRegistry, _stepperRegistry)
    def updateAttributes(self):
        self.childCell.targetVolume=self.parentCell.targetVolume
        self.childCell.lambdaVolume=self.parentCell.lambdaVolume

        if self.parentCell.type==1:
            self.childCell.type=2
        else:
            self.childCell.type=1

