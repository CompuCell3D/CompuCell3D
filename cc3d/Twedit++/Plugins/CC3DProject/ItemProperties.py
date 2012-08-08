import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_itemproperties
import sys
import os.path

MAC = "qt_mac_set_native_menubar" in dir()

class ItemProperties(QDialog,ui_itemproperties.Ui_ItemProperties):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,parent=None):
        super(ItemProperties, self).__init__(parent)
        self.cc3dProjectTreeWidget=parent
        self.resourceReference=None
        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows 
        if sys.platform.startswith('win'): 
            self.setWindowFlags(Qt.Drawer) # dialogs without context help - only close button exists
        # self.gotolineSignal.connect(self.editorWindow.goToLine)
        self.projectPath=""
        self.setupUi(self)
        
  
        
        # if not MAC:
            # self.cancelButton.setFocusPolicy(Qt.NoFocus)
        # self.updateUi()

    # @pyqtSignature("") # signature of the signal emited by the button
    # def on_okButton_clicked(self):
        # self.findChangedConfigs()        
        # self.close()

    def setResourceReference(self,_ref):
        self.resourceReference=_ref
        print "\n\n\n\n\n\n self.resourceReference=",self.resourceReference
    #initialize properties dialog
    def updateUi(self):
        self.pathLabel.setText(self.resourceReference.path)
        self.typeLabel.setText(self.resourceReference.type)
        self.moduleLE.setText(self.resourceReference.module)
        self.originLE.setText(self.resourceReference.origin)
        self.copyCHB.setChecked(self.resourceReference.copy)
        # self.locationLE.setText("Simulation/") # default storage of simulation files
        
        # tw=self.cc3dProjectTreeWidget
        # projItem=tw.getProjectParent(tw.currentItem())
        
        # pdh=None
        # try:
            # pdh=tw.plugin.projectDataHandlers[projItem]        
            
        # except LookupError,e:
            # # could not find simulation data handler for this item
            # return        
            
        # self.projectPath = pdh.cc3dSimulationData.basePath
        
        # self.projectDirLE.setText(pdh.cc3dSimulationData.basePath)
        # # construct a list of available file types
        # if pdh.cc3dSimulationData.xmlScript=="":
            # self.fileTypeCB.insertItem(0,"XML Script")
        # if pdh.cc3dSimulationData.pythonScript=="":
            # self.fileTypeCB.insertItem(0,"Main Python Script")
            
        return
        