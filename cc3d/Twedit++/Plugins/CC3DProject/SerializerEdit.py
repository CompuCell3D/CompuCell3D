import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_serializereditdlg
import sys
import os.path

MAC = "qt_mac_set_native_menubar" in dir()

class SerializerEdit(QDialog,ui_serializereditdlg.Ui_SerializerEditDlg):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,parent=None):
        super(SerializerEdit, self).__init__(parent)
        self.setupUi(self)
  
        
        self.updateUi()

        if not self.enableSerializationCHB.isChecked():
            self.outputGB.setEnabled(False)

        
    def setupDialog(self,_serializerResource):
        sr=_serializerResource
        if sr.outputFrequency>0:
            self.enableSerializationCHB.setChecked(True)
            self.frequencySB.setValue(sr.outputFrequency)
        else:    
            
            self.frequencySB.setValue(1)
            self.outputGB.setEnabled(False)
            
            self.enableSerializationCHB.setChecked(False)
            
            
        if sr.fileFormat.lower()=='text':
            self.fileFormatCB.setCurrentIndex(0)
        elif sr.fileFormat.lower()=='binary':    
            self.fileFormatCB.setCurrentIndex(1)
        else:
            self.fileFormatCB.setCurrentIndex(0)
            
        self.multipleDirCHB.setChecked(sr.allowMultipleRestartDirectories)
        
        if sr.restartDirectory!='':
            self.enableRestartCHB.setChecked(True)
            
    
    
    def modifySerializerResource(self,_serializerResource):
        
        sr=_serializerResource
        if self.enableSerializationCHB.isChecked(): 
            sr.outputFrequency=self.frequencySB.value()
            sr.fileFormat=str(self.fileFormatCB.currentText())        
            sr.allowMultipleRestartDirectories=self.multipleDirCHB.isChecked()
        else:
            sr.outputFrequency=0
            sr.fileFormat=''
            sr.allowMultipleRestartDirectories=False
            
        if not self.enableRestartCHB.isChecked():
            sr.disableRestart()
        else:
            sr.enableRestart()
            
            
            
    def updateUi(self):
        self.frequencySB.setMinimum(1)
        return
