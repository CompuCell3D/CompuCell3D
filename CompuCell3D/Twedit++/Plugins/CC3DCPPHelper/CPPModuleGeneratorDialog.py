import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_c_plus_plus_module_dialog
import sys
import os.path

MAC = "qt_mac_set_native_menubar" in dir()

class CPPModuleGeneratorDialog(QDialog,ui_c_plus_plus_module_dialog.Ui_C_Plus_Plus_Module_Dialog):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    # _cc3dProject=None
    def __init__(self,parent=None):
        super(CPPModuleGeneratorDialog, self).__init__(parent)
        
        self.__ui=parent
        
        # self.cc3dProject=_cc3dProject

        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows 
        if sys.platform.startswith('win'): 
            self.setWindowFlags(Qt.Drawer) # dialogs without context help - only close button exists
        # self.gotolineSignal.connect(self.editorWindow.goToLine)
        self.projectPath=""
        self.setupUi(self)
        self.steppableDirRegex=re.compile('steppables')
        self.pluginDirRegex=re.compile('plugins')
  
    @pyqtSignature("") # signature of the signal emited by the button
    def on_okPB_clicked(self):        
        errorFlag=False
        moduleDir=str(self.moduleDirLE.text())
        moduleDir.strip()
        
        if str(self.moduleCoreNameLE.text()).strip()=="":
            QMessageBox.warning(self,"Empty Core Module Name","Please specify C++ core module name")            
            errorFlag=True
        if str(self.moduleDirLE.text()).strip()=="":
            QMessageBox.warning(self,"Empty Module Directory Name","Please specify root directory where subdirectory with module files will be stored")            
            errorFlag=True            
            
        # performing rudimentary check to make sure that steppable are written into steppables directory and plugins into plugins directory
        if self.steppableRB.isChecked():
            steppableDirFound=re.search(self.steppableDirRegex,moduleDir)
            if not steppableDirFound:
                ret=QMessageBox.warning(self,"Possible Directory Name Mismatch","Are you sure you want to create steppable in <br> %s ?" %moduleDir,QMessageBox.No|QMessageBox.Yes)            
                if ret==QMessageBox.No:
                    errorFlag=True            
                    
        if self.pluginRB.isChecked():
            pluginDirFound=re.search(self.pluginDirRegex,moduleDir)
            if not pluginDirFound:
                ret=QMessageBox.warning(self,"Possible Directory Name Mismatch","Are you sure you want to create plugin in <br> %s ?" %moduleDir,QMessageBox.No|QMessageBox.Yes)            
                if ret==QMessageBox.No:
                    errorFlag=True            
                
        if not errorFlag:
            self.accept()
        
    @pyqtSignature("") # signature of the signal emited by the button
    def on_moduleDirPB_clicked(self):        
        recentDir=self.moduleDirLE.text()
        dirName = QFileDialog.getExistingDirectory(self,"Module root directory - subdirectory named after module core name will be created",recentDir)
        dirName=str(dirName)
        dirName.rstrip()
        
        if dirName!='':
            dirName=os.path.abspath(dirName) # normalizing path
            self.moduleDirLE.setText(dirName)