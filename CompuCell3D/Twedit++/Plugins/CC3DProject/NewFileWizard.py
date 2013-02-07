import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.QtCore as QtCore
import ui_newfilewizard
import sys
import os.path

MAC = "qt_mac_set_native_menubar" in dir()

class NewFileWizard(QWizard,ui_newfilewizard.Ui_NewFileWizard):
    #signals
    # gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,parent=None):
        super(NewFileWizard, self).__init__(parent)
        self.cc3dProjectTreeWidget=parent

        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows 
        if sys.platform.startswith('win'): 
            self.setWindowFlags(Qt.Drawer) # dialogs without context help - only close button exists
        # self.gotolineSignal.connect(self.editorWindow.goToLine)
        self.projectPath=""
        self.setupUi(self)
  
        
        # if not MAC:
            # self.cancelButton.setFocusPolicy(Qt.NoFocus)
        self.updateUi()

    # @pyqtSignature("") # signature of the signal emited by the button
    # def on_okButton_clicked(self):
        # self.findChangedConfigs()        
        # self.close()
           
    @pyqtSignature("") # signature of the signal emited by the button
    def on_nameBrowsePB_clicked(self):
        fileName=str(QFileDialog.getOpenFileName(self,"Save File",self.projectPath,"*"))        
        fileName=os.path.abspath(fileName) # normalizing path
        self.nameLE.setText(fileName)
        
    @pyqtSignature("") # signature of the signal emited by the button
    def on_locationBrowsePB_clicked(self):
        dirName=directory=QFileDialog.getExistingDirectory ( self, "Directory (within current project) for the new file...",self.projectPath)
        dirName=str(dirName)
        dirName=os.path.abspath(dirName) # normalizing path
        print "self.projectPath=",self.projectPath
        print "dirName=",dirName
        relativePath=self.findRelativePath(self.projectPath,dirName)
        
        if dirName==relativePath:
            QMessageBox.warning(self,"Directory outside the project","You are trying to create new file outside project directory.<br> This is not allowed",QMessageBox.Ok )
            relativePath="Simulation/"
        self.locationLE.setText(relativePath)
        
    def findRelativePathSegments(self,basePath,p, rest=[]):
    
        """
            This function finds relative path segments of path p with respect to base path    
            It returns list of relative path segments and flag whether operation succeeded or not    
        """
        
        h,t = os.path.split(p)
        pathMatch=False
        if h==basePath:
            pathMatch=True
            return [t]+rest,pathMatch
        print "(h,t,pathMatch)=",(h,t,pathMatch)
        if len(h) < 1: return [t]+rest,pathMatch
        if len(t) < 1: return [h]+rest,pathMatch
        return self.findRelativePathSegments(basePath,h,[t]+rest)
        
    def findRelativePath(self,basePath,p):
        relativePathSegments,pathMatch=self.findRelativePathSegments(basePath,p)
        if pathMatch:
            relativePath=""
            for i in range(len(relativePathSegments)):
                segment=relativePathSegments[i]
                relativePath+=segment
                if i !=len(relativePathSegments)-1:
                    relativePath+="/" # we use unix style separators - they work on all (3) platforms
            return relativePath
        else:
            return p    
    #initialize wizard page
    def updateUi(self):
        self.locationLE.setText("Simulation/") # default storage of simulation files
        
        tw=self.cc3dProjectTreeWidget
        projItem=tw.getProjectParent(tw.currentItem())
        
        pdh=None
        try:
            pdh=tw.plugin.projectDataHandlers[projItem]        
            
        except LookupError,e:
            # could not find simulation data handler for this item
            return        
            
        self.projectPath = pdh.cc3dSimulationData.basePath
        
        self.projectDirLE.setText(pdh.cc3dSimulationData.basePath)
        # construct a list of available file types
        if pdh.cc3dSimulationData.xmlScript=="":
            self.fileTypeCB.insertItem(0,"XML Script")
        if pdh.cc3dSimulationData.pythonScript=="":
            self.fileTypeCB.insertItem(0,"Main Python Script")
            
        return
        