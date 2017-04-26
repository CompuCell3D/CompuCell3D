import re
# from PyQt4.QtCore import *
# from PyQt4.QtGui import *
# import PyQt4.QtCore as QtCore
from utils.global_imports import *
import ui_gotolinedlg
import sys
from Messaging import stdMsg, dbgMsg, errMsg, dbgMsg
MAC = "qt_mac_set_native_menubar" in dir()



class GoToLineDlg(QDialog,ui_gotolinedlg.Ui_GoToLineDlg):
    #signals
    gotolineSignal = pyqtSignal(int)
    
    def __init__(self,_currentEditor=None,parent=None):
        super(GoToLineDlg, self).__init__(parent)
        self.editorWindow=parent
        self.currentEditor=_currentEditor
        self.gotolineSignal.connect(self.editorWindow.goToLine)
        self.setupUi(self)
        #ensuring that only integers greater than 0 can be entered
        self.intValidator=QIntValidator(self.goToLineEdit)
        self.intValidator.setBottom(1)
        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows 
        if sys.platform.startswith('win'): 
            self.setWindowFlags(Qt.Drawer) # dialogs without context help - only close button exists
        # print dir(self.currentEditor)
        self.intValidator.setTop(self.currentEditor.lines())
        self.goToLineEdit.setValidator(self.intValidator)
        
        if not MAC:
            self.closeButton.setFocusPolicy(Qt.NoFocus)
        self.updateUi()

    #def showEvent(self, event):

        ## this is quite strange but on windows there is no need to posigion dialog box, but on Linux, you have to do it manually
        #self.move(0,0) #always position popup at (0,0) then calculate required shift
        #self.adjustSize()
            
            
        ##setting position of the find dialog widget
        #geom=self.editorWindow.geometry()            
        #pGeom=self.geometry()            
        
        #pCentered_x=geom.x()+(geom.width()-pGeom.width())/2
        #pCentered_y=geom.y()+(geom.height()-pGeom.height())
        
        #self.move(pCentered_x,pCentered_y)
            
            ##pGeom=self.cycleTabsPopup.geometry()
        ###self.setFocus(True)
        ##self.activateWindow()
        #QDialog.showEvent(self,event) # necesary to move dialog to the correct position - we do it only if we override show event

    @pyqtSlot() # signature of the signal emited by the button
    def on_goButton_clicked(self):
        # print "this is on go button clicked=",self.goToLineEdit.text()
        line_num = int(self.goToLineEdit.text())
        if line_num:
            self.gotolineSignal.emit(line_num)
        # we close the dialog right after user hits Go button  . If the entry is invalid no action is trigerred  
        self.close()    
        return

    def updateUi(self):
        # if self.findComboBox.lineEdit():
            # print "got text to find ", self.findComboBox.lineEdit().text()
        
        pass        

        
