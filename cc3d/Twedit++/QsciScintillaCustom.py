from PyQt4.QtCore import *
from PyQt4.QtGui import *

from PyQt4.Qsci import *

from PyQt4 import QtCore, QtGui

# have to implement custom class for QSciScintilla to handle properly wheel even with and without ctrl pressed 
class QsciScintillaCustom(QsciScintilla):
    def __init__(self,parent=None,_panel=None):
        super(QsciScintillaCustom,self).__init__(parent)
        self.editorWindow=parent
        self.panel=_panel        
        self.mousePressEventOrig=self.mousePressEvent
        
    def wheelEvent(self,event):
        if qApp.keyboardModifiers()==Qt.ControlModifier:
            # Forwarding wheel event to editor windowwheelEvent
            event.ignore()    
             
        else:
            # # calling wheelEvent from base class - regular scrolling
            super(QsciScintillaCustom,self).wheelEvent(event)
    def focusInEvent(self,event):
        editorTab=0
        if self.panel==self.editorWindow.panels[1]:
            editorTab=1    
        # print "FOCUS IN EVENT tabEidget=", editorTab," editor=",self
        
        self.editorWindow.activeTabWidget=self.panel
        
        self.editorWindow.handleNewFocusEditor(self)
        
        super(self.__class__,self).focusInEvent(event)
    # def mousePressEvent(self,event):
        # print "parent=",self.editorWindow
        # print "self.editorWindow.editTab.hasFocus()=",self.editorWindow.editTab.hasFocus()
        # print "self.editorWindow.editTabExtra.hasFocus()=",self.editorWindow.editTabExtra.hasFocus()
        
        # for i in range(self.editorWindow.editTab.count()):
            # if self==self.editorWindow.editTab.widget(i):
                # self.tabWidget=self.editorWindow.editTab
                # break
        # for i in range(self.editorWindow.editTabExtra.count()):
            # if self==self.editorWindow.editTabExtra.widget(i):
                # self.tabWidget=self.editorWindow.editTabExtra
                # tmpEditTab=self.editorWindow.editTab
                # self.editorWindow.editTab=self.editorWindow.editTabExtra
                # self.editorWindow.editTabExtra=tmpEditTab
                # print "AFTER SWAP self.editorWindow.editTab=",self.editorWindow.editTab
                # break
                
        # print "self.tabWidget=",self.tabWidget
        
        # # swap self.editorWindow.editTab with elf.editorWindow.editTabExtra
        