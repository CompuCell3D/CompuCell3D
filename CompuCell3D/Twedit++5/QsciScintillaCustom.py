# from PyQt4.QtCore import *
# from PyQt4.QtGui import *
#
# from PyQt4.Qsci import *
#
# from PyQt4 import QtCore, QtGui
# import sys
from utils.global_imports import *


# have to implement custom class for QSciScintilla to handle properly wheel even with and without ctrl pressed 
class QsciScintillaCustom(QsciScintilla):
    def __init__(self,parent=None,_panel=None):
        super(QsciScintillaCustom,self).__init__(parent)
        self.editorWindow=parent
        self.panel=_panel        
        self.mousePressEventOrig=self.mousePressEvent
        self.CtrlKeyEquivalent=Qt.Key_Control
        self.scintillaDefinedLetterShortcuts=[ord('D'),ord('L'),ord('T'),ord('U'),ord('/'),ord(']')]
        self.customContextMenu=None
        self.linesChanged.connect(self.linesChangedHandler)
        if sys.platform.startswith("darwin"):        
            self.CtrlKeyEquivalent=Qt.Key_Alt
            
        # print 'key code=',ord('d')+(QsciScintilla.SCMOD_CTRL<<16)
        
    def wheelEvent(self,event):
        if qApp.keyboardModifiers()==Qt.ControlModifier:
            # Forwarding wheel event to editor windowwheelEvent
            event.ignore()                 
        else:
            # # calling wheelEvent from base class - regular scrolling
            super(QsciScintillaCustom,self).wheelEvent(event)
            
    def handleScintillaDefaultShortcut(self,modifierKeysText,event):    
    
        if event.key() in self.scintillaDefinedLetterShortcuts:            
            try:
                import ActionManager as am
                action=am.actionDict[am.shortcutToActionDict[modifierKeysText+'+'+chr(event.key())]]
                action.trigger()
                event.accept()
            except LookupError:
                super(QsciScintillaCustom,self).keyPressEvent(event)
        else:
            super(QsciScintillaCustom,self).keyPressEvent(event)
    
    def registerCustomContextMenu(self,_menu):
        self.customContextMenu=_menu
        
    def unregisterCustomContextMenu(self):
        self.customContextMenu=None
        
    def contextMenuEvent(self,_event):
        if not self.customContextMenu:
            super(QsciScintillaCustom,self).contextMenuEvent(_event)
        else:
            self.customContextMenu.exec_(_event.globalPos())
        
        
    
    def keyPressEvent(self, event):   
        """
            senses if scintilla predefined keyboard shortcut was pressed.  
        """
        
        if event.modifiers() == Qt.ControlModifier: 
            
            self.handleScintillaDefaultShortcut('Ctrl',event)
        elif event.modifiers() & Qt.ControlModifier and event.modifiers() & Qt.ShiftModifier:

            self.handleScintillaDefaultShortcut('Ctrl+Shift',event)
            
        else:
            super(QsciScintillaCustom,self).keyPressEvent(event)
            
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
        
    def linesChangedHandler(self):
        ''' adjusting width of the line number margin
        '''
        # print '__linesChangedHandler self.marginWidth(0)=',self.marginWidth(0)
        if self.marginLineNumbers(0):
            # self.setMarginLineNumbers(0, _flag)
            numberOfLines=self.lines()
            
            from math import log        
            
            numberOfDigits= int(log(numberOfLines,10))+2 if numberOfLines>0 else 2
            self.setMarginWidth(0,'0'*numberOfDigits)
                    
    
        