import re
# from PyQt4.QtCore import *
# from PyQt4.QtGui import *
# import PyQt4.QtCore as QtCore
from utils.global_imports import *

import ui_KeyboardShortcuts
import sys
import ActionManager as am
from KeyShortcut import KeyShortcutDlg
from Messaging import stdMsg, dbgMsg, errMsg, dbgMsg
MAC = "qt_mac_set_native_menubar" in dir()
import Configuration

"""
Have to check translation to native formats for KeySequence string opertations
"""

class KeyboardShortcutsDlg(QDialog,ui_KeyboardShortcuts.Ui_KeyboardShortcutsDlg):
    #signals
    gotolineSignal = QtCore.pyqtSignal( ('int',))
    
    def __init__(self,_currentEditor=None,parent=None):
        super(KeyboardShortcutsDlg, self).__init__(parent)
        self.editorWindow=parent
        self.currentEditor=_currentEditor        
        # there are issues with Drawer dialog not getting focus when being displayed on linux
        # they are also not positioned properly so, we use "regular" windows 
        if sys.platform.startswith('win'): 
            self.setWindowFlags(Qt.Drawer) # dialogs without context help - only close button exists        
        self.setupUi(self)
        # self.connect(self.shortcutTable,SIGNAL("cellClicked (int,int)"),self.shortcutCellClicked)
        self.shortcutTable.cellClicked.connect(self.shortcutCellClicked)
        self.lastClickPosition=None
        self.changesInActionShortcutList=[] # action name -> shortcut sequence for newly defined shortcuts
                
        self.shortcutItemDict={} # shortcut shortcut -> QTableItem
        
    #making sure that columns fill entire qtable widget    
    def resizeEvent(self,e):
        shortcutTableSize=self.shortcutTable.size()
        self.shortcutTable.setColumnWidth(0,shortcutTableSize.width()/2)
        self.shortcutTable.setColumnWidth(1,shortcutTableSize.width()/2)
        e.accept()

    def initializeShortcutTables(self):
        # delete all rows first
        # dbgMsg("self.shortcutTable.rowCount()=",self.shortcutTable.rowCount())
        self.changesInActionShortcutList=[]
        for i in range(self.shortcutTable.rowCount()-1,-1,-1):
            self.shortcutTable.removeRow(i)
            # dbgMsg(" i=",i)
        rowIdx=0
        actionsSorted=am.actionToShortcutDict.keys()
        actionsSorted.sort()
        
        #empty QTableWidgetItem used to extract/prepare item format
        
        item=QTableWidgetItem()
        flags=item.flags()
        flags &= ~flags
        font=item.font()
        font.setBold(True)
        
        foregroundBrush=item.foreground()
        
        for action in actionsSorted:
            shortcut=am.actionToShortcutDict[action]
            self.shortcutTable.insertRow(rowIdx)
            actionItem=QTableWidgetItem(action)
            flags=actionItem.flags()
            flags &= ~flags
            actionItem.setFlags(flags)
            actionItem.setFont(font)
            actionItem.setForeground(foregroundBrush)
            
            
            self.shortcutTable.setItem(rowIdx,0,actionItem)
            shortcutItem=QTableWidgetItem(shortcut)
            shortcutItem.setFlags(flags)
            shortcutItem.setFont(font)
            shortcutItem.setForeground(foregroundBrush)
            self.shortcutTable.setItem(rowIdx,1,shortcutItem)
            
            # self.shortcutItemDict[shortcutItem]=shortcut
            
            rowIdx+=1
            
    def assignNewShortcut(self,_newKeySequence,_actionItem,_shortcutItem):
        # this is simple linear operation - can use dictionaries to speed it up but for now we will use simple solution
        keySequenceText=str(_newKeySequence.toString())
        actionText=str(_actionItem.text())
        for i in range(self.shortcutTable.rowCount()):
            shortcutItem=self.shortcutTable.item(i,1)        
            
            if str(shortcutItem.text())=='':
                continue # do not look for action with empty shortcut name
                
            if str(shortcutItem.text())==keySequenceText:
                actionItemLocal=self.shortcutTable.item(i,0)
                
                if str(actionText)!=str(actionItemLocal.text()): #do nothing if changed shortcut for the action is same as old shortcut
                    
                    shortcutItemLocal=self.shortcutTable.item(i,1)
                    shortcutItemLocal.setText('')
                    self.changesInActionShortcutList.append(str(actionItemLocal.text()))
                    self.changesInActionShortcutList.append(QKeySequence(''))
                    
                    self.currentEditor.configuration.setKeyboardShortcut(str(actionItemLocal.text()),'')                
                    
                    break
                    
        _shortcutItem.setText(_newKeySequence.toString())       
        
        self.changesInActionShortcutList.append(actionText)
        self.changesInActionShortcutList.append(_newKeySequence)
                    
        self.currentEditor.configuration.setKeyboardShortcut(actionText,keySequenceText)                
                
    
    def shortcutCellClicked(self,_row,_column):
        
        if _column==1:
            #display grab shortcut widget
            shortcutItem=self.shortcutTable.item(_row,1)
            actionItem=self.shortcutTable.item(_row,0)
            
            shortcutText=shortcutItem.text()
            actionText=actionItem.text()
            keyShortcutDlg=KeyShortcutDlg(self,str(actionText),str(shortcutText))
            
            
            
            ret=keyShortcutDlg.exec_()
            if ret:
                newKeySequence=keyShortcutDlg.getKeySequence()
                dbgMsg("THIS IS NEW SHORTCUT:",str(newKeySequence.toString())) 
                # dbgMsg("THIS IS NEW SHORTCUT:",str(newKeySequence.toString(QKeySequence.NativeText))) # QKeySequence.NativeText does not work well on OSX
                self.assignNewShortcut(newKeySequence,actionItem,shortcutItem)
                # self.changesInActionShortcutDict[actionText]=newKeySequence
                # shortcutItem.setText(newKeySequence.toString())
                
    def reassignNewShortcuts(self):
        for changeIdx in range(0,len(self.changesInActionShortcutList),2):
            dbgMsg("actionText=",self.changesInActionShortcutList[changeIdx])
            dbgMsg("sequence=",str(self.changesInActionShortcutList[changeIdx+1].toString()))
            
            am.setActionKeyboardShortcut(self.changesInActionShortcutList[changeIdx],self.changesInActionShortcutList[changeIdx+1])
    
        # for actionText,keySequence in self.changesInActionShortcutDict.iteritems():
            # am.setActionKeyboardShortcut(actionText,keySequence)
            