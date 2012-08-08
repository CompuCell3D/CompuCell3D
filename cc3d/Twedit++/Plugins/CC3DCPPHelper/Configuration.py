from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.Qsci import *

from PyQt4 import QtCore, QtGui
from Messaging import stdMsg, dbgMsg, errMsg, setDebugging


class Configuration:
    def __init__(self,_settings):
            
        self.settings=_settings
        #default settings
        self.defaultConfigs={}

        self.defaultConfigs["RecentModuleDirectory"]=QString("")
                
        self.modifiedKeyboardShortcuts={} # dictionary actionName->shortcut for modified keyboard shortcuts - only reassinged shortcuts are stored
        
        self.initSyncSettings()

    def setting(self,_key):
        if _key in ["RecentModuleDirectory"]:
            val = self.settings.value(_key)
            if val.isValid():
                return val.toString()
            else:
                return self.defaultConfigs[_key]
            

                
    def setSetting(self,_key,_value):

            
        if _key in ["RecentModuleDirectory"]: # string values
            self.settings.setValue(_key,QVariant(_value))
            
            
    # def setKeyboardShortcut(self,_actionName,_keyboardshortcut):
        # self.modifiedKeyboardShortcuts[_actionName]=_keyboardshortcut    
        
    # def keyboardShortcuts(self):
        # return self.modifiedKeyboardShortcuts
    
    # def prepareKeyboardShortcutsForStorage(self):
        # self.modifiedKeyboardShortcutsStringList=QStringList()
        # for actionName in self.modifiedKeyboardShortcuts.keys():
            # self.modifiedKeyboardShortcutsStringList.append(actionName)
            # self.modifiedKeyboardShortcutsStringList.append(self.modifiedKeyboardShortcuts[actionName])
            
        # self.setSetting("KeyboardShortcuts",self.modifiedKeyboardShortcutsStringList)    
            
        
    
    def initSyncSettings(self):
        for key in self.defaultConfigs.keys():
            
            val = self.settings.value(key)
            if not val.isValid():
                self.setSetting(key,self.defaultConfigs[key])
                
            
        
            