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
        self.defaultConfigs["RecentProject"]=QString("")
        self.defaultConfigs["RecentNewProjectDir"]=QString("")
        self.defaultConfigs["RecentProjects"]=QStringList()
        self.defaultConfigs["RecentProjectDirectories"]=QStringList()        
        self.defaultConfigs["ShowCC3DProjectPanel"]=True
        
        
        self.modifiedKeyboardShortcuts={} # dictionary actionName->shortcut for modified keyboard shortcuts - only reassinged shortcuts are stored
        

        self.initSyncSettings()
        # self.updatedConfigs={}
        

        

        
    # def configuration(self,_key):
        # return self.configs[_key]

    def setting(self,_key):
        if _key in ["ShowCC3DProjectPanel"]: # Boolean values
            val = self.settings.value(_key)
            if val.isValid():
                return val.toBool()
            else:
                return self.defaultConfigs[_key]
                
        elif _key in ["RecentProject","RecentNewProjectDir"]:
            val = self.settings.value(_key)
            if val.isValid():
                return val.toString()
            else:
                return self.defaultConfigs[_key]
                
        elif _key in ["RecentProjects","RecentProjectDirectories"]: # QStringList values
            val = self.settings.value(_key)
            if val.isValid():
                return val.toStringList() 
            else:
                return self.defaultConfigs[_key]

        # elif _key in ["TabSpaces","ZoomRange","ZoomRangeFindDisplayWidget","AutocompletionThreshold","FRSyntaxIndex","FROpacity","CurrentTabIndex"]: # integer values
            # val = self.settings.value(_key)
            # if val.isValid():
                # return val.toInt()[0] # toInt returns tuple and first element of if is the integer the second one is flag
            # else:
                # return self.defaultConfigs[_key]
                
        # elif _key in ["InitialSize"]: # QSize values
            # val = self.settings.value(_key)
            # if val.isValid():
                # return val.toSize() 
            # else:
                # return self.defaultConfigs[_key]                             

        # elif _key in ["InitialPosition"]: # QPoint values
            # val = self.settings.value(_key)
            # if val.isValid():
                # return val.toPoint() 
            # else:
                # return self.defaultConfigs[_key]
                

                
    def setSetting(self,_key,_value):
        if _key in ["ShowCC3DProjectPanel"]: # Boolean values
            self.settings.setValue(_key,QVariant(_value))
            
        # elif _key in ["TabSpaces","ZoomRange","ZoomRangeFindDisplayWidget","AutocompletionThreshold","FRSyntaxIndex","FROpacity","CurrentTabIndex"]: # integer values
            # self.settings.setValue(_key,_value)

            
        if _key in ["RecentProject","RecentNewProjectDir","RecentProjects","RecentProjectDirectories"]: # string values
            self.settings.setValue(_key,QVariant(_value))
            
        # elif _key in ["InitialSize","InitialPosition","ListOfOpenFiles","FRSyntax","FRFindHistory","FRReplaceHistory","FRFiltersHistory","FRDirectoryHistory","KeyboardShortcuts"]: # QSize, QPoint,QStringList , QString values
            # self.settings.setValue(_key,QVariant(_value))
        # else:
            # dbgMsg("Wrong format of configuration option:",_key,":",_value)
            
    def setKeyboardShortcut(self,_actionName,_keyboardshortcut):
        self.modifiedKeyboardShortcuts[_actionName]=_keyboardshortcut    
        
    def keyboardShortcuts(self):
        return self.modifiedKeyboardShortcuts
    
    def prepareKeyboardShortcutsForStorage(self):
        self.modifiedKeyboardShortcutsStringList=QStringList()
        for actionName in self.modifiedKeyboardShortcuts.keys():
            self.modifiedKeyboardShortcutsStringList.append(actionName)
            self.modifiedKeyboardShortcutsStringList.append(self.modifiedKeyboardShortcuts[actionName])
            
        self.setSetting("KeyboardShortcuts",self.modifiedKeyboardShortcutsStringList)    
            
        
    
    def initSyncSettings(self):
        for key in self.defaultConfigs.keys():
            
            val = self.settings.value(key)
            if not val.isValid():
                self.setSetting(key,self.defaultConfigs[key])
                
        # # initialize self.modifiedKeyboardShortcuts
        # self.modifiedKeyboardShortcutsStringList=self.setting("KeyboardShortcuts")
        # for i in range(0,self.modifiedKeyboardShortcutsStringList.count(),2): 
            # self.modifiedKeyboardShortcuts[str(self.modifiedKeyboardShortcutsStringList[i])]=str(self.modifiedKeyboardShortcutsStringList[i+1])        
            
        
            