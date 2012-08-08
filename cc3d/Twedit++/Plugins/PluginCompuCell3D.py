

"""
Module used to link Twedit++ with CompuCell3D.
"""

from PyQt4.QtCore import QObject, SIGNAL, QString
from PyQt4.QtGui import QMessageBox

from PyQt4 import QtCore, QtGui

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtNetwork import *

# import PluginAbout
# import PluginCompuCell3D_rc
# import PluginCompuCell3D_rc
import CompuCell3D.PluginCompuCell3D_rc
import CompuCell3D.CC3DListener

# Start-Of-Header
name = "CC3DApp Plugin"
author = "Maciej Swat"
autoactivate = True
deactivateable = False
version = "0.9.0"
className = "CC3DApp"
packageName = "__core__"
shortDescription = "PLugin linking Twedit++ with CompuCell3D"
longDescription = """This plugin provides functionality to link Twedit with CompuCell3D"""
# End-Of-Header

error = QString("")

class CC3DApp(QObject):
    """
    Class implementing the About plugin.
    """
    def __init__(self, ui):
        """
        Constructor
        
        @param ui reference to the user interface object (UI.UserInterface)
        """
        QObject.__init__(self, ui)
        self.__ui = ui

        self.listener=CompuCell3D.CC3DListener.CC3DListener(self.__ui)
        self.listener.setPluginObject(self)
        
        self.__initActions()        
        print "CC3D CONSTRUCTOR"
        
        
        
        
    def activate(self):
        """
        Public method to activate this plugin.
        
        @return tuple of None and activation status (boolean)
        """
        # print "CC3D PLUGIN ACTIVATE"
        # self.__initActions()
        # print "CC3D INIT ACTIONS"
        # self.__initMenu()
        
        return None, True

    def deactivate(self):
        """
        Public method to deactivate this plugin.
        """
        print "DEACTIVATE CC3D PLUGIN"
        self.listener.deactivate()
        # menu = self.__ui.getMenu("help")
        # if menu:
            # menu.removeAction(self.aboutAct)
            # menu.removeAction(self.aboutQtAct)
            # if self.aboutKdeAct is not None:
                # menu.removeAction(self.aboutKdeAct)
        # acts = [self.aboutAct, self.aboutQtAct]
        # if self.aboutKdeAct is not None:
            # acts.append(self.aboutKdeAct)
        # self.__ui.removeE4Actions(acts, 'ui')
    def __initToolbar(self):
        if not self.__ui.toolBar.has_key("CompuCell3D"):
            self.__ui.toolBar["CompuCell3D"] = self.__ui.addToolBar("CompuCell3D")
            self.__ui.insertToolBar(self.__ui.toolBar["File"],self.__ui.toolBar["CompuCell3D"])
            
    def __initActions(self):
        """
        Private method to initialize the actions.
        """
        # print "BEFORE IMPORTS"
        import CompuCell3D
        # import CompuCell3D.PluginCompuCell3D_rc
        # from CompuCell3D import CompuCell3D.PluginCompuCell3D_rc
        
        acts = []
        print "__initActions"
        # QtGui.QIcon('CompuCell3D/play.png')
        self.startCC3DAct=QtGui.QAction(QtGui.QIcon(':/CompuCell3D/cc3d_64x64_logo.png'),"Start CC3D", self, shortcut="", statusTip="Start CompuCell3D ", triggered=self.__startCC3D)
        
            
        self.__initToolbar()    
        
        self.__ui.toolBar["CompuCell3D"].addAction(self.startCC3DAct)
        
        
    def enableStartCC3DAction(self,_flag):
        self.startCC3DAct.setEnabled(_flag)
        
    def startCC3D(self,_simulationName=""):
        # start server and start cc3d passing port to the CC3D command line using  --port=.... option
        # This is a bit tricky here - if i start server before starting cc3d then the port is bound even after twedit closes
        # starting server after starting CC3D seems to work ... Have to find better solution though


        if not self.listener.isListening():        
            self.listener.startServer()    
        
        if not self.startCC3DAct.isEnabled():
            if self.listener.socket and self.listener.socket.state() == QAbstractSocket.ConnectedState:
                # self.listener.socket.flush()
                self.listener.socket.sendNewSimulation(_simulationName)
                # self.listener.socket.flush()
                
        else:            
            self.listener.startCC3D(_simulationName)
            self.enableStartCC3DAction(False)
        # self.listener.startCC3D(_simulationName)
        # self.enableStartCC3DAction(False)
        
        # if not self.listener.isListening():
            # self.listener.getOpenPort()
            
        # self.listener.startCC3D()
        
        # if not self.listener.isListening():                    
            # self.listener.startServer()
        
        # self.enableStartCC3DAction(False)
        
        print "starting CC3D"
    
    def __startCC3D(self):
        self.startCC3D()
        
    def __play(self):
        print "THIS IS PLAY ACTION"
    def __initMenu(self):
        """
        Private method to add the actions to the right menu.
        """
        return
        menu = self.__ui.getMenu("help")
        if menu:
            act = self.__ui.getMenuAction("help", "show_versions")
            if act:
                menu.insertAction(act, self.aboutAct)
                menu.insertAction(act, self.aboutQtAct)
                if self.aboutKdeAct is not None:
                    menu.insertAction(act, self.aboutKdeAct)
            else:
                menu.addAction(self.aboutAct)
                menu.addAction(self.aboutQtAct)
                if self.aboutKdeAct is not None:
                    menu.addAction(self.aboutKdeAct)
    
