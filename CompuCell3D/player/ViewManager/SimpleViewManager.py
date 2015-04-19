import os
import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *

#from PyQt4.QtXml import *
import Configuration
import DefaultData
gip = DefaultData.getIconPath

MODULENAME = '------- SimpleViewManager: '

# ViewManager inherits from QObject to use its methods (e.g. self.connect)

# class SimpleViewManager(QObject):
class SimpleViewManager():
    def __init__(self, ui):
        # QObject.__init__(self)
        self.visual = {}
        self.visual["CellsOn"]    = Configuration.getSetting("CellsOn")
        self.visual["CellBordersOn"]  = Configuration.getSetting("CellBordersOn")
        self.visual["ClusterBordersOn"]  = Configuration.getSetting("ClusterBordersOn")
        self.visual["CellGlyphsOn"]  = Configuration.getSetting("CellGlyphsOn")
        self.visual["FPPLinksOn"]  = Configuration.getSetting("FPPLinksOn")
#        self.visual["FPPLinksColorOn"]  = Configuration.getSetting("FPPLinksColorOn")
        self.visual["CC3DOutputOn"]  = Configuration.getSetting("CC3DOutputOn")
#        print MODULENAME, 'self.visual["CC3DOutputOn"] = ',self.visual["CC3DOutputOn"]

        # self.visual["ContoursOn"]   = Configuration.getSetting("ContoursOn")
        self.visual["ConcentrationLimitsOn"]   = Configuration.getSetting("ConcentrationLimitsOn")
        self.visual["ZoomFactor"]   = Configuration.getSetting("ZoomFactor")
        self.initActions()
        self.ui = ui
   
    def initFileMenu(self):
        menu = QMenu(QApplication.translate('ViewManager', '&File'), self.ui)
        menu.addAction(self.openAct)
        menu.addAction(self.saveAct)
        menu.addAction(self.openScreenshotDescriptionAct)
        menu.addAction(self.saveScreenshotDescriptionAct)
#        menu.addAction(self.openPlayerParamsAct)
        menu.addAction(self.savePlayerParamsAct)
        menu.addAction(self.openLDSAct) # LDS lattice description summary  - xml file that specifies what simulation data has been written to the disk
        menu.addSeparator()
        menu.addAction(self.tweditAct)
        # menu.addAction(self.closeAct)
        menu.addSeparator()
        recentSimulationsMenu=menu.addMenu("Recent Simulations...")
        menu.addSeparator()
        menu.addAction(self.exitAct)
        
        return (menu,recentSimulationsMenu)
    
    def initSimMenu(self):
        menu = QMenu(QApplication.translate('ViewManager', '&Simulation'), self.ui)
        menu.addAction(self.runAct)
        menu.addAction(self.stepAct)
        menu.addAction(self.pauseAct)
        menu.addAction(self.stopAct)
        
        menu.addAction(self.addVTKWindowAct)        
        
        menu.addSeparator()
        #--------------------
        menu.addAction(self.serializeAct)
        
        return menu

    def initVisualMenu(self):
        menu = QMenu(QApplication.translate('ViewManager', '&Visualization'), self.ui)
        menu.addAction(self.cellsAct)
        menu.addAction(self.borderAct)
        menu.addAction(self.clusterBorderAct)
        menu.addAction(self.cellGlyphsAct)
        menu.addAction(self.FPPLinksAct)
#        menu.addAction(self.FPPLinksColorAct)
        #menu.addAction(self.plotTypeAct)     
        # menu.addAction(self.contourAct)
        menu.addAction(self.limitsAct)
        menu.addSeparator()
        menu.addAction(self.cc3dOutputOnAct)
        # menu.addAction(self.configAct)
        
        return menu

    def initToolsMenu(self):
        menu = QMenu(QApplication.translate('ViewManager', '&Tools'), self.ui)
        # Don't remove. Will be implemented later
        #menu.addAction(self.pifGenAct)
        #menu.addAction(self.pifVisAct)     
        # # # menu.addAction(self.movieAct)
        menu.addSeparator()
        menu.addAction(self.configAct)
        # menu.addAction(self.pifFromVTKAct)
        # self.pifFromVTKAct.setEnabled(False)

        menu.addAction(self.pifFromSimulationAct)
        self.pifFromSimulationAct.setEnabled(False)
        
        return menu

    def initWindowMenu(self):
        menu = QMenu(QApplication.translate('ViewManager', '&Window'), self.ui)
        
        # NOTE initialization of the menu is done in the updateWindowMenu function in SimpleTabView
        
        
        # Don't remove. Will be implemented later
        #menu.addAction(self.pifGenAct)
        #menu.addAction(self.pifVisAct)     
        # # # menu.addAction(self.movieAct)
        
        # menu.addAction(self.newGraphicsWindowAct)
        # menu.addAction(self.tileAct)
        # menu.addAction(self.cascadeAct)
        # menu.addSeparator()
        # menu.addAction(self.closeActiveWindowAct)
        # menu.addAction(self.closeAdditionalGraphicsWindowsAct)
        # menu.addSeparator()
        
        
        return menu
        
    def initHelpMenu(self):
        menu = QMenu(QApplication.translate('ViewManager', '&Help'), self.ui)
        menu.addAction(self.quickAct)
        menu.addAction(self.tutorAct)     
        menu.addAction(self.refManAct)
        menu.addSeparator()
        menu.addAction(self.aboutAct)
        menu.addSeparator()
        menu.addAction(self.whatsThisAct)
        
        return menu

    def initFileToolbar(self):
        tb = QToolBar(QApplication.translate('ViewManager', 'File'), self.ui)
        tb.setIconSize(QSize(20, 18)) # UI.Config.ToolBarIconSize
        tb.setObjectName("FileToolbar")
        tb.setToolTip(QApplication.translate('ViewManager', 'File'))
        
        tb.addAction(self.openAct)        
        tb.addAction(self.saveAct)
        # tb.addAction(self.closeAct)
        tb.addAction(self.configAct)
        tb.addAction(self.tweditAct)
        return tb
       
       
    def initSimToolbar(self):
        tb = QToolBar(QApplication.translate('ViewManager', 'Simulation'), self.ui)
        tb.setIconSize(QSize(20, 18))#UI.Config.ToolBarIconSize
        tb.setObjectName("SimToolbar")
        tb.setToolTip(QApplication.translate('ViewManager', 'Simulation'))
        
        tb.addAction(self.runAct)    
        tb.addAction(self.stepAct)    
        tb.addAction(self.pauseAct)
        tb.addAction(self.stopAct)
        
        return tb
    
    def initCrossSectionToolbar(self):
        cstb = QToolBar("CrossSection", self)
        #viewtb.setIconSize(QSize(20, 18))
        cstb.setObjectName("CrossSection")
        cstb.setToolTip("Cross Section")
        
        cstb.addWidget(QLabel("  ")) # Spacer, just make it look pretty 
        cstb.addWidget(self.threeDRB)
        cstb.addWidget(QLabel("  "))
        cstb.addWidget(self.xyRB)
        cstb.addWidget(self.xySB)
        cstb.addWidget(QLabel("  "))
        cstb.addWidget(self.xzRB)
        cstb.addWidget(self.xzSB)
        cstb.addWidget(QLabel("  "))
        cstb.addWidget(self.yzRB)
        cstb.addWidget(self.yzSB)
        cstb.addWidget(QLabel("    "))
        cstb.addWidget(self.fieldComboBox)
        
        return cstb
   
    def initWindowToolbar(self):
        wtb = QToolBar(QApplication.translate('ViewManager', 'Window'), self.ui)
        wtb.setIconSize(QSize(20, 18)) # UI.Config.ToolBarIconSize
        wtb.setObjectName("WindowToolbar")
        wtb.setToolTip(QApplication.translate('ViewManager', 'Window'))
        
        wtb.addAction(self.newGraphicsWindowAct)
        # wtb.addAction(self.newPlotWindowAct)
        return wtb
   
    def initActions(self):        
        # list containing all file actions
        self.fileActions         = []
        #self.viewActions        = []
        #self.toolbarsActions    = []
        self.simActions          = []
        self.crossSectionActions = []
        self.visualActions       = []
        self.toolsActions        = []
        self.helpActions         = []
        self.windowActions       = [] # list containing all window actions
                
        self.__initWindowActions()
        self.__initFileActions()
        #self.__initViewActions()
        self.__initSimActions()
        self.__initCrossSectionActions()
        self.__initVisualActions()
        self.__initToolsActions()
        self.__initHelpActions()   
        # self.__initTabActions()
    
    #def setModelEditor(self, modelEditor):
    #    self.__modelEditor = modelEditor

    def hello(self):
      #if self.modelAct.isChecked():
      #   print "Hello from modelAct"
        print "Welcome to CompuCell3D!"
      #sys.stderr.write(self.out)
        
    #def _getOpenFileFilter(self):
        """
        Protected method to return the active filename filter for a file open dialog.
        
        The appropriate filename filter is determined by file extension of
        the currently active editor.
        
        @return name of the filename filter (QString) or None
        """
        """
        if self.activeWindow() is not None and \
           self.activeWindow().getFileName():
            ext = os.path.splitext(self.activeWindow().getFileName())[1]
            rx = QRegExp(".*\*\.%s[ )].*" % ext[1:])
            filters = QScintilla.Lexers.getOpenFileFiltersList()
            index = filters.indexOf(rx)
            if index == -1:
                return QString(Preferences.getEditor("DefaultOpenFilter"))
            else:
                return filters[index]
        else:
            return QString(Preferences.getEditor("DefaultOpenFilter"))
        """

    def _getOpenStartDir(self):
        """
        Protected method to return the starting directory for a file open dialog. 
        
        The appropriate starting directory is calculated
        using the following search order, until a match is found:<br />
            1: Directory of currently active editor<br />
            2: Directory of currently active Project<br />
            3: CWD
        
        @return name of directory to start (string) or None
        """
        
        # if we have an active source, return its path
        if self.activeWindow() is not None and \
            self.activeWindow().getFileName():
            return os.path.dirname(self.activeWindow().getFileName())
        
        # Check, if there is an active project and return its path
        #elif e4App().getObject("Project").isOpen():
        #    return e4App().getObject("Project").ppath
        
        else:
            # None will cause open dialog to start with cwd
            return QString()
    
    def __initFileActions(self):
        # - Create Action -- act = QAction()
        # - Set status tip -- act.setStatusTip()
        # - Set what's this -- act.setWhatsThis()
        # - Connect signals -- self.connect(act, ...)
        # - Add to the action list - actList.append(act)
        
        self.openAct = QAction(QIcon(gip("fileopen.png")), "&Open Simulation File (.cc3d)", self)
        # self.openAct.setShortcut(QKeySequence(tr("Ctrl+O")))
        self.openAct.setShortcut(Qt.CTRL + Qt.Key_O)
        
        self.saveAct = QAction(QIcon(gip("save.png")), "&Save Simulation XML file", self)
        self.saveScreenshotDescriptionAct=QAction(QIcon(gip("screenshots_save_alt.png")), "&Save Screenshot Description...", self)
        self.openScreenshotDescriptionAct=QAction(QIcon(gip("screenshots_open.png")), "&Open Screenshot Description...", self)
        self.savePlayerParamsAct=QAction(QIcon(gip("screenshots_save_alt.png")), "&Save Player Parameters...", self)
#        self.openPlayerParamsAct=QAction(QIcon(gip("screenshots_open.png")), "&Open Player Parameters...", self)
        self.openLDSAct=QAction(QIcon(gip("screenshots_open.png")), "&Open Lattice Description Summary File...", self)
        
        # self.closeAct = QAction(QIcon("player/icons/close.png"), "&Close Simulation", self)
        self.exitAct = QAction(QIcon(gip("exit2.png")), "&Exit", self)
        
        self.tweditAct=QAction(QIcon(gip("twedit-icon.png")), "Start Twe&dit++", self)
        
        # Why do I need these appendings?
        self.fileActions.append(self.openAct)
        self.fileActions.append(self.saveAct)
        self.fileActions.append(self.openScreenshotDescriptionAct)
        self.fileActions.append(self.saveScreenshotDescriptionAct)
#        self.fileActions.append(self.openPlayerParamsAct)
        self.fileActions.append(self.savePlayerParamsAct)
        self.fileActions.append(self.openLDSAct)
        self.fileActions.append(self.tweditAct)
        
        # self.fileActions.append(self.closeAct)
        self.fileActions.append(self.exitAct)

    def __initCrossSectionActions(self):
        # Do I need actions? Probably not, but will leave for a while
        self.threeDAct = QAction(self)
        self.threeDRB  = QRadioButton("3D")
        self.threeDRB.addAction(self.threeDAct)

        self.xyAct = QAction(self)
        self.xyRB  = QRadioButton("xy")
        self.xyRB.addAction(self.xyAct)

        self.xySBAct = QAction(self)
        self.xySB  = QSpinBox()
        self.xySB.addAction(self.xySBAct)

        self.xzAct = QAction(self)
        self.xzRB  = QRadioButton("xz")
        self.xzRB.addAction(self.xzAct)

        self.xzSBAct = QAction(self)
        self.xzSB  = QSpinBox()
        self.xzSB.addAction(self.xzSBAct)

        self.yzAct = QAction(self)
        self.yzRB  = QRadioButton("yz")
        self.yzRB.addAction(self.yzAct)

        self.yzSBAct = QAction(self)
        self.yzSB  = QSpinBox()
        self.yzSB.addAction(self.yzSBAct)

        self.fieldComboBoxAct = QAction(self)
        self.fieldComboBox  = QComboBox()
        self.fieldComboBox.addAction(self.fieldComboBoxAct)
        self.fieldComboBox.addItem("-- Field Type --")
        #self.fieldComboBox.addItem("cAMP")
        
        # Why append?
        self.crossSectionActions.append(self.threeDAct)
        self.crossSectionActions.append(self.xyAct)
        self.crossSectionActions.append(self.xySBAct)
        self.crossSectionActions.append(self.xzAct)
        self.crossSectionActions.append(self.xzSBAct)
        self.crossSectionActions.append(self.yzAct)
        self.crossSectionActions.append(self.yzSBAct)
        self.crossSectionActions.append(self.fieldComboBoxAct)
    
    def __initSimActions(self):

        gip = DefaultData.getIconPath

        # self.runAct = QAction(QIcon("player/icons/play.png"), "&Run", self)
        self.runAct = QAction(QIcon(gip("play.png")), "&Run", self)
        self.runAct.setShortcut(Qt.CTRL + Qt.Key_M)
        self.stepAct = QAction(QIcon(gip("step.png")), "&Step", self)
        self.stepAct.setShortcut(Qt.CTRL + Qt.Key_E)
        self.pauseAct = QAction(QIcon(gip("pause.png")), "&Pause", self)
        self.pauseAct.setShortcut(Qt.CTRL + Qt.Key_D)
        self.stopAct = QAction(QIcon(gip("stop.png")), "&Stop", self)
        self.stopAct.setShortcut(Qt.CTRL + Qt.Key_X)
        self.serializeAct = QAction( "Serialize", self)
        
        self.addVTKWindowAct=QAction(QIcon(gip("stop.png")),'Add VTK Window',self )
        self.addVTKWindowAct.setShortcut(Qt.CTRL + Qt.Key_I)
        

        # Why append?
        self.simActions.append(self.runAct)
        self.simActions.append(self.stepAct)
        self.simActions.append(self.pauseAct)
        self.simActions.append(self.stopAct)
        self.simActions.append(self.serializeAct)
        self.simActions.append(self.addVTKWindowAct)
        

    def __initVisualActions(self):
        self.cellsAct = QAction("&Cells", self)
        self.cellsAct.setCheckable(True)
        self.cellsAct.setChecked(self.visual["CellsOn"])
        
        self.borderAct = QAction("Cell &Borders", self)
        self.borderAct.setCheckable(True)
        self.borderAct.setChecked(self.visual["CellBordersOn"])
        
        self.clusterBorderAct = QAction("Cluster Borders", self)
        self.clusterBorderAct.setCheckable(True)
        self.clusterBorderAct.setChecked(self.visual["ClusterBordersOn"])
        
        self.cellGlyphsAct = QAction("Cell &Glyphs", self)
        self.cellGlyphsAct.setCheckable(True)
        self.cellGlyphsAct.setChecked(self.visual["CellGlyphsOn"])
        
        self.FPPLinksAct = QAction("&FPP Links", self)  # callbacks for these menu items in child class SimpleTabView
        self.FPPLinksAct.setCheckable(True)
        self.FPPLinksAct.setChecked(self.visual["FPPLinksOn"])
#        self.connect(self.FPPLinksAct, SIGNAL('triggered()'), self.__fppLinksTrigger)
        
#        self.FPPLinksColorAct = QAction("&FPP Links(color)", self)
#        self.FPPLinksColorAct.setCheckable(True)
#        self.FPPLinksColorAct.setChecked(self.visual["FPPLinksColorOn"])
#        self.connect(self.FPPLinksColorAct, SIGNAL('triggered()'), self.__fppLinksColorTrigger)
        
        # Not implemented in version 3.4.0
        self.plotTypeAct = QAction("&Simulation Plot", self)
        self.plotTypeAct.setCheckable(True)
        
        # self.contourAct = QAction("&Concentration Contours", self)
        # self.contourAct.setCheckable(True)
        # self.contourAct.setChecked(self.visual["ContoursOn"])
        
        self.limitsAct = QAction("Concentration &Limits", self)
        self.limitsAct.setCheckable(True)
        self.limitsAct.setChecked(self.visual["ConcentrationLimitsOn"])

        self.cc3dOutputOnAct = QAction("&Turn On CompuCell3D Output", self)
        self.cc3dOutputOnAct.setCheckable(True)
        self.cc3dOutputOnAct.setChecked(self.visual["CC3DOutputOn"])
        
        

        # Why append?
        self.visualActions.append(self.cellsAct)
        self.visualActions.append(self.borderAct)
        self.visualActions.append(self.clusterBorderAct)
        self.visualActions.append(self.cellGlyphsAct)
        self.visualActions.append(self.FPPLinksAct)
#        self.visualActions.append(self.FPPLinksColorAct)
        #self.visualActions.append(self.plotTypeAct)
        # self.visualActions.append(self.contourAct)
        self.visualActions.append(self.limitsAct)
        self.visualActions.append(self.cc3dOutputOnAct)
        # self.visualActions.append(self.configAct)

#    def __fppLinksTrigger(self):
##        print MODULENAME,'----- __fppLinksTrigger called'
##        self.FPPLinksColorAct.setChecked(self.visual["FPPLinksColorOn"])
#        self.FPPLinksColorAct.setChecked(0)
#        
#    def __fppLinksColorTrigger(self):
##        print MODULENAME,'----- __fppLinksColorTrigger called'
##        self.FPPLinksColorAct.setChecked(self.visual["FPPLinksColorOn"])
#        self.FPPLinksAct.setChecked(0)
        
    def __initToolsActions(self):
        self.configAct = QAction(QIcon(gip("config.png")), "&Configuration...", self)
        
        self.configAct.setShortcut(Qt.CTRL + Qt.Key_Comma)
        
        self.configAct.setWhatsThis(self.trUtf8(
            """<b>Configuration</b>"""
            """<p>Set the configuration items of the simulation"""
            """ with your prefered values.</p>"""
        ))    

        # self.pifFromVTKAct = QAction("& Generate PIF File from VTK output ...", self)
        # self.configAct.setWhatsThis(self.trUtf8(
            # """<b>Generate PIF file from VTK output </b>"""
            # """<p>This will only work in the VTK simulation replay mode."""
            # """ Make sure you generated vtk output and then load *.dml file.</p>"""
        # ))    

        self.pifFromSimulationAct = QAction("& Generate PIF File from current snapshot ...", self)
        self.configAct.setWhatsThis(self.trUtf8(
            """<b>Generate PIF file from current simulation snapshot </b>"""
        ))    
        
        self.toolsActions.append(self.configAct)
        self.toolsActions.append(self.pifFromSimulationAct)
        # self.toolsActions.append(self.pifFromVTKAct)
        # self.pifGenAct = QAction("&Generate PIF", self)
        # self.pifGenAct.setCheckable(True)
        
        # Not implemented in version 3.4.0
        # self.pifVisAct = QAction("&PIF Visualizer", self)
        
        # # # self.movieAct = QAction("&Generate Movie", self)
        # # # self.movieAct.setCheckable(True)
        #self.movieAct.setChecked(True)
        
        # Why append?
        # self.toolsActions.append(self.pifGenAct)
        #self.toolsActions.append(self.pifVisAct)
        # # # self.toolsActions.append(self.movieAct)
    def __initWindowActions(self):        
        self.newGraphicsWindowAct = QAction(QIcon(gip("kcmkwm.png")),"&New Graphics Window", self)
        # self.newPlotWindowAct = QAction(QIcon("player/icons/plot.png"),"&New Plot Window", self)        
        self.tileAct=QAction("Tile", self)
        self.cascadeAct=QAction("Cascade", self)
        
        self.saveWindowsGeometryAct = QAction("Save Window(s) Geometry", self)
        
        self.minimizeAllGraphicsWindowsAct=QAction("Minimize All Graphics Windows",self)
        
        self.minimizeAllGraphicsWindowsAct.setShortcut(self.tr("Ctrl+Alt+M"))
        
        self.restoreAllGraphicsWindowsAct=QAction("Restore All Graphics Windows",self)
        self.restoreAllGraphicsWindowsAct.setShortcut(self.tr("Ctrl+Alt+N"))
        
        self.closeActiveWindowAct=QAction("Close Active Window", self)
        self.closeActiveWindowAct.setShortcut(self.tr("Ctrl+F4"))
        
        
        self.closeAdditionalGraphicsWindowsAct=QAction("Close Additional Graphics Windows", self)
                
        self.windowActions.append(self.newGraphicsWindowAct)
        # self.windowActions.append(self.newPlotWindowAct)
        self.windowActions.append(self.tileAct)
        self.windowActions.append(self.cascadeAct)
        self.windowActions.append(self.saveWindowsGeometryAct)
        
        self.windowActions.append(self.minimizeAllGraphicsWindowsAct)
        self.windowActions.append(self.restoreAllGraphicsWindowsAct)
        
        self.windowActions.append(self.closeActiveWindowAct)
        self.windowActions.append(self.closeAdditionalGraphicsWindowsAct)
        
        
    def __initHelpActions(self):
        self.quickAct = QAction("&Quick Start", self)
        self.tutorAct = QAction("&Tutorials", self)
        self.refManAct = QAction(QIcon(gip("man.png")), "&Reference Manual", self)
        self.aboutAct = QAction(QIcon(gip("cc3d_64x64_logo.png")), "&About CompuCell3D", self)
        self.connect(self.aboutAct, SIGNAL('triggered()'), self.__about)
        
        self.whatsThisAct = QAction(QIcon(gip("whatsThis.png")), "&What's This?", self)
        self.whatsThisAct.setWhatsThis(self.trUtf8(
            """<b>Display context sensitive help</b>"""
            """<p>In What's This? mode, the mouse cursor shows an arrow with a question"""
            """ mark, and you can click on the interface elements to get a short"""
            """ description of what they do and how to use them. In dialogs, this"""
            """ feature can be accessed using the context help button in the"""
            """ titlebar.</p>"""
        ))
        self.connect(self.whatsThisAct, SIGNAL('triggered()'), self.__whatsThis)
        
        # Why append?
        self.helpActions.append(self.quickAct)
        self.helpActions.append(self.tutorAct)
        self.helpActions.append(self.refManAct)
        self.helpActions.append(self.aboutAct)
        self.helpActions.append(self.whatsThisAct)

    # def __initTabActions(self):
        # self.closeTab = QToolButton(self)
        # self.closeTab.setIcon(QIcon("player/icons/close.png"))
        # self.closeTab.setToolTip("Close the tab")
        # self.closeTab.hide()
    
    def __about(self):
        versionStr='3.6.0'    
        revisionStr='0'
    
        try:
            import Version            
            versionStr=Version.getVersionAsString()    
            revisionStr=Version.getSVNRevisionAsString()    
        except ImportError,e:
            pass
            
        # import Configuration
        # versionStr=Configuration.getVersion()
        aboutText= "<h2>CompuCell3D</h2> Version: "+versionStr+" Revision: "+revisionStr+"<br />\
                          Copyright &copy; Biocomplexity Institute, <br />\
                          Indiana University, Bloomington, IN\
                          <p><b>CompuCell Player</b> is a visualization engine for CompuCell.</p>"
        lMoreInfoText="More information at:<br><a href=\"http://www.compucell3d.org/\">http://www.compucell3d.org/</a>"

        lVersionString = "<br><br><small><small>Support library information:<br>Python runtime version: %s<br>Qt runtime version: %s<br>Qt compile-time version: %s<br>PyQt version: %s</small></small>" % \
            ( str(sys.version_info[0])+"."+str(sys.version_info[1])+"."+str(sys.version_info[2])+" - "+str(sys.version_info[3])+" - "+str(sys.version_info[4]) , \
            qVersion(), QT_VERSION_STR, PYQT_VERSION_STR)
#            PyQt4.QtCore.QT_VERSION_STR, PyQt4.QtCore.qVersion(), PyQt4.QtCore.PYQT_VERSION_STR)

        QMessageBox.about(self, "CompuCell3D", aboutText+lMoreInfoText+lVersionString)
          
    def __whatsThis(self):
        """
        Private slot called in to enter Whats This mode.
        """
        QWhatsThis.enterWhatsThisMode()
   
    
    def __TBMenuTriggered(self, act):
        """
        Private method to handle the toggle of a toolbar.
        
        @param act reference to the action that was triggered (QAction)
        """

        name = unicode(act.data().toString())
        if name:
            tb = self.__toolbars[name][1]
            if act.isChecked():
                tb.show()
            else:
                tb.hide()
        
    # Any commented methods that are related to the tabs should not be 
    # in the class ViewManager!        
    #def closeTabWindow(self, tabIdx):
        """
        Public method to close an arbitrary tab.
        
        @param tab to be closed
        """
        
    #    self.removeTab(tabIdx)
        
        """
        if tab is None:
            return
        
        res = self.closeTab(tab)
        if res and tab == self.currentTab:
            self.currentTab = None
        """
        #print tab

    #def closeTab(self, tabIdx):
        """
        Public method to close a tab window.
        
        @param tab tab window to be closed
        @return flag indicating success (boolean)
        """
        
        #self.removeTab(tabIdx)
        
        """
        # This complication can be used later. Now I need just to close the plugin tab!

        # save file if necessary
        if not self.checkDirty(editor):
            return False
        
        # get the filename of the editor for later use
        fn = editor.getFileName()
        
        # remove the window
        self._removeView(editor)
        self.editors.remove(editor)
        
        # send a signal, if it was the last editor for this filename
        if fn and self.getOpenEditor(fn) is None:
            self.emit(SIGNAL('editorClosed'), fn)
        self.emit(SIGNAL('editorClosedEd'), editor)
        
        # send a signal, if it was the very last editor
        if not len(self.editors):
            self.__lastEditorClosed()
            self.emit(SIGNAL('lastEditorClosed'))
        """
        
        #return True