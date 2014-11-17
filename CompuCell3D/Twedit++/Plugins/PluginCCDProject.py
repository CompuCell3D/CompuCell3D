"""
    TO DO:
    * Keyboard events - Del
    * New Simulation wizard
    * resource properties display
    * 
"""

"""
Module used to link Twedit++ with CompuCell3D.
"""
# THIS HAS TO BE REVRITTEN USING MVC, otherwise ti is hard to maintain

from PyQt4.QtCore import QObject, SIGNAL, QString
from PyQt4.QtGui import QMessageBox

from PyQt4 import QtCore, QtGui

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import os.path
import string
from CC3DProject  import CC3DProject_rc
from CC3DProject.Configuration  import Configuration
import re
import os

# Start-Of-Header
name = "CC3D Project Plugin"
author = "Maciej Swat"
autoactivate = True
deactivateable = False
version = "0.9.0"
className = "CC3DProject"
packageName = "__core__"
shortDescription = "Plugin to manage CC3D Projects"
longDescription = """This plugin provides functionality that allows users to manage *.cc3d projects"""
# End-Of-Header

error = QString("")

# this is bidirectional dictionary - tree-item to CC3DResource and path of the resource to item
class ItemLookupData:
    def __init__(self):
        self.itemToResource={}
        self.pathToItem={}
        self.dirtyFlag=False
        
        self.itemToGenericResource={}
        self.genericResourceToItem={}
        # here we will store twedit tabs and associated file names that were opened from Project widget .
        # Later before closing we will ask users if they want to save documents in those tabs
        # if the tab does not exist or document changed name we will ignore such tab
        # to make sure that we don't store too many items before opening new document from project widget 
        # we will make sure that tab for previously opened tab are removed dictionary before reopening new one    
        self.projectLinkedTweditTabs={}
        
    def insertnewGenericResource(self,_item,_resource):    
        self.itemToGenericResource[_item]=_resource
        self.genericResourceToItem[_resource]=_item

        
    def insertNewItem(self,_item,_fullPath):    
        self.itemToResource[_item]=_fullPath
        self.pathToItem[_fullPath]=_item
        
    def removeItem(self,_item):
        try:
            path=self.itemToResource[_item]
            del self.itemToResource[_item]
            del self.pathToItem[path]
        except :
            pass
            
        try:
            resource=self.itemToGenericResource[_item]
            del self.itemToGenericResource[_item]
            del self.genericResourceToItem[resource]
        except :
            pass
            
    def getResourceName(self,_item):
        try:
            return self.itemToGenericResource[_item].resourceName
        except LookupError,e:
            return ''
        except :
            return ''

    def getResource(self,_item):
        try:
            return self.itemToGenericResource[_item]
        except LookupError,e:
            return None
        except :
            return None
            
            
    def getFullPath(self,_item):          
        try:
            return self.itemToResource[_item].path
        except LookupError,e:
            return ""
    
    
class CC3DProjectTreeWidget(QTreeWidget):
    def __init__(self,parent=None):
        QTreeWidget.__init__(self,parent)
        self.plugin=None
        self.__ui=None # Twedit++ user interface    
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setColumnCount(1)
        self.setItemsExpandable(True)
        self.setHeaderLabels(["CC3D Simulation"])
        
        self.projects={}
        self.itemToProject={}
        self.style=None # np++ style - usually this is Global override style defined in themes xml file
        self.N2C=None # convenience function reference from theme manager to convert npp color convention to QColor
        self.itemChanged.connect(self.__restyle)
        
        self.__iconDict={} # used to store icons for actions shown in the context menu - have to do this becaue of qt quirks on OSX  
        self.hideContextMenuIcons=False        
        import platform
        mac_ver=platform.mac_ver()
        if mac_ver[0]:
            self.hideContextMenuIcons=True # on OSX we hide context menu icons
            
            
            
    def setCC3DProjectPlugin(self,_plugin): 
        """
            Set reference to CC3DProject plugin
        """
        
        self.plugin=_plugin
        self.__ui=self.plugin.getUI()
    
    # get super-parent for the item - this is project item (all items belonging to the projects are under this item)  
    def getProjectParent(self,_item):
        if not _item:
            return _item
            
        curItem=_item   
        parentItem=curItem.parent()
        while parentItem:
            curItem=parentItem
            parentItem=curItem.parent()
            
        return curItem    
        
    def getFullPath(self,_item):
        #first determine the parent
        
        projParent=self.getProjectParent(_item)
        if not projParent:
            return ""
        
        print "projParent=",projParent.text(0)
        
        ild=self.projects[projParent]
        
                
        return ild.getFullPath(_item)


    def getResourceName(self,_item):
        #first determine the parent
        
        projParent=self.getProjectParent(_item)
        if not projParent:
            return ""
        
        # print "projParent=",projParent.text(0)
        
        ild=self.projects[projParent]
        
                
        return ild.getResourceName(_item)
        
    def getCurrentResource(self):
        return self.getResource(self.currentItem())
        
    def getResource(self,_item):
        #first determine the parent
        
        projParent=self.getProjectParent(_item)
        if not projParent:
            return ""
        
        # print "projParent=",projParent.text(0)
        
        ild=self.projects[projParent]
        
                
        return ild.getResource(_item)

        
        
        
    def getItemByText(self,_parentItem,_text=""):    
        if not _parentItem:
            return None
        for i in range(_parentItem.childCount()):    
            childItem=_parentItem.child(i)
            text=str(childItem.text(0))
            if text==str(_text):
                return childItem
                
        return None    
        
    def mouseDoubleClickEvent(self,event):
        
        projItem=self.getProjectParent(self.currentItem())       
        
        if not projItem:
            return 
        
        # print 'self.getFullPath(self.currentItem()=',self.getFullPath(self.currentItem())
        
        if self.getFullPath(self.currentItem())!="":
            self.plugin.actions["Open In Editor"].trigger()
            
        elif projItem==self.currentItem():
            self.plugin.actions["Open XML/Python In Editor"].trigger()
            
    def restoreIcons(self):

        for action,icon in self.__iconDict.iteritems():
            action.setIcon(icon)
        
        
    def addActionToContextMenu(self,_menu,_action):


        if self.hideContextMenuIcons:            
            self.__iconDict[_action]=_action.icon()                
            _action.setIcon(QIcon())
        
        _menu.addAction(_action)
        
    def contextMenuEvent(self , event):
        
        self.__iconDict={} # resetting icon dictionary
    
        menu=QMenu(self)
        
        menu.aboutToHide.connect(self.restoreIcons)
        
        projItem=self.getProjectParent(self.currentItem())       
        
        pdh=None
        try:
            pdh=self.plugin.projectDataHandlers[projItem]                    
        except LookupError,e:
            print "could not find simulation data handler for this item"
            return              
        
        if self.currentItem()==projItem:            
            
            self.addActionToContextMenu(menu,self.plugin.actions["Open XML/Python In Editor"])
            self.addActionToContextMenu(menu,self.plugin.actions["Open in Player"])
            self.addActionToContextMenu(menu,self.plugin.actions["Add Parameter Scan"])
            
            
            
            #--------------------------------------------------------------------
            menu.addSeparator()
            if not pdh.cc3dSimulationData.serializerResource:
                self.addActionToContextMenu(menu,self.plugin.actions["Add Serializer..."])                
                #--------------------------------------------------------------------
                menu.addSeparator()
            
        
        # menu.addAction(self.plugin.actions["Open CC3D Project..."])
        if self.getFullPath(self.currentItem())!="":
            self.addActionToContextMenu(menu,self.plugin.actions["Open In Editor"])
            self.addActionToContextMenu(menu,self.plugin.actions["Properties"])
            #--------------------------------------------------------------------
            self.addGenerateSteppableMenu(menu,projItem)
            self.addConvertXMLToPythonMenu(menu,projItem)            
            
            
            
            
            menu.addSeparator()
            
        # parameter scan menus        
        self.addParameterScanMenus(menu,projItem)            
            
        resourceName=self.getResourceName(self.currentItem())
        print '\n\n\n RESOURCENAME',resourceName
        if resourceName=='CC3DSerializerResource':            
            self.addActionToContextMenu(menu,self.plugin.actions["Serializer..."])    
            
        # if resourceName=='CC3DParameterScanResource':            
            # menu.addAction(self.plugin.actions["Reset Parameter Scan"])    
        
        self.addActionToContextMenu(menu,self.plugin.actions["Save CC3D Project"])
        self.addActionToContextMenu(menu,self.plugin.actions["Add Resource..."])
        
        
        

        
        # if selection.size():
            # menu.addAction(self.plugin.actions["Remove Resources"])
            
        self.addActionToContextMenu(menu,self.plugin.actions["Remove Resources"])
        
        
        
        print "CurrentItem=",self.currentItem().text(0)," parent=",self.currentItem().parent()
        print "getFullPath=", self.getFullPath(self.currentItem())
        # if self.getFullPath(self.currentItem())!="":
            # #--------------------------------------------------------------------
            # menu.addSeparator()
            # menu.addAction(self.plugin.actions["Open In Editor"])
            
        #--------------------------------------------------------------------
        menu.addSeparator()
        self.addActionToContextMenu(menu,self.plugin.actions["Close Project"])
            
        # if self.currentItem().parent()==self:            
            # print "GOT TOP LEVEL ITEM"
            
        menu.exec_(event.globalPos())
        
    def addGenerateSteppableMenu(self,_menu,_projItem):
    
        # print "TRYING TO ADD GENERATE STEPPEBLE MENU"
        
        pdh=None
        try:
           pdh = self.plugin.projectDataHandlers[_projItem]
        except LookupError,e:
        
            return          
        # check if thei file to which we are trying to add Steppable is Python resource        
        itemFullPath=str(self.getFullPath(self.currentItem()))
        basename, extension = os.path.splitext(itemFullPath)
        
        basename = os.path.basename(itemFullPath)
        # print "basename=",basename," ext=",extension
        
        try:
            cc3dResource=pdh.cc3dSimulationData.resources[itemFullPath]
            if cc3dResource.type=="Python":
                self.addActionToContextMenu(_menu,self.plugin.actions["Add Steppable..."])
                                 
        except LookupError,e:
            return
                        
                        
    def addParameterScanMenus(self,_menu,_projItem):
        pdh=None
        try:
           pdh = self.plugin.projectDataHandlers[_projItem]
        except LookupError,e:
        
            return          
        _menu.addSeparator()    
        resourceName=self.getResourceName(self.currentItem())    
        
        itemFullPath=str(self.getFullPath(self.currentItem()))
        basename, extension = os.path.splitext(itemFullPath)
        #adding menu to parameter scan xml file
        if pdh.cc3dSimulationData.parameterScanResource and itemFullPath==pdh.cc3dSimulationData.parameterScanResource.path:
            self.addActionToContextMenu(_menu,self.plugin.actions["Reset Parameter Scan"])
            
        #adding menu to parameter scan node
        if resourceName=='CC3DParameterScanResource':            
            self.addActionToContextMenu(_menu,self.plugin.actions["Reset Parameter Scan"])    
            
            
            
        try:
            cc3dResource=pdh.cc3dSimulationData.resources[itemFullPath]
            if cc3dResource.type=="Python":
                self.addActionToContextMenu(_menu,self.plugin.actions["Open Scan Editor"])
                                 
        except LookupError,e:
            pass
        

        
        if pdh.cc3dSimulationData.xmlScript==itemFullPath or pdh.cc3dSimulationData.pythonScript==itemFullPath:
            self.addActionToContextMenu(_menu,self.plugin.actions["Open Scan Editor"])
            
        _menu.addSeparator()        
    

        
    def addConvertXMLToPythonMenu(self,_menu,_projItem):
    
        # print "TRYING TO ADD GENERATE STEPPEBLE MENU"
        
        pdh=None
        try:
           pdh = self.plugin.projectDataHandlers[_projItem]
        except LookupError,e:
        
            return          
        # check if the file to which we are trying to add Steppable is Python resource        
        itemFullPath=str(self.getFullPath(self.currentItem()))
        basename, extension = os.path.splitext(itemFullPath)
        
        print "itemFullPath=",itemFullPath
        
        print 'extension=',extension
        if extension.lower()=='.xml':
            self.addActionToContextMenu(_menu,self.plugin.actions["Convert XML to Python"])
            self.plugin.xmlFileToConvert=itemFullPath
            
            return             
        
        if pdh.cc3dSimulationData.xmlScript!='':
            self.addActionToContextMenu(_menu,self.plugin.actions["Convert XML to Python"])
            self.plugin.xmlFileToConvert=str(pdh.cc3dSimulationData.xmlScript)
            
    def __restyle(self):
        root_item=self.invisibleRootItem()
        self.styleChildItems(root_item,self.style)
                
    
    def styleChildItems(self,_item,_style): 
        if not _style: return
        
        _item.setForeground(0,QBrush(self.N2C(_style.fgColor)))        
        for idx in range(_item.childCount()):
            childItem=_item.child(idx)
            childItem.setForeground(0,QBrush(self.N2C(_style.fgColor)))
            self.styleChildItems(childItem,_style)
            
    def applyStyleFromTheme(self,_styleName,_themeName):


        
        themeManager=self.__ui.themeManager
        self.style=themeManager.getStyleFromTheme(_styleName = _styleName , _themeName = _themeName)
        
#         print '_styleName=',_styleName
#         print 'self.style=',self.style
#         import time
#         time.sleep(5)

        self.setIconSize(QSize(16,16))
        
        if self.style:
            self.N2C=themeManager.npStrToQColor
            qtVersion=str(QtCore.QT_VERSION_STR).split('.')
            if qtVersion>=2:
                bgColorQt=self.N2C(self.style.bgColor)
                colorString='rgb('+str(bgColorQt.red())+','+str(bgColorQt.green())+','+str(bgColorQt.blue())+')'
                # because we used style sheets for the qt app  (twedit_plus_plus.py) we have to use stylesheet to color QTreeWidget  
                # at least on OSX 10.9 using stylesheets for the app requires using them to set properties of widget
                self.setStyleSheet( "QTreeWidget {background-color: "+colorString+ " ;}" )
            
        else:
            pal=self.palette()
            pal.setBrush( QPalette.Base, QBrush(self.N2C(self.style.bgColor)))
            self.setPalette(pal)
            
        self.__restyle()
        
#         bgColorQt=self.N2C(self.style.bgColor)
#         colorString='rgb('+str(bgColorQt.red())+','+str(bgColorQt.green())+','+str(bgColorQt.blue())+')'
#         # because we used style sheets for the qt app  (twedit_plus_plus.py) we have to use stylesheet to color QTreeWidget  
#         # at least on OSX 10.9 using stylesheets for the app requires using them to set properties of widget
#         self.setStyleSheet( "QTreeWidget {background-color: "+colorString+ " ;}" )
 

class CustomDockWidget(QDockWidget):
    def __init__(self,_parent=None):
        QDockWidget.__init__(self,_parent)
        self.cc3dProject=None
    def setCC3DProject(self,cc3dProject):
        self.cc3dProject=cc3dProject
            
    def closeEvent(self,ev):
        print 'close event custom dock widget'
        self.cc3dProject.showProjectPanel(False)
        ev.ignore()
        
class CC3DProject(QObject):
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
        
        self.configuration=Configuration(self.__ui.configuration.settings)
        
        self.actions={}
        self.projectDataHandlers={}
        self.openProjectsDict={}
        
        self.hideContextMenuIcons=False        
        import platform
        mac_ver=platform.mac_ver()
        if mac_ver[0]:
            self.hideContextMenuIcons=True # on OSX we hide context menu icons
        
        # self.listener=CompuCell3D.CC3DListener.CC3DListener(self.__ui)
        # self.listener.setPluginObject(self)
        self.__initActions()        
        self.__initMenus()
        self.__initUI()
        self.__initToolbar()        
        self.steppableTemplates=None
        self.xmlFileToConvert=None
        
        #parameter scan globals
        self.parameterScanEditor=None # only one scan editor is allowed at any given time
        self.scannedFileName='' # this is the path to the file which is open in parameterScanEditor
        # self.parameterScanXMLHandler=None
        # self.parameterScanFile=''
        
        # self.openCC3Dproject("/Users/m/install_projects/CC3D_3.7.1/Demos/SBMLSolverExamples/SBMLSolver/SBMLSolver.cc3d")
        

        # # # self.openCC3Dproject('/home/m/CC3DProjects/CellSorting/CellSorting.cc3d')

        # # # # self.treeWidget.applyStyle(self.defaultStyle)
        self.treeWidget.applyStyleFromTheme(_styleName = 'Default Style' , _themeName = self.__ui.currentThemeName)        
        # # # self.styleItems()

        self.hideContextMenuIcons=False        
        import platform
        mac_ver=platform.mac_ver()
        if mac_ver[0]:
            self.hideContextMenuIcons=True # on OSX we hide context menu icons
        
    def getUI(self):
        return self.__ui
        
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
        # have to close all the projects
        projItems=self.projectDataHandlers.keys()
        for projItem in projItems:
            self.closeProjectUsingProjItem(projItem)
        
        showCC3DProjectPanel=self.configuration.setSetting("ShowCC3DProjectPanel",not self.cc3dProjectDock.isHidden())
      
        
        return
        # print "DEACTIVATE CC3D PLUGIN"
        # self.listener.deactivate()
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
            
        self.__ui.toolBar["CompuCell3D"].addAction(self.actions["Open CC3D Project..."])
            
    def __initMenus(self):
        
        
        self.cc3dProjectMenu=QMenu("CC3D Projec&t",self.__ui.menuBar())
        #inserting CC3D Project Menu as first item of the menu bar of twedit++
        self.__ui.menuBar().insertMenu(self.__ui.fileMenu.menuAction(),self.cc3dProjectMenu)
        
        self.cc3dProjectMenu.addAction(self.actions["New CC3D Project..."])
        self.cc3dProjectMenu.addAction(self.actions["Open CC3D Project..."])
        self.cc3dProjectMenu.addAction(self.actions["Save CC3D Project"])
        self.cc3dProjectMenu.addAction(self.actions["Save CC3D Project As..."])
        self.cc3dProjectMenu.addSeparator()
        #---------------------------------------
        
        self.cc3dProjectMenu.addAction(self.actions["Open in Player"])
        self.cc3dProjectMenu.addSeparator()
        #---------------------------------------
        
        
        # self.cc3dProjectMenu.addAction(self.actions["Save CC3D Project As..."])
        self.cc3dProjectMenu.addAction(self.actions["Add Resource..."])
        self.cc3dProjectMenu.addAction(self.actions["Remove Resources"])
        self.cc3dProjectMenu.addSeparator()
        #---------------------------------------
        self.cc3dProjectMenu.addAction(self.actions["Open In Editor"])
        self.cc3dProjectMenu.addAction(self.actions["Open XML/Python In Editor"])
        
        self.cc3dProjectMenu.addSeparator()
        #---------------------------------------------------
        #Parameter scan Menu
        self.cc3dProjectMenu.addAction(self.actions["Add Parameter Scan"])
        self.cc3dProjectMenu.addAction(self.actions["Add To Scan..."])
        
        

        self.cc3dProjectMenu.addSeparator()
        #---------------------------------------
        self.recentProjectsMenu = self.cc3dProjectMenu.addMenu("Recent Projects...")
        self.connect(self.recentProjectsMenu , SIGNAL("aboutToShow()"), self.updateRecentProjectsMenu)
        
        self.recentProjectDirectoriesMenu=self.cc3dProjectMenu.addMenu("Recent Project Directories...")        
        self.connect(self.recentProjectDirectoriesMenu , SIGNAL("aboutToShow()"), self.updateRecentProjectDirectoriesMenu)

        
        self.cc3dProjectMenu.addSeparator()
        #---------------------------------------
        self.cc3dProjectMenu.addAction(self.actions["Show Project Panel"])
        
        
        self.cc3dProjectMenu.addSeparator()
        #---------------------------------------
        
        self.cc3dProjectMenu.addAction(self.actions["Close Project"])
    
    def __loadRecentProject(self):
        print '__loadRecentProject'
        action=self.sender()
        fileName=''
        if isinstance(action,QAction):            
            fileName=str(action.data().toString())            
            self.openCC3Dproject(fileName)
        
    def __openRecentProjectDirectory(self):
        action=self.sender()      
        if isinstance(action,QAction):            
            fileName=str(action.data().toString())            
            self.openCC3Dproject(fileName)
        
        
    def __openRecentProjectDirectory(self):        
        action=self.sender()
        dirName=''
        if isinstance(action,QAction):            
            dirName=str(action.data().toString())            
            dirName=os.path.abspath(dirName)
            self.__ui.addItemtoConfigurationStringList(self.configuration,"RecentProjectDirectories",dirName)                        
            self.showOpenProjectDialogAndLoad(dirName)
            
 
    def updateRecentProjectsMenu(self):
        self.__ui.updateRecentItemMenu(self,self.recentProjectsMenu,self.__loadRecentProject,self.configuration,"RecentProjects")
    
    def updateRecentProjectDirectoriesMenu(self):
        self.__ui.updateRecentItemMenu(self,self.recentProjectDirectoriesMenu,self.__openRecentProjectDirectory,self.configuration,"RecentProjectDirectories")
        
        
    def applyStyleFromTheme(self,_styleDict):
        print '_styleDict=',_styleDict
        try:
            styleName=_styleDict['styleName']
            themeName=_styleDict['themeName']
            print 'self.treeWidget=',self.treeWidget
            
            self.treeWidget.applyStyleFromTheme(_styleName = styleName , _themeName = themeName)
        except LookupError,e:
            return
            
    def __initUI(self):
        self.cc3dProjectDock=self.__createDockWindow("CC3D Project")
        self.textEdit=QTextEdit()
        self.treeWidget=CC3DProjectTreeWidget()
        self.treeWidget.setCC3DProjectPlugin(self)
        
        
        self.__setupDockWindow(self.cc3dProjectDock,Qt.LeftDockWidgetArea,self.treeWidget,"CC3D Project")
        
        showCC3DProjectPanel=self.configuration.setting("ShowCC3DProjectPanel")
        if not showCC3DProjectPanel:
            self.showProjectPanel(False)
        
    def __createDockWindow(self, name):
        """
        Private method to create a dock window with common properties.
        
        @param name object name of the new dock window (string or QString)
        @return the generated dock window (QDockWindow)
        """

        dock=CustomDockWidget(self.__ui)
        dock.setCC3DProject(self)
#         dock = QDockWidget(self.__ui)
        
        dock.setObjectName(name)
        #dock.setFeatures(QDockWidget.DockWidgetFeatures(QDockWidget.AllDockWidgetFeatures))
        return dock
     
    def __setupDockWindow(self, dock, where, widget, caption):
        """
        Private method to configure the dock window created with __createDockWindow().
        
        @param dock the dock window (QDockWindow)
        @param where dock area to be docked to (Qt.DockWidgetArea)
        @param widget widget to be shown in the dock window (QWidget)
        @param caption caption of the dock window (string or QString)
        """
        if caption is None:
            caption = QString()
        self.__ui.addDockWidget(where, dock)
        dock.setWidget(widget)
        dock.setWindowTitle(caption)
        dock.show()
        
    def __initActions(self):
        """
        Private method to initialize the actions.
        """
        # print "BEFORE IMPORTS"
        self.actions["New CC3D Project..."]=QtGui.QAction(QIcon(':/icons/new-project.png'),"New CC3D Project...", self, shortcut="Ctrl+Shift+N", statusTip="New CC3D Project Wizard ", triggered=self.__newCC3DProject)
        self.actions["Open CC3D Project..."]=QtGui.QAction(QIcon(':/icons/open-project.png'),"Open CC3D Project...", self, shortcut="Ctrl+Shift+O", statusTip="Open CC3D Project ", triggered=self.__openCC3DProject)
        self.actions["Open in Player"]=QtGui.QAction(QIcon(':/icons/player-icon.png'),"Open In Player", self, shortcut="", statusTip="Open simulation in Player ", triggered=self.__runInPlayer) 
        self.actions["Save CC3D Project"]=QtGui.QAction(QIcon(':/icons/save-project.png'),"Save CC3D Project", self, shortcut="", statusTip="Save CC3D Project ", triggered=self.__saveCC3DProject)
        self.actions["Save CC3D Project As..."]=QtGui.QAction("Save CC3D Project As...", self, shortcut="Ctrl+Shift+A", statusTip="Save CC3D Project As ", triggered=self.__saveCC3DProjectAs)
        
        
        self.actions["Add Resource..."]=QtGui.QAction(QIcon(':/icons/add.png'),"Add Resource...", self, shortcut="", statusTip="Add Resource File ", triggered=self.__addResource)
        self.actions["Add Serializer..."]=QtGui.QAction(QIcon(':/icons/add-serializer.png'),"Add Serializer ...", self, shortcut="", statusTip="Add Serializer ", triggered=self.__addSerializerResource)        
        
        self.actions["Remove Resources"]=QtGui.QAction(QIcon(':/icons/remove.png'),"Remove Resources", self, shortcut="", statusTip="Remove Resource Files ", triggered=self.__removeResources)        
        
        self.actions["Open In Editor"]=QtGui.QAction(QIcon(':/icons/open-in-editor.png'),"Open In Editor", self, shortcut="", statusTip="Open Document in Editor ", triggered=self.__openInEditor)
        self.actions["Open XML/Python In Editor"]=QtGui.QAction("Open XML/Python In Editor", self, shortcut="", statusTip="Open XML and Python scripts from the current project in editor ", triggered=self.__openXMLPythonInEditor)
        
        self.actions["Properties"]=QtGui.QAction("Properties", self, shortcut="", statusTip="Display/Edit Project Item Properties ", triggered=self.__displayProperties) 
        
        self.actions["Serializer..."]=QtGui.QAction(QIcon(':/icons/save-simulation.png'),"Serializer...", self, shortcut="", statusTip="Edit serialization properties fo the simulation ", triggered=self.__serializerEdit)
        
        self.actions["Close Project"]=QtGui.QAction("Close Project", self, shortcut="Ctrl+Shift+X", statusTip="Close Project ", triggered=self.__closeProject)
        
        self.actions["Show Project Panel"]=QtGui.QAction("Show Project Panel", self, shortcut="", statusTip="Show Project Panel")
        self.actions["Show Project Panel"].setCheckable(True)
        self.actions["Show Project Panel"].setChecked(True)
        self.connect(self.actions["Show Project Panel"],    SIGNAL('triggered(bool)'),  self.showProjectPanel)
        
        self.actions["Add Steppable..."]=QtGui.QAction(QIcon(':/icons/addSteppable.png'),"Add Steppable...", self, shortcut="", statusTip="Adds Steppable to Python File (Cannot be Python Main Script) ", triggered=self.__addSteppable)
        
        self.actions["Convert XML to Python"]=QtGui.QAction(QIcon(':/icons/xml-icon.png'),"Convert XML to Python", self, shortcut="", statusTip="Converts XML into equivalent Python script", triggered=self.__convertXMLToPython)
        
        self.actions["Add Parameter Scan"]=QtGui.QAction(QIcon(':/icons/scan_32x32.png'),"Add Parameter Scan", self, shortcut="Ctrl+Shift+P", statusTip="Add Parameter Scan ", triggered=self.__addParameterScan)        

        #on osx 10.9 context menu icons are not rendered properly so we do not include them at all on OSX
        if self.hideContextMenuIcons: addToScanIcon=QIcon()
        else:addToScanIcon=QIcon(':/icons/add.png')               

        self.actions["Add To Scan..."]=QtGui.QAction(addToScanIcon,"Add To Scan...", self, shortcut="Ctrl+I", statusTip="Add Parameter To Scan", triggered=self.__addToScan)        
        self.actions['Open Scan Editor']=QtGui.QAction(QIcon(':/icons/editor.png'),"Open Scan Editor", self, shortcut="", statusTip="Open Scan Editor", triggered=self.__openScanEditor)        
        self.actions['Reset Parameter Scan']=QtGui.QAction(QIcon(':/icons/reset_32x32.png'),"Reset Parameter Scan", self, shortcut="", statusTip="Reset Parameter Scan", triggered=self.__resetParameterScan)                        

    def  __resetParameterScan(self):
        tw=self.treeWidget
        ret=QMessageBox.warning(tw,"Parameter Scan Reset","You are about to reset parameter scan to start from the beginning. Do you want to proceed?",QMessageBox.Yes |QMessageBox.No)
        
        if ret == QMessageBox.No:return
            

        projItem=tw.getProjectParent(tw.currentItem())            

        pdh=None
        try:
            pdh=self.projectDataHandlers[projItem]                    
        except LookupError,e:
            print "could not find simulation data handler for this item"
            return              
        
        if pdh.cc3dSimulationData.parameterScanResource:
            from ParameterScanUtils import ParameterScanUtils as PSU
            psu=PSU()
            psu.resetParameterScan(pdh.cc3dSimulationData.parameterScanResource.path)
        #    
            # self.__ui.deactivateChangeSensing=True
            self.__ui.checkIfDocumentsWereModified()
            # self.__ui.deactivateChangeSensing=False
        
    def __addParameterScan(self):

        tw=self.treeWidget
        projItem=tw.getProjectParent(tw.currentItem())            

        pdh=None
        try:
            pdh=self.projectDataHandlers[projItem]                    
        except LookupError,e:
            print "could not find simulation data handler for this item"
            return              
        
        if pdh.cc3dSimulationData.parameterScanResource:
            QMessageBox.warning(tw,"Parameter Scan is already defined","You cannot have more than one parameter scan specifications in the simulation")
            return
        
        
        pdh.cc3dSimulationData.addNewParameterScanResource()# adding empty parameter scan resource
        resourceFileName = pdh.cc3dSimulationData.parameterScanResource.path   

        #insert new file into the tree
        self.insertNewGenericResourceTreeItem(pdh.cc3dSimulationData.parameterScanResource)
        
        pdh.cc3dSimulationData.parameterScanResource.writeParameterScanSpecs()
        pdh.cc3dSimulationData.parameterScanResource.basePath=pdh.cc3dSimulationData.basePath # setting same base path for parameter scan as for the project - necessary to get relative paths in the parameterSpec file        

        self.__saveCC3DProject()

            
            
    def  __closeScanEditor(self):
        if not self.parameterScanEditor :return
        
        
        panel,idx=self.__ui.getTabWidgetAndWidgetIndex(self.parameterScanEditor)
        if panel and idx>=0:
            self.__ui.closeTab( index=idx,_askToSave=False,_panel=panel)
        self.parameterScanEditor=None
        
    # def  __closeScanEditorEvent(self,event):
        # print 'LOCAL CLOSE EVENT'
        # self.parameterScanEditor=None
        
    def __openScanEditor(self):        
    
        if self.parameterScanEditor :
            self.__closeScanEditor()

        import os
        
        tw=self.treeWidget
        
        projItem=tw.getProjectParent(tw.currentItem())
        
        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return                    
        
        
        
        if not pdh.cc3dSimulationData.parameterScanResource: 
            QMessageBox.warning(tw,"Please Add Parameter Scan","Parameter scan editor can only be open when project includes Parameter Scan. Please add parameter scan first ")
            return
        
            print ''
            return
        
        pScanResource=pdh.cc3dSimulationData.parameterScanResource
        
        itemFullPath=str(tw.getFullPath(tw.currentItem()))
        basename, extension = os.path.splitext(itemFullPath)
        
        pScanResource.fileTypeForEditor=extension.lower()
        
        self.scannedFileName=itemFullPath # store scanned file name in the global - we can only have one parameter scan editor open 
        
        #opening editor
        self.__ui.newFile()
        editor=self.__ui.getCurrentEditor()
        editor.setReadOnly(True)
        

        
        # # set tab font color color 
        # tabBar=activePanel.tabBar()
        # tabBar.setTabIcon()
        # tabBar.setStyleSheet('background-color: blue;')
        
        lexer=self.__ui.guessLexer("tmp"+pScanResource.fileTypeForEditor)
        if lexer[0]:
            editor.setLexer(lexer[0])
        self.__ui.setEditorProperties(editor)         


        
        editor.registerCustomContextMenu(self.createParameterScanMenu(editor))
        
        #initialize globals
        self.parameterScanEditor=editor      
        
        # self.parameterScanEditor.closeEvent=self.__closeScanEditorEvent # close event will be handled via local function 

        # pScanResource.parameterScanEditor=editor
        
        if pScanResource.fileTypeForEditor=='.xml': # for xml we have to get generate line to access path map and line to element map for easier handling of parameter scan generation
            
            import XMLUtils
            import os
            cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
            root_element=cc3dXML2ObjConverter.Parse(self.scannedFileName)
            
            from ParameterScanUtils import XMLHandler        
            xmlHandler=XMLHandler()
            
            xmlHandler.outputXMLWithAccessPaths(self.scannedFileName)
            
            print xmlHandler.lineToElem
            print xmlHandler.lineToAccessPath
            
        
            editor.insertAt(xmlHandler.xmlString,0,0)
            editor.setModified(False)
            pScanResource.parameterScanXMLHandler=xmlHandler
                        
        if pScanResource.fileTypeForEditor=='.py':    
            
            editor.insertAt(open(self.scannedFileName).read(),0,0)
            editor.setModified(False)
            
        # setting graphical  properties for parameter scan editor tab widget      
        activePanel, currentindex=self.__ui.getCurrentTabWidgetAndIndex()
        activePanel.setTabText(currentindex,'Parameter Scan Tmp File')
        activePanel.setTabIcon(currentindex,QIcon(':/icons/scan_32x32.png'))
        tabBar=activePanel.tabBar()        
        tabBar.setTabTextColor(currentindex,QColor('blue'))
            
        
    def __addToScan(self):
        
        tw=self.treeWidget
        
        projItem=tw.getProjectParent(tw.currentItem())
        print'projItem=',projItem
        
        print 'self.projectDataHandlers=',self.projectDataHandlers
        
        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return                    
        
        
        csd=pdh.cc3dSimulationData
        pScanResource=pdh.cc3dSimulationData.parameterScanResource
        
    
        print '__addToScan'
        if not self.parameterScanEditor:return
        
        #check if the editor is still open
        editorExists=self.__ui.checkIfEditorExists(self.parameterScanEditor)
        if not  editorExists:            
            self.parameterScanEditor=None
            return
        
        line,col=self.parameterScanEditor.getCursorPosition()
        print 'line,col=',(line,col)
        
        
        # if pScanResource
        scannedFileExt= os.path.splitext(self.scannedFileName)[1].lower() 
        if scannedFileExt == '.xml':
            psXMLHandler=pScanResource.parameterScanXMLHandler
            if psXMLHandler:
                try:
                    accessPath=psXMLHandler.lineToAccessPath[line]
                    xmlElem=psXMLHandler.lineToElem[line]
                except LookupError,e:
                    accessPath=''
                
                print 'AccessPath=',accessPath
            
                
            if not accessPath:
                return
                
            print 'self.scannedFileName=',self.scannedFileName,'\n\n\n\n\n'
            from CC3DProject.ParameterDialog import  ParameterDialog
            pdlg=ParameterDialog(self.parameterScanEditor)
            print 'DICT BEFORE=',csd.parameterScanResource.parameterScanFileToDataMap
            try:
                pdlg.displayXMLScannableParameters(xmlElem,psXMLHandler.lineToAccessPath[line],pScanResource.path)
                ret=pdlg.exec_()
                if ret:                
                    haveNewItems=False
                    csd.parameterScanResource.psu.refreshParamSpecsContent(pScanResource.path) # before adding new parameter scan we need to read file again om case user made any change
                    for key,val in pdlg.parameterScanDataMap.iteritems():
                        print 'Adding key,val=',(key,val)                        
                        csd.parameterScanResource.addParameterScanData(self.scannedFileName,val)
                        haveNewItems=True
                        
                        # csd.parameterScanResource.parameterScanDataMap[key]=val ###
                    if haveNewItems:    
                        pScanResource.writeParameterScanSpecs()

            except LookupError,e: # to protect against elements that are not in psXMLHandler.lineToAccessPath
                return
                
        
        elif scannedFileExt == '.py':
            from ParameterScanUtils import ParameterScanUtils as PSU
            psu=PSU()
            
            pythonLine=str(self.parameterScanEditor.text(line))
            
            foundGlobalVar=psu.checkPythonLineForGlobalVariable(pythonLine)
            if foundGlobalVar:
                try:
                    varName,varValue=psu.extractGlobalVarFromLine(pythonLine)
                    print 'varName,varValue=',(varName,varValue)
                except:
                    QMessageBox.warning(tw,"Problem Parsing Python Line","Could Not Parse Python Line to find global variable. Make sure you declare global variables in separate lines e.g. myvar=10 ")
                    
                    
                from CC3DProject.ParValDlg import ParValDlg
                
                parvaldlg=ParValDlg(self.parameterScanEditor)
                
                from ParameterScanEnums import PYTHON_GLOBAL
                
                parvaldlg.initParameterScanData(_parValue=varValue,_parName=varName,_parType=PYTHON_GLOBAL,_parAccessPath='')
                
                if parvaldlg.exec_():
                    try:
                        parvaldlg.recordValues()
                    except ValueError,e:
                        QMessageBox.warning(tw,"Error Parsing Parameter List","Please make sure that parameter list entries have correct type")
                        return
                        
                    psd=parvaldlg.psd
                    
                    if len(psd.customValues):
                        csd.parameterScanResource.addParameterScanData(self.scannedFileName,psd)
                        pScanResource.writeParameterScanSpecs()
                        
                else:
                    #user canceled
                    return


        
        
            
        self.__ui.checkIfDocumentsWereModified()
        # # # print 'DICT AFTER=',csd.parameterScanResource.parameterScanFileToDataMap
        # # # # get path to ParameterScan XML file
        # # # parScanResources=csd.getResourcesByType ('ParameterScan') 
        # # # print 'parScanResources=',parScanResources     
        
        # # # if not csd.parameterScanResource:return
        
        # # # resource=csd.parameterScanResource
        # # # # paramScanXMLFilePath=resource.path

        # resource.writeParameterScanSpecs()

        
        
    def restoreIcons(self):
        print 'restore icons for scan menu'
        for action,icon in self.__iconDict.iteritems():
            action.setIcon(icon)
        
        
    def addActionToContextMenu(self,_menu,_action):

#         if self.hideContextMenuIcons:            
#             self.__iconDict[_action]=_action.icon()                
#             _action.setIcon(QIcon())
            
        _menu.addAction(_action)
        

        
    def  createParameterScanMenu(self,_widget):
        self.__iconDict={} # resetting icon dictionary
        self.hideContextMenuIcons=True
        
        menu=QMenu(_widget)
        
        menu.aboutToHide.connect(self.restoreIcons)
        
        self.addActionToContextMenu(menu,self.actions["Add To Scan..."])
#         menu.addAction(self.actions["Add To Scan..."])
        
        return menu
        
    
    def __serializerEdit(self):
        from CC3DProject.SerializerEdit import SerializerEdit
        se=SerializerEdit(self.treeWidget)
        
        resource=self.treeWidget.getCurrentResource()
 
        se.setupDialog(resource)
        if se.exec_():
 
            se.modifySerializerResource(resource)
            projItem=self.treeWidget.getProjectParent(self.treeWidget.currentItem())       
            self.markProjectDirty(projItem)
    
    def __convertXMLToPython(self):
        print "CONVERTING XML TO PYTHON"
        print "self.xmlFileToConvert=",self.xmlFileToConvert
        if self.xmlFileToConvert:
            import XMLUtils
            import os
            cc3dXML2ObjConverter = XMLUtils.Xml2Obj()
            root_element=cc3dXML2ObjConverter.Parse(self.xmlFileToConvert)
            
            
            dirToStoreTmpFile=os.path.dirname(self.xmlFileToConvert)
            
            tmpFilePath=os.path.join(dirToStoreTmpFile,'tmp.py')
            tmpFilePath=os.path.abspath(tmpFilePath) # normalizing the path
            import CC3DProject.CC3DPythonGenerator as cc3dPythonGen
            
            configureSimFcnBody=cc3dPythonGen.generateConfigureSimFcnBody(root_element,tmpFilePath)
            configureSimFcnBody+='\n'
            
            
            self.__ui.newFile()
            editor=self.__ui.getCurrentEditor()
            editor.insertAt(configureSimFcnBody,0,0)
            lexer=self.__ui.guessLexer("tmp.py")
            if lexer[0]:
                editor.setLexer(lexer[0])
            self.__ui.setEditorProperties(editor) 
            
            self.xmlFileToConvert=None
            
        
        

    def __addSteppable (self):
        
        # curItem here points to Python resource file meaning it is a viable file to paste steppable
        
        print "\n\n\n\n\n ADDING STEPPABLE CODE"
        tw=self.treeWidget        
        curItem=tw.currentItem()        
        
        projItem=tw.getProjectParent(curItem)   
        
        if not projItem:
            return 

        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return        
            
        mainPythonScriptPath=pdh.cc3dSimulationData.pythonScriptResource.path
        
        # check if thei file to which we are trying to add Steppable is Python resource        
        itemFullPath=str(tw.getFullPath(curItem))
        basename, extension = os.path.splitext(itemFullPath)
        
        basename = os.path.basename(itemFullPath)
        basenameForImport,ext=os.path.splitext(basename)
        
        # print "basename=",basename," ext=",extension
        
        # try:
            # cc3dResource=pdh.cc3dSimulationData.resources[itemFullPath]
            # if cc3dResource.type=="Python":
                # _menu.addAction(self.plugin.actions["Add Steppable..."])
                                 
        # except LookupError,e:
            # return
        mainScriptEditorWindow=None
        steppableScriptEditorWindow=None
        
        if mainPythonScriptPath!="":
            self.openFileInEditor(mainPythonScriptPath)
            editor=self.__ui.getCurrentEditor()
            if str(self.__ui.getCurrentDocumentName())==mainPythonScriptPath:
                mainScriptEditorWindow=editor
            
            
            
        self.openFileInEditor(itemFullPath)
        editor=self.__ui.getCurrentEditor()
        if str(self.__ui.getCurrentDocumentName())==itemFullPath:
            steppableScriptEditorWindow=editor
        
        if not steppableScriptEditorWindow:
            QMessageBox.warning(tw,"File Open Problem","Could not open steppable file in Twedit++-CC3D")
            return
            
        
        entryLine,indentationLevel=self.findEntryPointForSteppableRegistration(mainScriptEditorWindow)
                
        from CC3DProject.SteppableGeneratorDialog import SteppableGeneratorDialog
        
        sgd=SteppableGeneratorDialog(tw)
        sgd.mainScriptLB.setText(mainPythonScriptPath)
        if not sgd.exec_():
            return
            
        steppebleName=str(sgd.steppebleNameLE.text())
        frequency=sgd.freqSB.value()
        type="Generic"
        if sgd.genericLB.isChecked():
            type="Generic"
        elif  sgd.mitosisRB.isChecked():   
            type="Mitosis"
        elif  sgd.clusterMitosisRB.isChecked():   
            type="ClusterMitosis"
        elif  sgd.runBeforeMCSRB.isChecked():   
            type="RunBeforeMCS"
            
        extraFields=[]
        if sgd.scalarCB.isChecked():
            extraFields.append("Scalar")
        if sgd.scalarCellLevelCB.isChecked():
            extraFields.append("ScalarCellLevel")
        if sgd.vectorCB.isChecked():
            extraFields.append("Vector")
        if sgd.vectorCellLevelCB.isChecked():
            extraFields.append("VectorCellLevel")
            
        # adding steppable 
        
        # will instantiate steppablel templates only when needed        
        from CC3DProject.SteppableTemplates import SteppableTemplates
        if not self.steppableTemplates:
            self.steppableTemplates=SteppableTemplates()
        
        steppableCode=self.steppableTemplates.generateSteppableCode(steppebleName,frequency,type,extraFields)
                
        
        
        if steppableCode=="" :            
            QMessageBox.warning(tw,"Problem With Steppable Generation","Could not generate steppable")
            return
        
        maxLineIdx=steppableScriptEditorWindow.lines()
        col=steppableScriptEditorWindow.lineLength(maxLineIdx-1)
        steppableScriptEditorWindow.insertAt(steppableCode,maxLineIdx,col)
        
        steppableScriptEditorWindow.ensureLineVisible(maxLineIdx+20)
        
        # Registration of steppable
        
        if not mainScriptEditorWindow:
            QMessageBox.warning(tw,"Problem with Main Python script","Please edit python main script to register steppable . Could not open main Python script")
            return
        
        
        if entryLine==-1:
            QMessageBox.warning(tw,"Please check Python main script","Please edit python main script to register steppable . Could not determine where to put steppeble registration code ")
            return
            
        steppableRegistrationCode=self.steppableTemplates.generateSteppableRegistrationCode(steppebleName, frequency, basenameForImport, indentationLevel, mainScriptEditorWindow.indentationWidth())        
        
        
        if indentationLevel==-1:
            QMessageBox.warning(tw,"Possible indentation problem","Please edit python main script position properly steppable registration code ")
            
        mainScriptEditorWindow.insertAt(steppableRegistrationCode,entryLine,0)
        mainScriptEditorWindow.ensureLineVisible(maxLineIdx+10)
        
        # steppableScriptEditorWindow
        
        print "ENTRY LINE FOR REGISTRATION OF STEPPABLE IS ",entryLine

    def findEntryPointForSteppableRegistration(self,_mainScriptEditorWindow):
        mainLoopRegex=re.compile('^[\s]*CompuCellSetup\.mainLoop')
        print "Looking for entry point for steppable registration"
        if not _mainScriptEditorWindow:
            return -1.-1
        
        lastLine=_mainScriptEditorWindow.lines()-1
        for lineIdx in range(lastLine,-1,-1):
            lineText=_mainScriptEditorWindow.text(lineIdx)
            lineTextStr=str(lineText)
            mainLoopRegexFound=re.match(mainLoopRegex,lineText)
            if mainLoopRegexFound:
                print "Indentation for mainLoop line is: ", _mainScriptEditorWindow.indentation(lineIdx), " indentation width=",_mainScriptEditorWindow.indentationWidth()
                indentationLevel=_mainScriptEditorWindow.indentation(lineIdx)/_mainScriptEditorWindow.indentationWidth()
                if _mainScriptEditorWindow.indentation(lineIdx) % _mainScriptEditorWindow.indentationWidth():
                    indentationLevel=-1 # problems with indentation will used indentation 0 and informa user about the issue
                return lineIdx,indentationLevel
                
        return  -1  
            # if lineText.startswith('CompuCellSetup.mainLoop'):
                # return lineIdx
        
    
    def showProjectPanel(self,_flag):
        """
            THIS SLOT WILL BE CALLED MULTIPLE TIMES AS IT IS LINKED TO TWO DIFFERENT SIGNALS - THIS IS NOT A PROBLEM IN THIS PARTICULAR CASE THOUGH
        """
        print "showProjectPanel CALLED ",_flag
        if _flag:
            self.cc3dProjectDock.show()
        else:    
            self.cc3dProjectDock.hide()
                
        if self.actions["Show Project Panel"].isChecked()!=_flag:
            self.actions["Show Project Panel"].setChecked(_flag)
         
        
    def __runInPlayer(self):
        tw=self.treeWidget
        
        curItem=tw.currentItem()
        projItem=tw.getProjectParent(curItem)
                
        
        if not projItem:
            numberOfprojects=self.treeWidget.topLevelItemCount()
            if numberOfprojects==1:
                projItem=self.treeWidget.topLevelItem(0)
            elif numberOfprojects>1:
                QMessageBox.warning(self.treewidget,"Please Select Project","Please first click inside project that you wish to open in the PLayer and try again")
            else:    
                return 
            
        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return        
    
        projectFullPath=pdh.cc3dSimulationData.path

        print "projectFullPath=",projectFullPath
        # get CompuCell3D Twedit Plugin - it allows to start CC3D from twedit
        cc3dPlugin=self.__ui.pm.getActivePlugin("PluginCompuCell3D")
        if not cc3dPlugin:
            return
            
        cc3dPlugin.startCC3D(projectFullPath)
        
    def __newCC3DProject(self):
        tw=self.treeWidget
        
        from CC3DProject.NewSimulationWizard import NewSimulationWizard
        
        nsw=NewSimulationWizard(tw)
        if nsw.exec_():
            print "New Simulation"
            nsw.generateNewProject()
            
        else:
            print "New Simulation Abbandoned"
            
            
            
    def __displayProperties(self):
        tw=self.treeWidget
        
        projItem=tw.getProjectParent(tw.currentItem())   
        if not projItem:
            return 

        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return          
            
        ild=None
        try:
            ild = tw.projects[projItem]
        except LookupError,e:
            return   
            
        resource=None
        try:    
            resource=ild.itemToResource[tw.currentItem()]
        except LookupError,e:
            return
            
        if not resource:
            return
            
        from CC3DProject.ItemProperties import ItemProperties
        import os.path
        
        print "resource=",resource
        
        ip=ItemProperties(self.treeWidget)        
        ip.setResourceReference(resource)
           
        ip.updateUi()
        
        if ip.exec_():
            print "Changes were made"
            dirtyFlagLocal=False
            if resource.module != str(ip.moduleLE.text()):
                dirtyFlagLocal=True
            if resource.origin != str(ip.originLE.text()):
                dirtyFlagLocal=True
            if resource.copy != ip.copyCHB.isChecked():
                dirtyFlagLocal=True
                
            resource.module=str(ip.moduleLE.text())
            resource.origin=str(ip.originLE.text())
            resource.copy=ip.copyCHB.isChecked()
            
            print "resource=",resource
            print "copy=",resource.copy
            
            #set dirtyFlag to True        
            try:                
                self.treeWidget.projects[projItem].dirtyFlag=dirtyFlagLocal
            except LookupError,e:
                pass            
            
        else:
            print "No Changes were made"
            
            
        

        
    def __openXMLPythonInEditor(self):
        tw=self.treeWidget
        projItem=tw.getProjectParent(tw.currentItem())   
        
        if not projItem:
            return 

        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return        
    
        print "__openXMLPythonInEditor pdh.cc3dSimulationData.xmlScript=",pdh.cc3dSimulationData.xmlScript
        print "__openXMLPythonInEditor pdh.cc3dSimulationData.xmlScriptResource.path=",pdh.cc3dSimulationData.xmlScriptResource.path
        
        # in order to do deeper level expansion we first have to expand top level
        projItem.setExpanded(True)        
        
        if pdh.cc3dSimulationData.xmlScript!="":
            self.openFileInEditor(pdh.cc3dSimulationData.xmlScript)
            xmlItem=tw.getItemByText(projItem,"XML Script")
            if xmlItem:
                xmlItem.setExpanded(True)
        if pdh.cc3dSimulationData.pythonScript!="":
            self.openFileInEditor(pdh.cc3dSimulationData.pythonScript)
            pythonItem=tw.getItemByText(projItem,"Main Python Script")
            if pythonItem:
                pythonItem.setExpanded(True)
            
        for path,resource in pdh.cc3dSimulationData.resources.iteritems():
            if resource.type=="Python":
                self.openFileInEditor(path)
                pythonItem=tw.getItemByText(projItem,"Python")
                if pythonItem:
                    pythonItem.setExpanded(True)
                
        
        
        return
        
    
    def openFileInEditor(self,_fileName=""):
        if _fileName=="":
            return
            
        tw=self.treeWidget        
        projItem=tw.getProjectParent(tw.currentItem())               
        if not projItem:
            return 
        
        
        
        ild=None
        try:
            ild = tw.projects[projItem]
        except LookupError,e:
            pass        
            
        if _fileName!="":
            # we will check if current tab before and after opening new document are the same (meaning an attampt to open same document twice)
            currentTabWidgetBefore=self.__ui.getCurrentEditor()
            self.__ui.loadFile(_fileName)
            currentTabWidgetAfter=self.__ui.getCurrentEditor()
            
            currentDocumentName = self.__ui.getCurrentDocumentName()
            currentTabWidget = self.__ui.getCurrentEditor()
            
            # check if opening of document was successful
            if currentDocumentName==_fileName:
                #next we check if _fileName is already present in self.projectLinkedTweditTabs as a value and linked to tab different than currentTabWidget
                # this happens when user opens _fileName from project widget, renames it in Twedit and then attempts to open _fileName again from project widget
                if ild:
                    tabReferencesToRemove=[]
                    for tabWidget,path in ild.projectLinkedTweditTabs.iteritems():
                        if path==_fileName and tabWidget != currentTabWidget:
                            tabReferencesToRemove.append(tabWidget)
                            
                            
                    for tab in tabReferencesToRemove:
                    
                        try :
                            del ild.projectLinkedTweditTabs[tab]
                        except LookupError,e:
                            pass
                                                
                    # insert current tab and associate it with _fileName - 
                    # if projectLinkedTweditTabs[currentTabWidget] is already present we will next statement is ignored - at most it changes value projectLinkedTweditTabs[currentTabWidget] 
                    ild.projectLinkedTweditTabs[currentTabWidget]=_fileName
        
    def __openInEditor(self):
    
        tw=self.treeWidget
        projItem=tw.getProjectParent(tw.currentItem())   
        if not projItem:
            return 

        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return        
    
        # fileName=pdh.cc3dSimulationData.path
        
        
        idl=None
        try:
            idl = tw.projects[projItem]
        except LookupError,e:
            pass
            
        fileName=tw.getFullPath(tw.currentItem())        
        
        if fileName!="":
            self.openFileInEditor(fileName)

            
    def closeProjectUsingProjItem(self,_projItem=None):
                
        if not _projItem:
            return 
            
        tw=self.treeWidget
        
        pdh=None
        try:
           pdh = self.projectDataHandlers[_projItem]
        except LookupError,e:
        
            return        
    
        fileName=pdh.cc3dSimulationData.path
        
        #check if project is dirty
        dirtyFlag=False
        try:
            dirtyFlag = self.treeWidget.projects[_projItem].dirtyFlag
        except LookupError,e:
            pass
            
        if dirtyFlag:
            ret=QMessageBox.warning(self.treeWidget,"Save Project Changes?","Project was modified.<br>Do you want to save changes?",QMessageBox.Yes|QMessageBox.No)
            if ret==QMessageBox.Yes:
                self.__saveCC3DProject()
        
        # ask users if they want to save unsaved documents associated with the project
        
        # close tabs associated with the project
        idl=None
        try:
            idl = tw.projects[_projItem]
        except LookupError,e:
            pass
        
        for tab in idl.projectLinkedTweditTabs.keys():
            index=tab.panel.indexOf(tab)
            self.__ui.closeTab(index,True,tab.panel)
            # self.__ui.closeTab(index)
        # remove self.treeWidget.projects[_projItem],self.treeWidget.projects[fileName] and self.projectDataHandlers[_projItem]from dictionaries
        
        try:
        
            del self.projectDataHandlers[_projItem]
            del self.treeWidget.projects[_projItem]
            del self.treeWidget.projects[fileName]
            
        except LookupError,e:
        
            pass
        
        for i in range(tw.topLevelItemCount()):
            if tw.topLevelItem(i)==_projItem:
                
                tw.takeTopLevelItem(i)
                break
                
        # tw.removeChild(projItem)
        
    def __closeProject(self):
        tw=self.treeWidget
        
        
        
        
        projItem=tw.getProjectParent(tw.currentItem())   
        
        pathToRemove=""
        for path, projItemLocal in self.openProjectsDict.iteritems():
            if projItemLocal==projItem:
                pathToRemove=path
                
        try:
            del self.openProjectsDict[pathToRemove]
        except LookupError,e:
            pass
            
        self.closeProjectUsingProjItem(projItem)
        
    
        
    def markProjectDirty(self,projItem):
        try:
            self.treeWidget.projects[projItem].dirtyFlag=True
        except LookupError,e:
            pass 
            
    def __removeResources(self):
    
        tw=self.treeWidget
        projItem=tw.getProjectParent(tw.currentItem())            
        if not projItem:
            return 
        
        ret=QMessageBox.warning(tw,"Delete Selected Items?","Are you sure you want to delete selected items?<br>This cannot be undone.<br> Proceed?",QMessageBox.Yes|QMessageBox.No)
        if ret==QMessageBox.No:
            return 
        
        
        ild=None
        
        
        try:
            ild=self.treeWidget.projects[projItem]                
        except:
            print "COULD NOT FIND PROJECT DATA"
            return
            
        pdh=None
        try:
            pdh=self.projectDataHandlers[projItem]        
            
        except LookupError,e:
            print "could not find simulation data handler for this item"
            return               
            
        selection=self.treeWidget.selectedItems()
        # divide the selection into type level items (e.g. items like Main Python Script, Python ,PIF File etc.) and leaf items (i.e files)
        typeItems=[]
        leafItems=[]
        
        for item in selection:
            if projItem==item.parent():
                typeItems.append(item)
            elif item != projItem:
                leafItems.append(item)        
                
                
        # first process leaf items - remove them from the project
        print "typeItems=",typeItems
        
        for item in leafItems:
            parent=item.parent()

            pdh.cc3dSimulationData.removeResource(ild.getFullPath(item))            


            if ild.getResourceName(item)=='CC3DSerializerResource':
                pdh.cc3dSimulationData.removeSerializerResource()                
                
            ild.removeItem(item)
            
            parent.removeChild(item)            
            if not parent.childCount() and parent not in typeItems:                
                ild.removeItem(parent)
                projItem.removeChild(parent)
                
        # process typeItems
        for item in typeItems:
            childrenList=[]
            for i in range(item.childCount()):
                
                childItem=item.child(i)
                childrenList.append(item.child(i))
            for childItem in childrenList:

                pdh.cc3dSimulationData.removeResource(ild.getFullPath(item))            

                
                # pdh.cc3dSimulationData.removeResource(ild.getFullPath(childItem))
                ild.removeItem(childItem)                
                item.removeChild(childItem)

            if ild.getResourceName(item)=='CC3DSerializerResource':
                pdh.cc3dSimulationData.removeSerializerResource()                
                
            elif ild.getResourceName(item)=='CC3DParameterScanResource':
                pdh.cc3dSimulationData.removeParameterScanResource()                
                
            ild.removeItem(item)
            
            projItem.removeChild(item)
            
        #mark project as dirty
        self.markProjectDirty(projItem)       
        
    def checkFileExtension(self,_extension="",_expectedExtensions=[]):
        if not len(_expectedExtensions):
            return ""
            
        
        if _extension in _expectedExtensions:
            return ""
        else:
            return _expectedExtensions[0] 
                
        
        
    def __addResource(self):
        from CC3DProject.NewFileWizard import NewFileWizard
        import os.path
        wz=NewFileWizard(self.treeWidget)        
        if wz.exec_():
            name=wz.nameLE.text()
            
            name=str(name)
            name=string.rstrip(name)
            # dont allow empty file names
            if name=="":
                return
            fileName=os.path.basename(name)
            
            base,extension=os.path.splitext(fileName)
            
            location=str(wz.locationLE.text())
            fileType=""
            if wz.customTypeCHB.isChecked():
                fileType=str(wz.customTypeLE.text())
            else:
                # have to replace it with dictionary 
                fileType=str(wz.fileTypeCB.currentText())
                if fileType=="Main Python Script":
                    fileType="PythonScript"
                elif  fileType=="XML Script":
                    fileType="XMLScript"
                elif  fileType=="PIF File":   
                    fileType="PIFFile"
                elif  fileType=="Python File" :  
                    fileType="Python"                    
                elif  fileType=="Concentration File" :  
                    fileType="ScalarField"

                # check file extensions
                if fileType=="Python" or fileType=="PythonScript":
                    if extension=="":
                        name=name+'.py'                        
                    else:
                        suggestedExtension=self.checkFileExtension(extension,['.py','.pyw'])    
                        if suggestedExtension!="":
                            ret=QMessageBox.warning(self.treeWidget,"Possible Extension Mismatch","Python script typically has extension <b>.py</b> .<br> Your file has extension <b>%s</b> . <br> Do you want to continue?" % extension , QMessageBox.Yes|QMessageBox.No)
                            if ret==QMessageBox.No:
                                return

                if fileType=="XMLScript" :
                    if extension=="":
                        name=name+'.xml'                        
                    else:
                        suggestedExtension=self.checkFileExtension(extension,['.xml'])    
                        if suggestedExtension!="":
                            ret=QMessageBox.warning(self.treeWidget,"Possible Extension Mismatch","XML script typically has extension <b>.xml</b> .<br> Your file has extension <b>%s</b> . <br> Do you want to continue?" % extension , QMessageBox.Yes|QMessageBox.No)
                            if ret==QMessageBox.No:
                                return
                
                if fileType=="PIFFile" :
                    if extension=="":
                        name=name+'.piff'                        
                    else:
                        suggestedExtension=self.checkFileExtension(extension,['.piff'])    
                        if suggestedExtension!="":
                            ret=QMessageBox.warning(self.treeWidget,"Possible Extension Mismatch","PIF File typically has extension <b>.piff</b> .<br> Your file has extension <b>%s</b> . <br> Do you want to continue?" % extension , QMessageBox.Yes|QMessageBox.No)
                            if ret==QMessageBox.No:
                                return
                    
            # extract project data handler
            tw=self.treeWidget
            projItem=tw.getProjectParent(tw.currentItem())            
            
            if not projItem:
                return 
            
            pdh=None
            try:
                pdh=self.projectDataHandlers[projItem]        
                
            except LookupError,e:
                print "could not find simulation data handler for this item"
                return              
                    
            # first check if location has not changed - this is a relative path w.r.t root of the simulation        
            fullLocation=""
            
            if location=="" or location =="Simulation" or location=="Simulation/":
                location="Simulation"
                fullLocation=os.path.join(pdh.cc3dSimulationData.basePath,"Simulation")
            else:

                try:
                    fullLocation=os.path.join(pdh.cc3dSimulationData.basePath,str(location))
                    self.makeDirectory(fullLocation)                    
                except IOError,e:
                    print "COULD NOT MAKE DIRECTORY ",pdh.cc3dSimulationData.basePath
                    QMessageBox.warning(self,"COULD NOT MAKE DIRECTORY","Write permission error. You do not have write permissions to %s directory" %(pdh.cc3dSimulationData.basePath),QMessageBox.Ok)
                    return
            # check if a file exists in which case we have to copy it to current directory
            name=str(name)
            resourceName=""
            # print "name=",name
            # print "fullLocation=",fullLocation
            
            try:
                
                open(name)
                # if file exist we will copy it to the 'fullLocation' directory     
                import shutil
                fileName=os.path.basename(name)
#                 print 'name=',name
#                 print 'fullLocation=',fullLocation
#                 print 'fileName=',fileName
                resourceName=os.path.join(fullLocation,fileName)
                try:
                
                    shutil.copy(name,resourceName)
                except shutil.Error, e:
                    QMessageBox.warning(self.__ui,"COULD NOT COPY FILE","Could not copy %s to %s . " % (name,fullLocation),QMessageBox.Ok)
                    pass # ignore any copy errors
                
            except IOError,e:
                # file does not exist 
                try:
                    from distutils.file_util import write_file
                    resourceName=os.path.join(fullLocation,name)
                    write_file(os.path.join(fullLocation,name),"")
                except IOError,e:
                    print "COULD NOT CREATE FILE"
                    QMessageBox.warning(self.__ui,"COULD NOT CREATE FILE","Write permission error. You do not have write permissions to %s directory" % (fullLocation),QMessageBox.Ok)
                    return
            # Those 2 fcn calls have to be paired        
            #attach new file to the project
            pdh.cc3dSimulationData.addNewResource(resourceName,fileType)  
            #insert new file into the tree
            self.insertNewTreeItem(resourceName,fileType)
            
#             # instead of marking it as dirty we simply save it
#             self.saveCC3DProject(projItem)
            #mark project as dirty
            self.markProjectDirty(projItem)       

            
        return
        
    def __addSerializerResource(self):

        tw=self.treeWidget
        projItem=tw.getProjectParent(tw.currentItem())            

        pdh=None
        try:
            pdh=self.projectDataHandlers[projItem]                    
        except LookupError,e:
            print "could not find simulation data handler for this item"
            return              
        
        if pdh.cc3dSimulationData.serializerResource:
            QMessageBox.warning(tw,"Serializer is already defined","You cannot have more than one serializer per simulation")
            return
        
        

        from CC3DProject.SerializerEdit import SerializerEdit
        se=SerializerEdit(self.treeWidget)
        
        resource=self.treeWidget.getCurrentResource()
        print 'resource=',resource
        # se.setupDialog(resource)
        if se.exec_():
       
            pdh.cc3dSimulationData.addNewSerializerResource()# adding empty serializer resource
            
            se.modifySerializerResource(pdh.cc3dSimulationData.serializerResource)
            projItem=self.treeWidget.getProjectParent(self.treeWidget.currentItem())       
            self.markProjectDirty(projItem)

            #insert new file into the tree
            self.insertNewGenericResourceTreeItem(pdh.cc3dSimulationData.serializerResource)
        
        
    def insertNewGenericResourceTreeItem(self,_resource):
        
        projItem=self.treeWidget.getProjectParent(self.treeWidget.currentItem())
        if not projItem:
            return 
        
        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return        
        
        pd=pdh.cc3dSimulationData # project data
        
        ild=None
        
        try:
            ild=self.treeWidget.projects[projItem]                
        except:
            print "COULD NOT FIND PROJECT DATA"
            return
            
        if _resource.resourceName=='CC3DSerializerResource':
        
        
            item=QTreeWidgetItem(projItem)
            item.setText(0,"Serializer")
            item.setIcon(0,QIcon(':/icons/save-simulation.png'))
            
            try:
                
                ild.insertnewGenericResource(item,_resource)
            except LookupError,e:
                # print "pd.resources[resourceName]=",pd.resources[resourceName]
                pass
        
        elif _resource.resourceName=='CC3DParameterScanResource':
        
            # make new branch to store this item 
            item=QTreeWidgetItem(projItem)
            item.setText(0,"ParameterScan")
            item.setIcon(0,QIcon(':/icons/scan_32x32.png'))            
            
            item1=QTreeWidgetItem(item)
            item1.setText(0,os.path.basename(_resource.path))            

            
            try:
                ild.insertnewGenericResource(item,_resource)
                ild.insertNewItem(item1,_resource) # file path has to be entered into ild using insertNewItem to enable proper behavior of tree widget                
                
            except LookupError,e:
                # print "pd.resources[resourceName]=",pd.resources[resourceName]
                pass
        
        
    def insertNewTreeItem(self,resourceName,fileType):
        #first find the node where to insert new item
        projItem=self.treeWidget.getProjectParent(self.treeWidget.currentItem())
        if not projItem:
            return 
        
        fileNameBase=os.path.basename(resourceName)
        print "resourceName=",resourceName
        print "fileType=",fileType
        
        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return        
        
        pd=pdh.cc3dSimulationData # project data
        
        ild=None
        
        try:
            ild=self.treeWidget.projects[projItem]                
        except:
            print "COULD NOT FIND PROJECT DATA"
            return
        
        
        
        if fileType=="PythonScript":
            typeItem=self.findTypeItemByName("Main Python Script")
            
            # we will replace Python script with new one
            if typeItem:
                
                
                ild.removeItem(typeItem.child(0))
                typeItem.removeChild(typeItem.child(0))
                pythonScriptItem=QTreeWidgetItem(typeItem)
                pythonScriptItem.setText(0,fileNameBase)
                
                ild.insertNewItem(pythonScriptItem,pd.pythonScriptResource)
                
            else: # make new branch to store this item 
                pythonScriptItem=QTreeWidgetItem(projItem)
                pythonScriptItem.setText(0,"Main Python Script")

                pythonScriptItem.setIcon(0,QIcon(':/icons/python-icon.png'))

                pythonScriptItem1=QTreeWidgetItem(pythonScriptItem)
                pythonScriptItem1.setText(0,fileNameBase)
                ild.insertNewItem(pythonScriptItem1,pd.pythonScriptResource)
                
                
        elif fileType=="XMLScript":
            typeItem=self.findTypeItemByName("XML Script")
            # we will replace XML script with new one
            if typeItem:
                
        
                ild.removeItem(typeItem.child(0))
                typeItem.removeChild(typeItem.child(0))
                xmlScriptItem=QTreeWidgetItem(typeItem)
                xmlScriptItem.setText(0,fileNameBase)
                ild.insertNewItem(xmlScriptItem,pd.xmlScriptResource)
                
            else: # make new branch to store this item 
                xmlScriptItem=QTreeWidgetItem(projItem)
                xmlScriptItem.setText(0,"XML Script")
                xmlScriptItem.setIcon(0,QIcon(':/icons/xml-icon.png'))
                xmlScriptItem1=QTreeWidgetItem(xmlScriptItem)
                xmlScriptItem1.setText(0,fileNameBase)
                ild.insertNewItem(xmlScriptItem1,pd.xmlScriptResource)
                
        elif fileType=="PIFFile":
            typeItem=self.findTypeItemByName("PIF File")
            
            # we will do not replace PIF File with new one - just add another one
            
            if typeItem:
                #check if new path  exists in this branch
                for i in range(typeItem.childCount()):
                    if str(ild.getFullPath(typeItem.child(i)))==str(resourceName):
                        return
            
                pifFileItem=QTreeWidgetItem(typeItem)
                pifFileItem.setText(0,fileNameBase)
                #check if full path exist in this branch
                
                        
                
                try:
                    ild.insertNewItem(pifFileItem,pd.resources[resourceName])
                except LookupError,e:
                    pass
                
            else: # make new branch to store this item 
                pifFileItem=QTreeWidgetItem(projItem)
                pifFileItem.setText(0,"PIF File")
                pifFileItem.setIcon(0,QIcon(':/icons/pifgen_64x64.png'))
                
                pifFileItem1=QTreeWidgetItem(pifFileItem)
                pifFileItem1.setText(0,fileNameBase)
                print "PIF FILE RESOURCE=",os.path.abspath(resourceName)
                try:
                    ild.insertNewItem(pifFileItem1,pd.resources[os.path.abspath(resourceName)])
                except LookupError,e:
                    # print "pd.resources[resourceName]=",pd.resources[resourceName]
                    pass
                    
        else:
            typeItem=self.findTypeItemByName(fileType)
            if typeItem:
                #check if new path  exists in this branch
                for i in range(typeItem.childCount()):
                    if str(ild.getFullPath(typeItem.child(i)))==str(resourceName):
                        return
                        
                item=QTreeWidgetItem(typeItem)
                item.setText(0,fileNameBase)
                
                try:
                    ild.insertNewItem(item,pd.resources[resourceName])
                except LookupError,e:
                    pass
                
            else: # make new branch to store this item 
                item=QTreeWidgetItem(projItem)
                item.setText(0,fileType)
                
                item1=QTreeWidgetItem(item)
                item1.setText(0,fileNameBase)
                print "PIF FILE RESOURCE=",os.path.abspath(resourceName)
                try:
                    ild.insertNewItem(item1,pd.resources[os.path.abspath(resourceName)])
                except LookupError,e:
                    # print "pd.resources[resourceName]=",pd.resources[resourceName]
                    pass
            
    
    def makeDirectory(self,fullDirPath):        
        """
            This fcn attmpts to make directory or if directory exists it will do nothing
        """
        from distutils.dir_util  import mkpath
        # dirName=os.path.dirname(fullDirPath)
        try:
            mkpath(fullDirPath)
        except:
            raise IOError
            
        return
        
    def __saveCC3DProject(self):
        curItem=self.treeWidget.currentItem()
        projItem=self.treeWidget.getProjectParent(curItem)
        if not projItem:
            return 

        pdh=None
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return        
            
        fileName=pdh.cc3dSimulationData.path
#         print "ORIGINAL PROJECT FILE NAME=",fileName
        
        # fileName="D:/Program Files/COMPUCELL3D_3.5.1_install2/examples_PythonTutorial/infoPrinterDemo/infoPrinterDemo_new.cc3d"
        # first determine project to be saved based on current element
                
        
        pdh.writeCC3DFileFormat(fileName)
        
        
        #set dirtyFlag to False        
        try:
            self.treeWidget.projects[projItem].dirtyFlag=False
        except LookupError,e:
            pass
            
   
        return
        
    def __saveCC3DProjectAs(self):
    
        tw=self.treeWidget
        currentProjectDir=os.path.dirname(str(self.configuration.setting("RecentProject")))
        
        currentProjectDir=os.path.dirname(currentProjectDir) # going one level up to open dialog in the correct location
#         print 'currentProjectDir =',currentProjectDir 
        
#         fileName=QFileDialog.getSaveFileName(self.__ui,"Save CC3D Project...",currentProjectDir,"*.cc3d")
        projectDirName=QFileDialog.getSaveFileName(self.__ui,"Save CC3D Project...",currentProjectDir,'','',QFileDialog.DontConfirmOverwrite)
#         print "SAVE AS FILE NAME=",projectDirName
        
                
        projectDirName=str(projectDirName)
        
        # projectDirName can have extension because we are using getSaveFile, so we get rid of extension here        
        projectDirName,extension = os.path.splitext(projectDirName)
#         print 'projectDirName,extension=',(projectDirName,extension)
        
        projectCoreName = os.path.basename(projectDirName)
#         print 'projectCoreName =',projectCoreName 

        if str(projectDirName)=="":
            return

        curItem=self.treeWidget.currentItem()
        projItem=self.treeWidget.getProjectParent(curItem)

        
        if not projItem:
            numberOfprojects=self.treeWidget.topLevelItemCount()
            if numberOfprojects==1:
                projItem=self.treeWidget.topLevelItem(0)
            elif numberOfprojects>1:
                QMessageBox.warning(self.treewidget,"Please Select Project","Please first click inside project that you wish to save and try again")
            else:    
                return 

        pdh=None
        
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:
        
            return     

        # note, a lot of the code here was written assuming we will not need to reopen the project
        # however it turns out that proper handling of such situation woudl require more fefactoring so I decided to 
        # close new project and reopen it (users will need to open their files) 
        # Once we refactor this plugin we can always add proper handling without requiring project to be reopened
        csd=pdh.cc3dSimulationData
#         print 'csd.xmlScriptResource.path=',csd.xmlScriptResource.path
        
        fn2ew = self.__ui.getFileNameToEditorWidgetMap()


        if os.path.exists(projectDirName):
            ret=QMessageBox.warning(tw,"Directory or File %s Already Exists"%(projectCoreName) ,"Please choose different name for the project. Directory or file %s already exists"%(os.path.join(projectDirName,projectCoreName)),QMessageBox.Ok)    
            return
        else:
#             print 'will make new directory ', projectDirName
            os.makedirs(projectDirName)
            os.makedirs(os.path.join(projectDirName,'Simulation'))

        coreCsdResourceNames=['xmlScriptResource','pythonScriptResource','pifFileResource','windowScriptResource','serializerResource','parameterScanResource']
        resourceList = [getattr(csd,resourceName) for resourceName in coreCsdResourceNames]    
        
        for resourceKey, resource in csd.resources.iteritems():        
            resourceList.append(resource)
            

        
        for resource in resourceList:
            
            if not resource:continue
            if resource.path.strip() == '' :continue
            
            resourceBaseName = os.path.basename(resource.path)
            newResourcePath = os.path.join(os.path.abspath(projectDirName),'Simulation',resourceBaseName)     
#             print 'resource=',resource
            if resource.path in fn2ew.keys():
                # this means the editor with the resource is open
                
                editor = fn2ew [resource.path]                                
                self.__ui.saveFile(_fileName = newResourcePath , _editor=editor) #if project resource is open we save it in the new location                
            else:
                # this means the editor with the resource is not open so we simply copy files
                import shutil                
# # #                 print 'copy ',resource.path, ' to ',newResourcePath
                shutil.copy(resource.path,newResourcePath)
                
            oldResourcePath=resource.path
            
# # #             print 'changing resource.path=',resource.path,' to ',newResourcePath
            # resource.path = newResourcePath
            resource.path = 'Simulation/' + resourceBaseName

                
            try:
                
                del pdh.cc3dSimulationData.resources[oldResourcePath]
                pdh.cc3dSimulationData.resources[resource.path]=resource
                
            except LookupError:
                print 'could not find ', oldResourcePath, ' in pdh.cc3dSimulationData.resources'     
                pass
#             print 'pdh.cc3dSimulationData.resources=',pdh.cc3dSimulationData.resources    
            
            
            
        cc3dProjectFileName=os.path.join(projectDirName,projectCoreName+'.cc3d')    
        pdh.writeCC3DFileFormat(cc3dProjectFileName)            
        
        #after the project has been saved we need to update path to .cc3d fine and basePath
        csd.path=cc3dProjectFileName
        csd.basePath=os.path.dirname(csd.path)
        
#         self.__closeProject()
        self.closeProjectUsingProjItem(projItem)
        self.openCC3Dproject(cc3dProjectFileName)
        

#         print 'AFTER CHANGES csd.xmlScriptResource.path=',csd.xmlScriptResource.path
#         print 'AFTER CHANGES csd.xmlScriptResource.path=',pdh.cc3dSimulationData.xmlScriptResource.path
#         print 'AFTER CHANGES csd.xmlScript=',pdh.cc3dSimulationData.xmlScript
        
        return

    
        
    
    def openCC3Dproject(self,fileName):        
        projExist=True
        
        self.__ui.addItemtoConfigurationStringList(self.configuration,"RecentProjects",fileName)  
        
        #extract file directory name and add it to settings                    
        dirName=os.path.abspath(os.path.dirname(str(fileName)))
        self.__ui.addItemtoConfigurationStringList(self.configuration,"RecentProjectDirectories",dirName)                        
        
        try:
            
            self.openProjectsDict[fileName]
            
        except LookupError,e:
            projExist=False
        
        if  projExist:
            projItem=self.openProjectsDict[fileName]
            self.treeWidget.setCurrentItem(projItem)
            return 
            
        # from CC3DProject.CC3DSimulationDataHandler import CC3DSimulationDataHandler
        
        from CC3DSimulationDataHandler import CC3DSimulationDataHandler
        projItem=QTreeWidgetItem(self.treeWidget)
        projItem.setIcon(0,QIcon(':/icons/cc3d_64x64_logo.png'))
        
        
        #store a reference to data handler in a dictionary
        self.projectDataHandlers[projItem]=CC3DSimulationDataHandler(None)
        
        self.projectDataHandlers[projItem].readCC3DFileFormat(fileName)
        # we read manually the content of the parameter spec file
        if self.projectDataHandlers[projItem].cc3dSimulationData.parameterScanResource:
            self.projectDataHandlers[projItem].cc3dSimulationData.parameterScanResource.readParameterScanSpecs()
        
        
        self.__populateCC3DProjectWidget(projItem,fileName)
        
        self.configuration.setSetting("RecentProject",fileName)
        
        self.openProjectsDict[fileName]=projItem
    
    def showOpenProjectDialogAndLoad(self,_dir=''):
        fileName=QFileDialog.getOpenFileName(self.__ui,"Open CC3D file...",_dir,"*.cc3d")
        print "FILE NAME=",fileName
        if str(fileName)=="":
            return
            
        fileName=os.path.abspath(str(fileName)) # normalizing filename
                
        self.openCC3Dproject(fileName)
    
    def __openCC3DProject(self):
        

        
        currentFilePath=os.path.dirname(str(self.configuration.setting("RecentProject")))
        self.showOpenProjectDialogAndLoad(currentFilePath)
        
        # fileName=QFileDialog.getOpenFileName(self.__ui,"Open CC3D file...",currentFilePath,"*.cc3d")
        # print "FILE NAME=",fileName
        # if str(fileName)=="":
            # return
            
        # fileName=os.path.abspath(str(fileName)) # normalizing filename
        
        # # fileName=""
        
        # # for i in range(fileNames.count()):
            # # print"THIS IS FILE= ",  os.path.abspath(str(fileNames[i]))# "normalizing" file name to make sure \ and / are used in a consistent manner
            # # fileName=os.path.abspath(str(fileNames[i]))
        
        # # self.cc3dProjectHandler=CC3DSimulationDataHandler(None)
        
        
        
        # # # # fileName="D:/Program Files/COMPUCELL3D_3.5.1_install2/examples_PythonTutorial/infoPrinterDemo/infoPrinterDemo.cc3d"
        
        # self.openCC3Dproject(fileName)
        
        # # new project tree item - top level
        
    def findTypeItemByName(self,_typeName)   :
        projItem=self.treeWidget.getProjectParent(self.treeWidget.currentItem())
        if not projItem:
            return None
        
        for i in range(projItem.childCount()):
            childItem=projItem.child(i)
            print "childItem.text(0)=",childItem.text(0)," _typeName=",_typeName
            
            if str(_typeName)==str(childItem.text(0)):
                return childItem
    
        return None
        
    def __populateCC3DProjectWidget(self,projItem,fileName):    
    # def __populateCC3DProjectWidget(self,_treeWidget,fileName):    
        import os.path
        
        pdh=None        
        try:
           pdh = self.projectDataHandlers[projItem]
        except LookupError,e:        
            return
        
        
        # projItem=QTreeWidgetItem(_treeWidget)
        fileNameBase=os.path.basename(fileName)
        
        ild=ItemLookupData()
        self.treeWidget.projects[fileName] = ild
        self.treeWidget.projects[projItem] = ild
        
        #store a reference to data handler in a dictionary
        # self.projectDataHandlers[projItem]=self.cc3dProjectHandler
        
        projItem.setText(0,fileNameBase)
        
        pd=pdh.cc3dSimulationData # project data
        
        if pd.pythonScript!="":
            pythonScriptItem=QTreeWidgetItem(projItem)
            pythonScriptItem.setText(0,"Main Python Script")
            pythonScriptItem.setIcon(0,QIcon(':/icons/python-icon.png'))
            
            pythonScriptItem1=QTreeWidgetItem(pythonScriptItem)
            pythonScriptItem1.setText(0,os.path.basename(pd.pythonScript))
            # pythonScriptItem1.setIcon(0,QIcon(':/icons/python-icon.png'))
            ild.insertNewItem(pythonScriptItem1,pd.pythonScriptResource) 
            
            
        if pd.xmlScript!="":
            xmlScriptItem=QTreeWidgetItem(projItem)
            xmlScriptItem.setText(0,"XML Script")
            
            xmlScriptItem.setIcon(0,QIcon(':/icons/xml-icon.png'))
            
            xmlScriptItem1=QTreeWidgetItem(xmlScriptItem)
            xmlScriptItem1.setText(0,os.path.basename(pd.xmlScript))
            
            # xmlScriptItem1.setIcon(0,QIcon(':/icons/xml-icon.png'))
            
            ild.insertNewItem(xmlScriptItem1,pd.xmlScriptResource) 
            
        if pd.pifFile!="":
            pifFileItem=QTreeWidgetItem(projItem)
            pifFileItem.setText(0,"PIF File")
            pifFileItem.setIcon(0,QIcon(':/icons/pifgen_64x64.png'))
            
            pifFileItem1=QTreeWidgetItem(pifFileItem)
            pifFileItem1.setText(0,os.path.basename(pd.pifFile))            
            ild.insertNewItem(pifFileItem1,pd.pifFileResource) 
            
        resourcesItems={}    
        resourceTypes={}    
        # Resources
        
        for resourceKey, resource in pd.resources.iteritems():
            if resource.type in resourceTypes.keys():
                
                parentItem=resourceTypes[resource.type]
                newResourceItem=QTreeWidgetItem(parentItem)                
                newResourceItem.setText(0,os.path.basename(resource.path))
                
                ild.insertNewItem(newResourceItem,resource) 
            else:
                newResourceItem=QTreeWidgetItem(projItem)
                newResourceItem.setText(0,resource.type)
                if resource.type=="Python":
                    newResourceItem.setIcon(0,QIcon(':/icons/python-icon.png'))
                
                #inserting parent element for givent resporce type to dictionary
                resourceTypes[resource.type]=newResourceItem

                
                newResourceItem1=QTreeWidgetItem(newResourceItem)
                newResourceItem1.setText(0,os.path.basename(resource.path))
                    
                
                ild.insertNewItem(newResourceItem1,resource) 
                
                        
        # serialization data
        if pd.serializerResource:
            serializerItem=QTreeWidgetItem(projItem)
            serializerItem.setText(0,"Serializer")
            serializerItem.setIcon(0,QIcon(':/icons/save-simulation.png'))
            
            # outputFrequencyItem=QTreeWidgetItem(serializerItem)
            # outputFrequencyItem.setText(0,'Output Frequency')            
            # # outputFrequencyItem.setText(1,str(pd.serializerResource.outputFrequency))            
            
            # outputFrequencyItem1=QTreeWidgetItem(outputFrequencyItem)
            # outputFrequencyItem1.setText(0,str(pd.serializerResource.outputFrequency))            
            
            ild.insertnewGenericResource(serializerItem,pd.serializerResource)
            # ild.insertNewItem(pifFileItem1,pd.pifFileResource) 
            
        if pd.parameterScanResource:
            psResource=pd.parameterScanResource
            # make new branch to store this item 
            item=QTreeWidgetItem(projItem)
            item.setText(0,"ParameterScan")
            item.setIcon(0,QIcon(':/icons/scan_32x32.png'))            
            
            item1=QTreeWidgetItem(item)
            item1.setText(0,os.path.basename(psResource.path))            

            
            try:
                ild.insertnewGenericResource(item,psResource)
                ild.insertNewItem(item1,psResource) # file path has to be entered into ild using insertNewItem to enable proper behavior of tree widget                
                
            except LookupError,e:
                # print "pd.resources[resourceName]=",pd.resources[resourceName]
                pass        
  
                
                
            
            
    
