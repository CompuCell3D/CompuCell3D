# todo - fix context menu properties - old pyqt4 code is there. needs to be migrated
"""
    TO DO:
    * Keyboard events - Del
    * New Simulation wizard
    * resource properties display
    *
    * add version number to project
"""

"""

Module used to link Twedit++5 with CompuCell3D.

"""

# THIS HAS TO BE REVRITTEN USING MVC, otherwise ti is hard to maintain

import os.path
from pathlib import Path
from typing import Union
from zipfile import ZipFile
from glob import glob
import re
import os
import shutil
import platform
from copy import deepcopy
from distutils.file_util import write_file
from distutils.dir_util import mkpath
from cc3d.twedit5.Plugins.CC3DProject.NewSimulationWizard import NewSimulationWizard
from cc3d.twedit5.Plugins.TweditPluginBase import TweditPluginBase
from cc3d.twedit5.Plugins.CC3DProject.SerializerEdit import SerializerEdit
from cc3d.twedit5.Plugins.CC3DProject.SteppableGeneratorDialog import SteppableGeneratorDialog
from cc3d.core.CC3DSimulationDataHandler import CC3DSimulationDataHandler
from cc3d.twedit5.Plugins.CC3DProject.XmlAccessPathDialog import XmlAccessPathDialog
from cc3d.twedit5.Plugins.CC3DProject.SteppableTemplates import SteppableTemplates
from cc3d.twedit5.Plugins.CC3DProject.ParValDlg import ParValDlg
from cc3d.twedit5.Plugins.CC3DProject.SerializerEdit import SerializerEdit
from cc3d.twedit5.Plugins.CC3DProject.NewFileWizard import NewFileWizard
from cc3d.core.ParameterScanUtils import XMLHandler
from cc3d.twedit5.Plugins.CC3DProject.ItemProperties import ItemProperties
from cc3d.core import XMLUtils
from cc3d.twedit5.twedit.utils.global_imports import *
from cc3d.twedit5.twedit.utils import qt_obj_hash
from cc3d.twedit5.Plugins.CC3DProject import CC3DProject_rc
from cc3d.twedit5.Plugins.CC3DProject.Configuration import Configuration
# from ParameterScanEnums import *
from cc3d.core.ParameterScanEnums import *
# from . import CC3DProject.CC3DPythonGenerator as cc3dPythonGen
import cc3d.twedit5.Plugins.CC3DProject.CC3DPythonGenerator as cc3dPythonGen
from cc3d.core.ParameterScanUtils import ParameterScanUtils
from cc3d.core.ParameterScanEnums import PYTHON_GLOBAL
from cc3d.gui_plugins.unzipper import Unzipper
import contextlib
from pathlib import Path

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

error = ''


# this is bidirectional dictionary - tree-item to CC3DResource and path of the resource to item

class ItemLookupData:

    def __init__(self):

        self.itemToResource = {}

        self.pathToItem = {}

        self.dirtyFlag = False

        self.itemToGenericResource = {}

        self.genericResourceToItem = {}

        # here we will store twedit tabs and associated file names that were opened from Project widget .

        # Later before closing we will ask users if they want to save documents in those tabs

        # if the tab does not exist or document changed name we will ignore such tab

        # to make sure that we don't store too many items before opening new document from project widget 

        # we will make sure that tab for previously opened tab are removed dictionary before reopening new one    

        self.projectLinkedTweditTabs = {}

    def insertnewGenericResource(self, _item, _resource):

        self.itemToGenericResource[qt_obj_hash(_item)] = _resource

        self.genericResourceToItem[_resource] = _item

    def insertNewItem(self, _item, _fullPath):

        self.itemToResource[qt_obj_hash(_item)] = _fullPath

        self.pathToItem[_fullPath] = _item

    def removeItem(self, _item):

        try:

            path = self.itemToResource[qt_obj_hash(_item)]

            del self.itemToResource[qt_obj_hash(_item)]

            del self.pathToItem[path]

        except:

            pass

        try:

            resource = self.itemToGenericResource[qt_obj_hash(_item)]

            del self.itemToGenericResource[qt_obj_hash(_item)]

            del self.genericResourceToItem[resource]

        except:

            pass

    def getResourceName(self, _item):

        try:

            return self.itemToGenericResource[qt_obj_hash(_item)].resourceName

        except LookupError as e:

            return ''

        except:

            return ''

    def getResource(self, _item):

        try:

            return self.itemToGenericResource[qt_obj_hash(_item)]

        except LookupError as e:

            return None

        except:

            return None

    def getFullPath(self, _item):

        try:

            return self.itemToResource[qt_obj_hash(_item)].path

        except LookupError as e:

            return ""


class CC3DProjectTreeWidget(QTreeWidget):

    def __init__(self, parent=None):

        QTreeWidget.__init__(self, parent)

        self.plugin = None

        self.__ui = None  # Twedit++ user interface    

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.setColumnCount(1)

        self.setItemsExpandable(True)

        self.setHeaderLabels(["CC3D Simulation"])

        self.projects = {}

        self.itemToProject = {}

        self.style = None  # np++ style - usually this is Global override style defined in themes xml file

        self.N2C = None  # convenience function reference from theme manager to convert npp color convention to QColor

        self.itemChanged.connect(self.__restyle)

        self.__iconDict = {}  # used to store icons for actions shown in the context menu - have to do this becaue of qt quirks on OSX

        self.hideContextMenuIcons = False

        mac_ver = platform.mac_ver()

        if mac_ver[0]:
            self.hideContextMenuIcons = True  # on OSX we hide context menu icons

    def setCC3DProjectPlugin(self, _plugin):

        """

            Set reference to CC3DProject plugin

        """

        self.plugin = _plugin

        self.__ui = self.plugin.getUI()

    # get super-parent for the item - this is project item (all items belonging to the projects are under this item)

    def getProjectParent(self, _item):

        if not _item:
            return _item

        curItem = _item

        parentItem = curItem.parent()

        while parentItem:
            curItem = parentItem

            parentItem = curItem.parent()

        return curItem

    def getFullPath(self, _item):

        # first determine the parent

        projParent = self.getProjectParent(_item)

        if not projParent:
            return ""

        print("projParent=", projParent.text(0))

        ild = self.projects[qt_obj_hash(projParent)]

        return ild.getFullPath(_item)

    def getResourceName(self, _item):

        # first determine the parent

        projParent = self.getProjectParent(_item)

        if not projParent:
            return ""

        # print "projParent=",projParent.text(0)

        ild = self.projects[qt_obj_hash(projParent)]

        return ild.getResourceName(_item)

    def getCurrentResource(self):

        return self.getResource(self.currentItem())

    def getResource(self, _item):

        # first determine the parent

        projParent = self.getProjectParent(_item)

        if not projParent:
            return ""

        # print "projParent=",projParent.text(0)

        ild = self.projects[qt_obj_hash(projParent)]

        return ild.getResource(_item)

    def getItemByText(self, _parentItem, _text=""):

        if not _parentItem:
            return None

        for i in range(_parentItem.childCount()):

            childItem = _parentItem.child(i)

            text = str(childItem.text(0))

            if text == str(_text):
                return childItem

        return None

    def mouseDoubleClickEvent(self, event):

        projItem = self.getProjectParent(self.currentItem())

        if not projItem:
            return

            # print 'self.getFullPath(self.currentItem()=',self.getFullPath(self.currentItem())

        if self.getFullPath(self.currentItem()) != "":

            self.plugin.actions["Open In Editor"].trigger()

        elif projItem == self.currentItem():

            self.plugin.actions["Open XML/Python In Editor"].trigger()

    def restoreIcons(self):

        for action, icon in self.__iconDict.items():
            action.setIcon(icon)

    def addActionToContextMenu(self, _menu, _action):

        if self.hideContextMenuIcons:
            self.__iconDict[_action] = _action.icon()

            _action.setIcon(QIcon())

        _menu.addAction(_action)

    def contextMenuEvent(self, event):

        self.__iconDict = {}  # resetting icon dictionary

        menu = QMenu(self)

        menu.aboutToHide.connect(self.restoreIcons)

        projItem = self.getProjectParent(self.currentItem())

        pdh = None

        try:

            pdh = self.plugin.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            print("could not find simulation data handler for this item")

            return

        if self.currentItem() == projItem:
            self.addActionToContextMenu(menu, self.plugin.actions["Open XML/Python In Editor"])

            self.addActionToContextMenu(menu, self.plugin.actions["Open in Player"])

            # --------------------------------------------------------------------

            menu.addSeparator()

            # if not pdh.cc3dSimulationData.serializerResource:
            #     self.addActionToContextMenu(menu, self.plugin.actions["Add Serializer..."])
            #
            #     # --------------------------------------------------------------------
            #
            #     menu.addSeparator()

        # menu.addAction(self.plugin.actions["Open CC3D Project..."])

        if self.getFullPath(self.currentItem()) != "":
            self.addActionToContextMenu(menu, self.plugin.actions["Open In Editor"])

            self.addActionToContextMenu(menu, self.plugin.actions["Properties"])

            # --------------------------------------------------------------------

            self.addGenerateSteppableMenu(menu, projItem)

            self.addConvertXMLToPythonMenu(menu, projItem)

            menu.addSeparator()

        resourceName = self.getResourceName(self.currentItem())

        # print('\n\n\n RESOURCENAME', resourceName)

        # if resourceName == 'CC3DSerializerResource':
        #     self.addActionToContextMenu(menu, self.plugin.actions["Serializer..."])

        # if resourceName=='CC3DParameterScanResource':

        # menu.addAction(self.plugin.actions["Reset Parameter Scan"])

        self.addActionToContextMenu(menu, self.plugin.actions["Save CC3D Project"])

        self.addActionToContextMenu(menu, self.plugin.actions["Go To Project Directory"])

        self.addActionToContextMenu(menu, self.plugin.actions["Zip It!"])

        # self.addActionToContextMenu(menu,self.plugin.actions["Zip'n'Mail"])

        # --------------------------------------------------------------------

        menu.addSeparator()

        # parameter scan menus

        self.addActionToContextMenu(menu, self.plugin.actions["Add Parameter Scan"])
        #
        # self.addParameterScanMenus(menu, projItem)

        # --------------------------------------------------------------------

        menu.addSeparator()

        self.addActionToContextMenu(menu, self.plugin.actions["Add Resource..."])

        # if selection.size():

        # menu.addAction(self.plugin.actions["Remove Resources"])

        self.addActionToContextMenu(menu, self.plugin.actions["Remove Resources"])

        print("CurrentItem=", self.currentItem().text(0), " parent=", self.currentItem().parent())

        print("getFullPath=", self.getFullPath(self.currentItem()))

        # if self.getFullPath(self.currentItem())!="":

        # #--------------------------------------------------------------------

        # menu.addSeparator()

        # menu.addAction(self.plugin.actions["Open In Editor"])

        # --------------------------------------------------------------------

        menu.addSeparator()

        self.addActionToContextMenu(menu, self.plugin.actions["Close Project"])

        # if self.currentItem().parent()==self:

        # print "GOT TOP LEVEL ITEM"

        menu.exec_(event.globalPos())

    def addGenerateSteppableMenu(self, _menu, _projItem):

        # print "TRYING TO ADD GENERATE STEPPEBLE MENU"

        pdh = None

        try:

            pdh = self.plugin.projectDataHandlers[qt_obj_hash(_projItem)]

        except LookupError as e:

            return

            # check if the file to which we are trying to add Steppable is Python resource

        itemFullPath = str(self.getFullPath(self.currentItem()))

        basename, extension = os.path.splitext(itemFullPath)

        basename = os.path.basename(itemFullPath)

        # print "basename=",basename," ext=",extension

        try:

            cc3dResource = pdh.cc3dSimulationData.resources[itemFullPath]

            if cc3dResource.type == "Python":
                self.addActionToContextMenu(_menu, self.plugin.actions["Add Steppable..."])



        except LookupError as e:

            return

    def addParameterScanMenus(self, _menu, _projItem):

        pdh = None

        try:

            pdh = self.plugin.projectDataHandlers[qt_obj_hash(_projItem)]

        except LookupError as e:

            return

        _menu.addSeparator()

        resourceName = self.getResourceName(self.currentItem())

        itemFullPath = str(self.getFullPath(self.currentItem()))

        basename, extension = os.path.splitext(itemFullPath)

        # adding menu to parameter scan xml file

        if pdh.cc3dSimulationData.parameterScanResource and itemFullPath == pdh.cc3dSimulationData.parameterScanResource.path:
            self.addActionToContextMenu(_menu, self.plugin.actions["Reset Parameter Scan"])

        # adding menu to parameter scan node

        if resourceName == 'CC3DParameterScanResource':
            self.addActionToContextMenu(_menu, self.plugin.actions["Reset Parameter Scan"])

        try:

            cc3dResource = pdh.cc3dSimulationData.resources[itemFullPath]

            if cc3dResource.type == "Python":
                self.addActionToContextMenu(_menu, self.plugin.actions["Open Scan Editor"])



        except LookupError as e:

            pass

        if pdh.cc3dSimulationData.xmlScript == itemFullPath or pdh.cc3dSimulationData.pythonScript == itemFullPath:
            self.addActionToContextMenu(_menu, self.plugin.actions["Open Scan Editor"])

            _menu.addSeparator()

            self.addActionToContextMenu(_menu, self.plugin.actions["Open XML Access Path Editor"])

        _menu.addSeparator()

    def addConvertXMLToPythonMenu(self, _menu, _projItem):

        # print "TRYING TO ADD GENERATE STEPPEBLE MENU"

        pdh = None

        try:

            pdh = self.plugin.projectDataHandlers[qt_obj_hash(_projItem)]

        except LookupError as e:

            return

            # check if the file to which we are trying to add Steppable is Python resource        

        itemFullPath = str(self.getFullPath(self.currentItem()))

        basename, extension = os.path.splitext(itemFullPath)

        print("itemFullPath=", itemFullPath)

        print('extension=', extension)

        if extension.lower() == '.xml':
            self.addActionToContextMenu(_menu, self.plugin.actions["Convert XML to Python"])

            self.plugin.xmlFileToConvert = itemFullPath

            return

        if pdh.cc3dSimulationData.xmlScript != '':
            self.addActionToContextMenu(_menu, self.plugin.actions["Convert XML to Python"])

            self.plugin.xmlFileToConvert = str(pdh.cc3dSimulationData.xmlScript)

    def __restyle(self):

        root_item = self.invisibleRootItem()

        self.styleChildItems(root_item, self.style)

    def styleChildItems(self, _item, _style):

        if not _style: return

        _item.setForeground(0, QBrush(self.N2C(_style.fgColor)))

        for idx in range(_item.childCount()):
            childItem = _item.child(idx)

            childItem.setForeground(0, QBrush(self.N2C(_style.fgColor)))

            self.styleChildItems(childItem, _style)

    def applyStyleFromTheme(self, _styleName, _themeName):

        themeManager = self.__ui.themeManager

        self.style = themeManager.getStyleFromTheme(_styleName=_styleName, _themeName=_themeName)

        self.setIconSize(QSize(16, 16))

        if self.style:

            self.N2C = themeManager.npStrToQColor

            qtVersion = str(QtCore.QT_VERSION_STR).split('.')

            if int(qtVersion[0]) >= 2:
                bgColorQt = self.N2C(self.style.bgColor)

                colorString = 'rgb(' + str(bgColorQt.red()) + ',' + str(bgColorQt.green()) + ',' + str(

                    bgColorQt.blue()) + ')'

                # because we used style sheets for the qt app  (twedit_plus_plus.py) we have to use stylesheet to color QTreeWidget  

                # at least on OSX 10.9 using stylesheets for the app requires using them to set properties of widget

                self.setStyleSheet("QTreeWidget {background-color: " + colorString + " ;}")



        else:

            pal = self.palette()

            pal.setBrush(QPalette.Base, QBrush(self.N2C(self.style.bgColor)))

            self.setPalette(pal)

        self.__restyle()


# bgColorQt=self.N2C(self.style.bgColor)

#         colorString='rgb('+str(bgColorQt.red())+','+str(bgColorQt.green())+','+str(bgColorQt.blue())+')'

#         # because we used style sheets for the qt app  (twedit_plus_plus.py) we have to use stylesheet to color QTreeWidget  

#         # at least on OSX 10.9 using stylesheets for the app requires using them to set properties of widget

#         self.setStyleSheet( "QTreeWidget {background-color: "+colorString+ " ;}" )


class CustomDockWidget(QDockWidget):

    def __init__(self, _parent=None):
        QDockWidget.__init__(self, _parent)

        self.cc3dProject = None

    def setCC3DProject(self, cc3dProject):
        self.cc3dProject = cc3dProject

    def closeEvent(self, ev):
        print('close event custom dock widget')

        self.cc3dProject.showProjectPanel(False)

        ev.ignore()


class CC3DProject(QObject, TweditPluginBase):
    """

    Class implementing the About plugin.

    """

    def __init__(self, ui):

        """

        Constructor

        

        @param ui reference to the user interface object (UI.UserInterface)

        """

        QObject.__init__(self, ui)
        TweditPluginBase.__init__(self)

        self.__ui = ui

        self.configuration = Configuration(self.__ui.configuration.settings)

        self.actions = {}

        self.projectDataHandlers = {}

        self.openProjectsDict = {}

        self.hideContextMenuIcons = False

        mac_ver = platform.mac_ver()

        if mac_ver[0]:
            self.hideContextMenuIcons = True  # on OSX we hide context menu icons

        # self.listener=CompuCell3D.CC3DListener.CC3DListener(self.__ui)

        # self.listener.setPluginObject(self)

        self.__initActions()

        self.__initMenus()

        self.__initUI()

        self.__initToolbar()

        self.steppableTemplates = None

        self.xmlFileToConvert = None

        # parameter scan globals

        self.parameterScanEditor = None  # only one scan editor is allowed at any given time

        self.scannedFileName = ''  # this is the path to the file which is open in parameterScanEditor

        self.access_path_editor = None  # only one access path editor is allowed at any given time

        self.access_path_fname = ''  # this is the path to the file which is open in access_path_editor

        self.access_path_xml_handler = None

        self.xml_elem_access_path = None

        self.xml_access_path_obj = None

        # self.parameterScanXMLHandler=None

        # self.parameterScanFile=''

        # self.openCC3Dproject("/Users/m/CC3DProjects/ExtraFields/ExtraFields.cc3d")

        # self.openCC3Dproject("/Users/m/CC3DProjects/scientificPlotsSimple/scientificPlots.cc3d")

        # self.openCC3Dproject("/Users/m/CC3DProjects/ParamScanDemo/ParamScanDemo.cc3d")

        # # # self.openCC3Dproject('/home/m/CC3DProjects/CellSorting/CellSorting.cc3d')

        # # # # self.treeWidget.applyStyle(self.defaultStyle)

        self.treeWidget.applyStyleFromTheme(_styleName='Default Style', _themeName=self.__ui.currentThemeName)

        # # # self.styleItems()

        self.hideContextMenuIcons = False

        mac_ver = platform.mac_ver()

        if mac_ver[0]:
            self.hideContextMenuIcons = True  # on OSX we hide context menu icons

    def getUI(self):

        return self.__ui

    def activate(self):

        """

        Public method to activate this plugin.

        

        @return tuple of None and activation status (boolean)

        """
        print("CC3D PLUGIN ACTIVATE ACTION")
        # print "CC3D PLUGIN ACTIVATE"

        # self.__initActions()

        # print "CC3D INIT ACTIONS"

        # self.__initMenu()

        return None, True

    def post_activate(self, **kwds):
        """
        Post activation function used to add actions to different menus to make certain actions more visible
        e.g. Add Steppable ... action
        :return:
        """

        print('Running post_activate')
        menu_bar = self.__ui.menuBar()
        cc3d_python_menu_title = "CC3D P&ython"
        cc3d_python_menu = None
        for menu in menu_bar.findChildren(QMenu):
            if menu.title() == cc3d_python_menu_title:
                cc3d_python_menu = menu
                break

        if cc3d_python_menu is None:
            print(f'Could not locate CC3D {cc3d_python_menu_title}')
            return

        cc3d_python_menu.addSeparator()
        cc3d_python_menu.addAction(self.actions["Add Steppable CC3D Python..."])

    def deactivate(self):

        """

        Public method to deactivate this plugin.

        """

        # have to close all the projects

        projItems = list(self.projectDataHandlers.keys())

        for projItem in projItems:
            self.closeProjectUsingProjItem(projItem)

        showCC3DProjectPanel = self.configuration.setSetting("ShowCC3DProjectPanel",

                                                             not self.cc3dProjectDock.isHidden())

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

        if "CompuCell3D" not in self.__ui.toolBar:
            self.__ui.toolBar["CompuCell3D"] = self.__ui.addToolBar("CompuCell3D")

            self.__ui.insertToolBar(self.__ui.toolBar["File"], self.__ui.toolBar["CompuCell3D"])

        self.__ui.toolBar["CompuCell3D"].addAction(self.actions["Open CC3D Project..."])

        self.__ui.toolBar["CompuCell3D"].addAction(self.actions["Save CC3D Project"])

    def __initMenus(self):

        self.cc3dProjectMenu = QMenu("CC3D Projec&t", self.__ui.menuBar())

        # inserting CC3D Project Menu as first item of the menu bar of twedit++

        self.__ui.menuBar().insertMenu(self.__ui.fileMenu.menuAction(), self.cc3dProjectMenu)

        self.cc3dProjectMenu.addAction(self.actions["New CC3D Project..."])

        self.cc3dProjectMenu.addAction(self.actions["Open CC3D Project..."])

        self.cc3dProjectMenu.addAction(self.actions['Open Most Recent CC3D Project'])

        self.cc3dProjectMenu.addAction(self.actions["Save CC3D Project"])

        self.cc3dProjectMenu.addAction(self.actions["Save CC3D Project As..."])

        self.cc3dProjectMenu.addAction(self.actions["Zip It!"])
        self.cc3dProjectMenu.addAction(self.actions["UnZip It..."])

        self.cc3dProjectMenu.addSeparator()

        # ---------------------------------------

        self.cc3dProjectMenu.addAction(self.actions["Open in Player"])

        self.cc3dProjectMenu.addSeparator()

        # ---------------------------------------

        # self.cc3dProjectMenu.addAction(self.actions["Save CC3D Project As..."])

        self.cc3dProjectMenu.addAction(self.actions["Add Resource..."])

        self.cc3dProjectMenu.addAction(self.actions["Remove Resources"])

        self.cc3dProjectMenu.addSeparator()

        # ---------------------------------------

        self.cc3dProjectMenu.addAction(self.actions["Open In Editor"])

        self.cc3dProjectMenu.addAction(self.actions["Open XML/Python In Editor"])

        self.cc3dProjectMenu.addSeparator()

        # ---------------------------------------------------

        # Parameter scan Menu

        self.cc3dProjectMenu.addAction(self.actions["Add Parameter Scan"])

        self.cc3dProjectMenu.addAction(self.actions["Add To Scan..."])
        self.cc3dProjectMenu.addAction(self.actions["Remove From Scan..."])

        # self.cc3dProjectMenu.addSeparator()

        # ---------------------------------------

        self.recentProjectsMenu = self.cc3dProjectMenu.addMenu("Recent Projects...")

        # self.connect(self.recentProjectsMenu, SIGNAL("aboutToShow()"), self.updateRecentProjectsMenu)

        self.recentProjectsMenu.aboutToShow.connect(self.updateRecentProjectsMenu)

        self.recentProjectDirectoriesMenu = self.cc3dProjectMenu.addMenu("Recent Project Directories...")

        # self.connect(self.recentProjectDirectoriesMenu, SIGNAL("aboutToShow()"),

        #              self.updateRecentProjectDirectoriesMenu)

        self.recentProjectDirectoriesMenu.aboutToShow.connect(self.updateRecentProjectDirectoriesMenu)

        self.cc3dProjectMenu.addSeparator()

        # ---------------------------------------

        self.cc3dProjectMenu.addAction(self.actions["Show Project Panel"])

        self.cc3dProjectMenu.addSeparator()

        # ---------------------------------------

        self.cc3dProjectMenu.addAction(self.actions["Close Project"])

    def __loadRecentProject(self):

        print('__loadRecentProject')

        action = self.sender()

        fileName = ''

        if isinstance(action, QAction):
            # fileName = str(action.data().toString())

            fileName = str(action.data())

            self.openCC3Dproject(fileName)

    def open_recent_project_directory(self):

        action = self.sender()

        if isinstance(action, QAction):
            dir_name = str(action.data())
            dir_name_path = Path(dir_name)
            if not dir_name_path.exists():
                QMessageBox.warning(self.treeWidget, 'Directory not found',
                                    f'Directory you are trying to access <br> '
                                    f'{dir_name} <br>'
                                    f'does not exist')
                self.__ui.remove_item_from_configuration_string_list(self.configuration, "RecentProjectDirectories",
                                                                     dir_name)

                return

            dir_name = os.path.abspath(dir_name)

            self.__ui.add_item_to_configuration_string_list(self.configuration, "RecentProjectDirectories", dir_name)

            self.showOpenProjectDialogAndLoad(dir_name)

    def updateRecentProjectsMenu(self):

        self.__ui.updateRecentItemMenu(self, self.recentProjectsMenu, self.__loadRecentProject, self.configuration,

                                       "RecentProjects")

    def updateRecentProjectDirectoriesMenu(self):

        self.__ui.updateRecentItemMenu(self, self.recentProjectDirectoriesMenu, self.open_recent_project_directory,

                                       self.configuration, "RecentProjectDirectories")

    def applyStyleFromTheme(self, _styleDict):

        print('_styleDict=', _styleDict)

        try:

            styleName = _styleDict['styleName']

            themeName = _styleDict['themeName']

            print('self.treeWidget=', self.treeWidget)

            self.treeWidget.applyStyleFromTheme(_styleName=styleName, _themeName=themeName)

        except LookupError as e:

            return

    def __initUI(self):

        self.cc3dProjectDock = self.__createDockWindow("CC3D Project")

        self.textEdit = QTextEdit()

        self.treeWidget = CC3DProjectTreeWidget()

        self.treeWidget.setCC3DProjectPlugin(self)

        self.__setupDockWindow(self.cc3dProjectDock, Qt.LeftDockWidgetArea, self.treeWidget, "CC3D Project")

        showCC3DProjectPanel = self.configuration.setting("ShowCC3DProjectPanel")

        if not showCC3DProjectPanel:
            self.showProjectPanel(False)

    def __createDockWindow(self, name):

        """

        Private method to create a dock window with common properties.

        

        @param name object name of the new dock window (string or QString)

        @return the generated dock window (QDockWindow)

        """

        dock = CustomDockWidget(self.__ui)

        dock.setCC3DProject(self)

        #         dock = QDockWidget(self.__ui)

        dock.setObjectName(name)

        # dock.setFeatures(QDockWidget.DockWidgetFeatures(QDockWidget.AllDockWidgetFeatures))

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

        self.actions["New CC3D Project..."] = QtWidgets.QAction(QIcon(':/icons/new-project.png'), "New CC3D Project...",
                                                                self, shortcut="Ctrl+Shift+N",
                                                                statusTip="New CC3D Project Wizard ",
                                                                triggered=self.__newCC3DProject)

        self.actions["Open CC3D Project..."] = QtWidgets.QAction(QIcon(':/icons/open-project.png'),
                                                                 "Open CC3D Project...",
                                                                 self, shortcut="Ctrl+Shift+O",
                                                                 statusTip="Open CC3D Project ",
                                                                 triggered=self.__openCC3DProject)

        self.actions["Open Most Recent CC3D Project"] = QtWidgets.QAction(QIcon(':/icons/open-project.png'),
                                                                          "Open Most Recent CC3D Project",
                                                                          self, shortcut="Ctrl+Shift+Alt+O",
                                                                          statusTip="Opens Most Recent CC3D Project ",
                                                                          triggered=self.open_most_recent_cc3d_project)

        self.actions["Open in Player"] = QtWidgets.QAction(QIcon(':/icons/player5-icon.png'), "Open In Player", self,
                                                           shortcut="", statusTip="Open simulation in Player ",
                                                           triggered=self.__runInPlayer)

        self.actions["Save CC3D Project"] = QtWidgets.QAction(QIcon(':/icons/save-project.png'), "Save CC3D Project",
                                                              self,
                                                              shortcut="Ctrl+Shift+D", statusTip="Save CC3D Project ",
                                                              triggered=self.__save_cc3d_project)

        self.actions["Save CC3D Project As..."] = QtWidgets.QAction("Save CC3D Project As...", self,

                                                                    shortcut="Ctrl+Shift+A",

                                                                    statusTip="Save CC3D Project As ",

                                                                    triggered=self.__saveCC3DProjectAs)

        self.actions["Zip It!"] = QtWidgets.QAction("Zip It!", self, shortcut="Ctrl+Shift+Z",

                                                    statusTip="Zips project directory", triggered=self.__zip_project)

        self.actions["UnZip It..."] = QtWidgets.QAction("UnZip It...", self,
                                                        statusTip="Unzip and open zipped CC3D project ",
                                                        triggered=self.__openCC3DProject)

        self.actions["Go To Project Directory"] = QtWidgets.QAction("Go To Project Directory", self, shortcut="",
                                                                    statusTip="Opens directory of the project in "
                                                                              "default file manager",
                                                                    triggered=self.__goToProjectDirectory)

        # self.actions["Zip'n'Mail"]=QtWidgets.QAction("Zip'n'Mail", self, statusTip="Zips project directory and opens email clinet with attachement", triggered=self.__zipAndMailProject)

        self.actions["Add Resource..."] = QtWidgets.QAction(QIcon(':/icons/add.png'), "Add Resource...", self,

                                                            shortcut="",

                                                            statusTip="Add Resource File ",

                                                            triggered=self.__addResource)

        self.actions["Add Serializer..."] = QtWidgets.QAction(QIcon(':/icons/add-serializer.png'), "Add Serializer ...",

                                                              self, shortcut="", statusTip="Add Serializer ",

                                                              triggered=self.__addSerializerResource)

        self.actions["Remove Resources"] = QtWidgets.QAction(QIcon(':/icons/remove.png'), "Remove Resources", self,

                                                             shortcut="", statusTip="Remove Resource Files ",

                                                             triggered=self.__removeResources)

        self.actions["Open In Editor"] = QtWidgets.QAction(QIcon(':/icons/open-in-editor.png'), "Open In Editor", self,

                                                           shortcut="", statusTip="Open Document in Editor ",

                                                           triggered=self.__openInEditor)

        self.actions["Open XML/Python In Editor"] = QtWidgets.QAction("Open XML/Python In Editor", self, shortcut="",

                                                                      statusTip="Open XML and Python scripts from the current project in editor ",

                                                                      triggered=self.__openXMLPythonInEditor)

        self.actions["Properties"] = QtWidgets.QAction("Properties", self, shortcut="",

                                                       statusTip="Display/Edit Project Item Properties ",

                                                       triggered=self.__displayProperties)

        self.actions["Serializer..."] = QtWidgets.QAction(QIcon(':/icons/save-simulation.png'), "Serializer...", self,

                                                          shortcut="",

                                                          statusTip="Edit serialization properties fo the simulation ",

                                                          triggered=self.__serializerEdit)

        self.actions["Close Project"] = QtWidgets.QAction("Close Project", self, shortcut="Ctrl+Shift+X",

                                                          statusTip="Close Project ", triggered=self.__closeProject)

        self.actions["Show Project Panel"] = QtWidgets.QAction("Show Project Panel", self, shortcut="",

                                                               statusTip="Show Project Panel")

        self.actions["Show Project Panel"].setCheckable(True)

        self.actions["Show Project Panel"].setChecked(True)

        # self.connect(self.actions["Show Project Panel"], SIGNAL('triggered(bool)'), self.showProjectPanel)

        self.actions["Show Project Panel"].triggered.connect(self.showProjectPanel)

        self.actions["Add Steppable..."] = QtWidgets.QAction(QIcon(':/icons/addSteppable.png'), "Add Steppable...",
                                                             self,
                                                             shortcut="",
                                                             statusTip="Adds Steppable to Python File "
                                                                       "(Cannot be Python Main Script) ",
                                                             triggered=self.add_steppable)

        self.actions["Add Steppable CC3D Python..."] = QtWidgets.QAction(QIcon(':/icons/addSteppable.png'),
                                                                         "Add Steppable...",
                                                                         self,
                                                                         shortcut="",
                                                                         statusTip="Adds Steppable to Python File ",
                                                                         triggered=self.add_steppable_cc3d_python)

        self.actions["Convert XML to Python"] = QtWidgets.QAction(QIcon(':/icons/xml-icon.png'),

                                                                  "Convert XML to Python",

                                                                  self, shortcut="",

                                                                  statusTip="Converts XML into equivalent Python script",

                                                                  triggered=self.__convertXMLToPython)

        self.actions["Add Parameter Scan"] = QtWidgets.QAction(QIcon(':/icons/scan_32x32.png'), "Add Parameter Scan",

                                                               self,

                                                               shortcut="Ctrl+Shift+P", statusTip="Add Parameter Scan ",

                                                               triggered=self.__addParameterScan)

        # on osx 10.9 context menu icons are not rendered properly so we do not include them at all on OSX

        if self.hideContextMenuIcons:

            addToScanIcon = QIcon()

        else:

            addToScanIcon = QIcon(':/icons/add.png')

        self.actions["Add To Scan..."] = QtWidgets.QAction(addToScanIcon, "Add To Scan...", self, shortcut="Ctrl+I",
                                                           statusTip="Add Parameter To Scan",
                                                           triggered=self.__addToScan)

        self.actions["Remove From Scan..."] = QtWidgets.QAction(QIcon(':/icons/remove.png'), "Remove From Scan...",
                                                                self, shortcut="Ctrl+Shift+I",
                                                                statusTip="Remove Parameter from Scan",
                                                                triggered=self.remove_from_parameter_scan)

        self.actions['Open Scan Editor'] = QtWidgets.QAction(QIcon(':/icons/editor.png'), "Open Scan Editor", self,
                                                             shortcut="", statusTip="Open Scan Editor",
                                                             triggered=self.__openScanEditor)

        self.actions['Reset Parameter Scan'] = QtWidgets.QAction(QIcon(':/icons/reset_32x32.png'),

                                                                 "Reset Parameter Scan",

                                                                 self, shortcut="", statusTip="Reset Parameter Scan",

                                                                 triggered=self.__resetParameterScan)

        # XML Access Path Handling

        self.actions['Open XML Access Path Editor'] = QtWidgets.QAction(QIcon(':/icons/editor.png'),

                                                                        "Open XML Access Path Editor", self,

                                                                        shortcut="",

                                                                        statusTip="Open XML Access Path Editor",

                                                                        triggered=self.__open_access_path_editor)

        self.actions["XML Access Path to Clipboard"] = QtWidgets.QAction(addToScanIcon, "XML Access Path to Clipboard",

                                                                         self, shortcut="Ctrl+Shift+X",

                                                                         statusTip="Copies XML access Path to Clipboard",

                                                                         triggered=self.get_access_path)

        self.actions['Get XML Element Value (Clipboard)'] = QtWidgets.QAction(QIcon(':/icons/editor.png'),

                                                                              "Get XML Element Value (Clipboard)", self,

                                                                              shortcut="",

                                                                              statusTip="Get XML Element Value (Clipboard)",

                                                                              triggered=self.__get_xml_element_value_snippet)

        self.actions['Set XML Element Value (Clipboard)'] = QtWidgets.QAction(QIcon(':/icons/editor.png'),

                                                                              "Set XML Element Value (Clipboard)", self,

                                                                              shortcut="",

                                                                              statusTip="Set XML Element Value (Clipboard)",

                                                                              triggered=self.__set_xml_element_value_snippet)

    def __resetParameterScan(self):

        tw = self.treeWidget

        ret = QMessageBox.warning(tw, "Parameter Scan Reset",

                                  "You are about to reset parameter scan to start from the beginning. Do you want to proceed?",

                                  QMessageBox.Yes | QMessageBox.No)

        if ret == QMessageBox.No: return

        projItem = tw.getProjectParent(tw.currentItem())

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            print("could not find simulation data handler for this item")

            return

        if pdh.cc3dSimulationData.parameterScanResource:
            psu = ParameterScanUtils()

            psu.resetParameterScan(pdh.cc3dSimulationData.parameterScanResource.path)

            #    

            # self.__ui.deactivateChangeSensing=True

            self.__ui.checkIfDocumentsWereModified()

            # self.__ui.deactivateChangeSensing=False

    def __addParameterScan(self):

        tw = self.treeWidget
        proj_item = tw.getProjectParent(tw.currentItem())

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            print("could not find simulation data handler for this item")
            return

        if pdh.cc3dSimulationData.parameterScanResource:
            QMessageBox.warning(tw, "Parameter Scan is already defined",

                                "You cannot have more than one parameter scan specifications in the simulation")

            return

        pdh.cc3dSimulationData.addNewParameterScanResource()  # adding empty parameter scan resource

        resourceFileName = pdh.cc3dSimulationData.parameterScanResource.path

        # insert new file into the tree

        self.insertNewGenericResourceTreeItem(pdh.cc3dSimulationData.parameterScanResource)

        pdh.cc3dSimulationData.parameterScanResource.write_parameter_scan_specs()

        # setting same base path for parameter scan as for the project - necessary to get relative
        # paths in the parameterSpec file
        pdh.cc3dSimulationData.parameterScanResource.basePath = pdh.cc3dSimulationData.basePath

        self.__save_cc3d_project()

    def __closeScanEditor(self):

        if not self.parameterScanEditor: return

        panel, idx = self.__ui.getTabWidgetAndWidgetIndex(self.parameterScanEditor)

        if panel and idx >= 0:
            self.__ui.closeTab(index=idx, _askToSave=False, _panel=panel)

        self.parameterScanEditor = None

    def __close_access_path_editor(self):

        if not self.access_path_editor: return

        panel, idx = self.__ui.getTabWidgetAndWidgetIndex(self.access_path_editor)

        if panel and idx >= 0:
            self.__ui.closeTab(index=idx, _askToSave=False, _panel=panel)

        self.access_path_editor = None

        # def  __closeScanEditorEvent(self,event):

        # print 'LOCAL CLOSE EVENT'

        # self.parameterScanEditor=None

    def __openScanEditor(self):

        if self.parameterScanEditor:
            self.__closeScanEditor()

        tw = self.treeWidget

        projItem = tw.getProjectParent(tw.currentItem())

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            return

        if not pdh.cc3dSimulationData.parameterScanResource:
            QMessageBox.warning(tw, "Please Add Parameter Scan",

                                "Parameter scan editor can only be open when project includes Parameter Scan. Please add parameter scan first ")

            return

            print('')

            return

        pScanResource = pdh.cc3dSimulationData.parameterScanResource

        itemFullPath = str(tw.getFullPath(tw.currentItem()))

        basename, extension = os.path.splitext(itemFullPath)

        pScanResource.fileTypeForEditor = extension.lower()

        self.scannedFileName = itemFullPath  # store scanned file name in the global - we can only have one parameter scan editor open

        # opening editor

        self.__ui.newFile()

        editor = self.__ui.getCurrentEditor()

        editor.setReadOnly(True)

        # # set tab font color color

        # tabBar=activePanel.tabBar()

        # tabBar.setTabIcon()

        # tabBar.setStyleSheet('background-color: blue;')

        lexer = self.__ui.guessLexer("tmp" + pScanResource.fileTypeForEditor)

        if lexer[0]:
            editor.setLexer(lexer[0])

        self.__ui.setEditorProperties(editor)

        editor.registerCustomContextMenu(self.createParameterScanMenu(editor))

        # initialize globals

        self.parameterScanEditor = editor

        # self.parameterScanEditor.closeEvent=self.__closeScanEditorEvent # close event will be handled via local function

        # pScanResource.parameterScanEditor=editor

        if pScanResource.fileTypeForEditor == '.xml':  # for xml we have to get generate line to access path map and line to element map for easier handling of parameter scan generation

            cc3dXML2ObjConverter = XMLUtils.Xml2Obj()

            root_element = cc3dXML2ObjConverter.Parse(self.scannedFileName)

            xmlHandler = XMLHandler()

            xmlHandler.outputXMLWithAccessPaths(self.scannedFileName)

            print(xmlHandler.lineToElem)

            print(xmlHandler.lineToAccessPath)

            editor.insertAt(xmlHandler.xmlString, 0, 0)

            editor.setModified(False)

            pScanResource.parameterScanXMLHandler = xmlHandler

        if pScanResource.fileTypeForEditor == '.py':
            editor.insertAt(open(self.scannedFileName).read(), 0, 0)

            editor.setModified(False)

        # setting graphical  properties for parameter scan editor tab widget

        activePanel, currentindex = self.__ui.getCurrentTabWidgetAndIndex()

        activePanel.setTabText(currentindex, 'Parameter Scan Tmp File')

        activePanel.setTabIcon(currentindex, QIcon(':/icons/scan_32x32.png'))

        tabBar = activePanel.tabBar()

        tabBar.setTabTextColor(currentindex, QColor('blue'))

    def __open_access_path_editor(self):

        if self.access_path_editor:
            self.__close_access_path_editor()

        tw = self.treeWidget

        projItem = tw.getProjectParent(tw.currentItem())

        itemFullPath = str(tw.getFullPath(tw.currentItem()))

        basename, extension = os.path.splitext(itemFullPath)

        # pScanResource.fileTypeForEditor = extension.lower()

        self.access_path_fname = itemFullPath  # store access path file name in the global - we can only have one parameter access path editor open

        # opening editor

        self.__ui.newFile()

        editor = self.__ui.getCurrentEditor()

        editor.setReadOnly(True)

        # # set tab font color color

        # tabBar=activePanel.tabBar()

        # tabBar.setTabIcon()

        # tabBar.setStyleSheet('background-color: blue;')

        # lexer = self.__ui.guessLexer("tmp" + pScanResource.fileTypeForEditor)

        lexer = self.__ui.guessLexer("tmp" + extension)

        if lexer[0]:
            editor.setLexer(lexer[0])

        self.__ui.setEditorProperties(editor)

        # editor.registerCustomContextMenu(self.createParameterScanMenu(editor))

        editor.registerCustomContextMenu(self.create_access_path_menu(editor))

        # initialize globals

        self.access_path_editor = editor

        # self.parameterScanEditor = editor

        # self.parameterScanEditor.closeEvent=self.__closeScanEditorEvent # close event will be handled via local function

        # pScanResource.parameterScanEditor=editor

        # if pScanResource.fileTypeForEditor == '.xml':  # for xml we have to get generate line to access path map and line to element map for easier handling of parameter scan generation

        if extension == '.xml':  # for xml we have to get generate line to access path map and line to element map for easier handling of parameter scan generation

            cc3dXML2ObjConverter = XMLUtils.Xml2Obj()

            root_element = cc3dXML2ObjConverter.Parse(self.access_path_fname)

            xml_handler = XMLHandler()

            xml_handler.outputXMLWithAccessPaths(self.access_path_fname)

            print(xml_handler.lineToElem)

            print(xml_handler.lineToAccessPath)

            editor.insertAt(xml_handler.xmlString, 0, 0)

            editor.setModified(False)

            self.access_path_xml_handler = xml_handler

            # pScanResource.parameterScanXMLHandler = xmlHandler

        #

        # if pScanResource.fileTypeForEditor == '.py':

        #     editor.insertAt(open(self.scannedFileName).read(), 0, 0)

        #     editor.setModified(False)

        #

        # setting graphical  properties for parameter scan editor tab widget

        activePanel, currentindex = self.__ui.getCurrentTabWidgetAndIndex()

        activePanel.setTabText(currentindex, 'XML Access Path Tmp File')

        activePanel.setTabIcon(currentindex, QIcon(':/icons/scan_32x32.png'))

        tabBar = activePanel.tabBar()

        tabBar.setTabTextColor(currentindex, QColor('blue'))

    def get_access_path(self):

        """

        Opens XML Access Path Read-only editor and extracts access path  (full access path to the XML element)

        :return: {None} access path gets copied to the clipboard

        """

        tw = self.treeWidget

        projItem = tw.getProjectParent(tw.currentItem())

        print('projItem=', projItem)

        print('self.projectDataHandlers=', self.projectDataHandlers)

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            return

        csd = pdh.cc3dSimulationData

        # pScanResource = pdh.cc3dSimulationData.parameterScanResource

        print('__addToScan')

        if not self.access_path_editor: return

        # check if the editor is still open

        editorExists = self.__ui.checkIfEditorExists(self.access_path_editor)

        if not editorExists:
            self.access_path_editor = None

            return

        line, col = self.access_path_editor.getCursorPosition()

        print('line,col=', (line, col))

        # if pScanResource

        access_fname_extension = os.path.splitext(self.access_path_fname)[1].lower()

        if access_fname_extension == '.xml':

            # psXMLHandler = pScanResource.parameterScanXMLHandler

            access_path_xml_handler = self.access_path_xml_handler

            self.access_path_obj = ''

            if access_path_xml_handler:

                try:

                    self.access_path_obj = access_path_xml_handler.lineToAccessPath[line]

                    self.xml_elem_access_path = access_path_xml_handler.lineToElem[line]

                except LookupError as e:

                    print('Could not figure out access path')

                print('AccessPath=', self.access_path_obj)

            if not self.access_path_obj:
                return

            clipboard = QApplication.clipboard()

            clipboard.setText(str(self.access_path_obj))

            return

    def get_xml_value_callback(self, orig_access_path, attribute_name=None):

        """

        Returns full or partial access path depending if we are accessing CDATA value or an attribute

        :param orig_access_path:{str} access path - full XML element

        :param attribute_name: {str or None} -  optional attribute name - if accessing attribute, otherwise None

        :return: {str} access path (possibly partial access path for attributes)

        """

        print('orig_access_path=', orig_access_path)

        print('attribute_name=', attribute_name)

        local_access_path = deepcopy(self.access_path_obj)

        if attribute_name is not None:

            last_access_path_segment = local_access_path[-1]

            for i, item in enumerate(last_access_path_segment):

                if str(item) == str(attribute_name):
                    break

            # removing i'th and the next elementn

            last_access_path_segment.pop(i)

            last_access_path_segment.pop(i)

            attr_access_path = local_access_path[:-1] + [last_access_path_segment]

        return local_access_path

    def process_xml_access_path_dialog(self):

        """

        Opens up xml access path dialog and extracts precise access path to the xml element component

        :return: {instance of  XmlAccessPathTuple}

        """

        self.get_access_path()

        access_path = QApplication.clipboard().text()

        s = 'access_path={}\n'.format(access_path)

        s += 'val=float(self.getXMLElementValue(*access_path))\n'

        QApplication.clipboard().setText(s)

        print('self.scannedFileName=', self.scannedFileName, '\n\n\n\n\n')

        xml_access_path = XmlAccessPathDialog(self.parameterScanEditor)

        # xml_access_path.setWindowTitle('XML Access Path Selection')

        try:

            xml_access_path.display_xml_attributes(self.xml_elem_access_path, access_path,

                                                   handle_xml_access_callback=self.get_xml_value_callback)

            ret = xml_access_path.exec_()

        except LookupError as e:  # to protect against elements that are not in psXMLHandler.lineToAccessPath

            return

        precise_xml_access_path_tuple = xml_access_path.get_precise_xml_access_path_tuple()

        return precise_xml_access_path_tuple

    def __get_xml_element_value_snippet(self):

        """

        Callback to get xml element value/attribute using xml access path. Copies

        a snippet to the clipboard

        :return:{None} - code gets copied into clipboard

        """

        precise_xml_access_path_tuple = self.process_xml_access_path_dialog()

        # default snippet value

        s = '__get_xml_element_value_snippet'

        if precise_xml_access_path_tuple.type == XML_CDATA:

            s = 'access_path={}\n'.format(precise_xml_access_path_tuple.access_path)

            s += 'val=float(self.getXMLElementValue(*access_path))\n'

        elif precise_xml_access_path_tuple.type == XML_ATTR:

            s = 'access_path={}\n'.format(precise_xml_access_path_tuple.access_path)

            s += "val=float(self.getXMLAttributeValue('{attr_name}',*access_path))\n".format(

                attr_name=precise_xml_access_path_tuple.name)

        else:

            print((

                'Expected precise_xml_access_path_tuple to be of type {}'.format('ParameterDialog.XmlAccessPathTuple')))

        print('code to modify elem = ', s)

        QApplication.clipboard().setText(s)

    def __set_xml_element_value_snippet(self):

        """

        Callback to get xml element value/attribute using xml access path. Copies

        a snippet to the clipboard

        :return:{None} - code gets copied into clipboard

        """

        precise_xml_access_path_tuple = self.process_xml_access_path_dialog()

        # default snippet value

        s = '__set_xml_element_value_snippet'

        if precise_xml_access_path_tuple.type == XML_CDATA:

            s = 'access_path={}\n'.format(precise_xml_access_path_tuple.access_path)

            s += 'self.setXMLElementValue(VALUE, *access_path)\n'

        elif precise_xml_access_path_tuple.type == XML_ATTR:

            s = 'access_path={}\n'.format(precise_xml_access_path_tuple.access_path)

            s += "self.setXMLAttributeValue('{attr_name}',VALUE,*access_path)\n".format(

                attr_name=precise_xml_access_path_tuple.name)

        else:

            raise TypeError(

                'Expected precise_xml_access_path_tupl to be of type {}'.format('ParameterDialog.XmlAccessPathTuple'))

        print('code to modify elem = ', s)

        QApplication.clipboard().setText(s)

    def check_current_editor_project(self):

        editor = self.__ui.getCurrentEditor()
        current_fname = self.__ui.get_editor_file_name(editor=editor)

        tw = self.treeWidget

        proj_item = tw.getProjectParent(tw.currentItem())

        if not proj_item:
            return

        try:
            ild = tw.projects[qt_obj_hash(proj_item)]
        except LookupError:
            ild = None

        if ild is None:
            return False

        for tw_item, cc3d_resource in ild.itemToResource.items():
            if os.path.abspath(cc3d_resource.path) == os.path.abspath(current_fname):
                return True

        return False

    def remove_from_parameter_scan(self):
        tw = self.treeWidget

        already_added_param_name = self.get_already_added_param()

        if already_added_param_name is None:
            QMessageBox.warning(tw, 'Not a Definition of Scanned Parameter',
                                'The selected Item is not a definition of the scanned parameter. '
                                'Scanned parameters are enclosed between <b>"{{"</b> and <b>"}}"</b>')
            return

        proj_item = tw.getProjectParent(tw.currentItem())

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except KeyError:
            QMessageBox.warning(tw, 'Could Not Find Active CC3D Project',
                                'Please Open or activate (by clicking on the project in the left panel) '
                                'CC3D project before trying tro modify parameter scan specifications')
            return

        csd = pdh.cc3dSimulationData

        if csd.parameterScanResource is None:
            QMessageBox.warning(tw, "Parameter Scan resource Does not Exist for This Project",
                                'Please add Parameter Scan Resource to the project')
            return

        selection_ok = self.check_selection_end_characters_param_scan()
        editor = self.__ui.getCurrentEditor()
        current_fname = self.__ui.get_editor_file_name(editor=editor)

        if not selection_ok:
            QMessageBox.warning(tw, "Parameter Scan Selection Issue",
                                "The selected fragment does not look like it can be part of the Parameter Scan. <br>"
                                "Notice: you can still delete it from parameter scan but you need to edit manually <br>"
                                f"<b>{current_fname}</b> <br> and <br> <b>ParameterScan.json</b> file. <br>"
                                "Please follow the guidelines outlined here: <br>"
                                "<a href='https://pythonscriptingmanual.readthedocs.io/en/latest/parameter_scans.html?highlight=parameter%20scan#parameter-scans'>Parameter Scan Tutorial</a>"
                                )
            return

        ans = QMessageBox.question(tw,
                                   'Remove Parameter From Scan?',
                                   f'Are you sure you want to delete <b>{already_added_param_name}</b> '
                                   f'from parameter scan', buttons=QMessageBox.Yes | QMessageBox.No)

        if ans == QMessageBox.Yes:

            ok_flag = self.check_current_editor_project()
            if not ok_flag:
                QMessageBox.warning(tw, "File Does Not Belong to Currently Select Project",
                                    f"File {current_fname} does belong to currenly active project. <br>"
                                    f"Please click on project in the Project Panel to which this file belongs to"
                                    )
                return

            psu = csd.parameterScanResource.psu
            existing_param_scan_data_dict = psu.get_parameter_scan_data_dict(already_added_param_name)
            with contextlib.suppress(KeyError):
                original_value = existing_param_scan_data_dict['original_value']

            psu.remove_from_param_scan(already_added_param_name)
            self.replace_selection_with(replacement_text=original_value)
            # reopening json file to show recent changes
            self.__ui.loadFile(csd.parameterScanResource.path)

            self.__ui.checkIfDocumentsWereModified()

    def get_current_param_scan_resource(self):
        tw = self.treeWidget

        proj_item = tw.getProjectParent(tw.currentItem())

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError as e:
            return

        p_scan_resource = pdh.cc3dSimulationData.parameterScanResource
        return p_scan_resource

    def __addToScan(self):

        tw = self.treeWidget

        proj_item = tw.getProjectParent(tw.currentItem())

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            QMessageBox.warning(tw, 'Could Not Find Active CC3D Project',
                                'Please Open or activate (by clicking on the project in the left panel) '
                                'CC3D project before trying tro modify parameter scan specifications')  
            return

        csd = pdh.cc3dSimulationData

        if csd.parameterScanResource is None:
            QMessageBox.warning(tw, "Parameter Scan resource Does not Exist for This Project",
                                'Please add Parameter Scan Resource to the project')
            return

        p_scan_resource = pdh.cc3dSimulationData.parameterScanResource

        editor = self.__ui.getCurrentEditor()
        current_fname = self.__ui.get_editor_file_name(editor=editor)

        selection_ok = self.check_selection_end_characters_param_scan()
        if not selection_ok:
            current_fname = self.__ui.get_editor_file_name(editor=editor)

            QMessageBox.warning(tw, "Parameter Scan Selection Issue",
                                "The selected fragment does not look like it can be part of the Parameter Scan. <br>"
                                "Notice: you can still add it to parameter scan but you need to edit manually <br>"
                                f"<b>{current_fname}</b> <br> and <br> <b>ParameterScan.json</b> file. <br>"
                                "Please follow the guidelines outlined here: <br>"
                                "<a href='https://pythonscriptingmanual.readthedocs.io/en/latest/parameter_scans.html?highlight=parameter%20scan#parameter-scans'>Parameter Scan Tutorial</a>"
                                )
            return

        ok_flag = self.check_current_editor_project()

        if not ok_flag:
            QMessageBox.warning(tw, "File Does Not Belong to Currently Select Project",
                                f"File {current_fname} does belong to currenly active project. <br>"
                                f"Please click on project in the Project Panel to which this file belongs to"
                                )
            return

        already_added_param_name = self.get_already_added_param()

        parvaldlg = ParValDlg(self.parameterScanEditor)
        if already_added_param_name is not None:
            parvaldlg.set_identifier(val=already_added_param_name)

        if parvaldlg.exec_():

            try:

                parvaldlg.record_values()

            except ValueError as e:
                warning_str = str(e)

                if not len(warning_str):
                    QMessageBox.warning(tw, "Error Parsing Parameter List",
                                        "Please make sure that parameter list entries have correct type")

                else:
                    QMessageBox.warning(tw, "Error With Parameter Specification", warning_str)

                return

            psd = parvaldlg.psd

            if len(psd.customValues):
                if already_added_param_name is not None:
                    psu = csd.parameterScanResource.psu
                    existing_param_scan_data_dict = psu.get_parameter_scan_data_dict(already_added_param_name)
                    with contextlib.suppress(KeyError):
                        original_value = existing_param_scan_data_dict['original_value']

                else:
                    original_value = editor.selectedText()

                try:

                    csd.parameterScanResource.addParameterScanData(psd, original_value=original_value)
                except ValueError as e:
                    QMessageBox.warning(tw, 'Could Not Add Parameter Scan Specs', str(e))
                    return

                # p_scan_resource.write_parameter_scan_specs()

            self.replace_selection_with(replacement_text='{{' + f'{psd.name}' + '}}')
            # reopening json file to show recent changes
            self.__ui.loadFile(p_scan_resource.path)

        self.__ui.checkIfDocumentsWereModified()
        return

    def check_selection_end_characters_param_scan(self):
        current_editor = self.__ui.getCurrentEditor()
        line_start, col_start, line_end, col_end = current_editor.getSelection()
        if line_end != line_start:
            QMessageBox.warning(current_editor, "Multi-Line Selection Detected",
                                "For Parameter Scans multi-line selections are disallowed")

            return False

        current_fname = self.__ui.get_editor_file_name(editor=current_editor)
        ext = Path(current_fname).suffix
        xml_allowed_pairs = [
            ('>', '<'),
            ('"', '"'),
            ('>', ','),
            ('>', ' '),
            (' ', '<'),
            (',', '<'),
        ]
        if ext.lower() in ['.xml', '.sbml']:
            for allowed_pair in xml_allowed_pairs:
                line_text = current_editor.text(line_start)

                try:
                    left_limit_char = line_text[col_start - 1]
                except IndexError:
                    left_limit_char = None

                try:
                    right_limit_char = line_text[col_end]
                except IndexError:
                    right_limit_char = None

                if allowed_pair[0] == left_limit_char and allowed_pair[1] == right_limit_char:
                    return True
        else:
            # for now we allow changes in all other files
            return True

    def get_already_added_param(self):
        """
        Returns label of already added parameter (if such exist) -  it woudl be par_name for
        {{par_name}} example, or returns None . Used to see if user has selected a parameter that has been already
        added.
        :return:
        """
        current_editor = self.__ui.getCurrentEditor()
        line_start, col_start, line_end, col_end = current_editor.getSelection()
        line_text = current_editor.text(line_start)
        left_braces = False
        try:
            if line_text[col_start:col_start + 2] == '{{':
                left_braces = True
        except IndexError:
            pass

        right_braces = False
        try:
            if line_text[col_end - 2:col_end] == '}}':
                right_braces = True
        except IndexError:
            pass

        if left_braces and right_braces:
            return line_text[col_start + 2: col_end - 2]

    def replace_selection_with(self, replacement_text):
        """
        Replaces selected text with replacemtn_text for the current editor
        :param replacement_text:
        :return:
        """
        current_editor = self.__ui.getCurrentEditor()
        line_start, col_start, line_end, col_end = current_editor.getSelection()
        current_editor.removeSelectedText()
        current_editor.insertAt(replacement_text, line_start, col_start)

        current_fname = self.__ui.get_editor_file_name(editor=current_editor)
        self.__ui.saveFile(current_fname, _editor=current_editor)

    def restoreIcons(self):

        print('restore icons for scan menu')

        for action, icon in self.__iconDict.items():
            action.setIcon(icon)

    def addActionToContextMenu(self, _menu, _action):

        _menu.addAction(_action)

    def createParameterScanMenu(self, _widget):

        # resetting icon dictionary
        self.__iconDict = {}

        self.hideContextMenuIcons = True

        menu = QMenu(_widget)

        menu.aboutToHide.connect(self.restoreIcons)

        self.addActionToContextMenu(menu, self.actions["Add To Scan..."])
        self.addActionToContextMenu(menu, self.actions["Remove From Scan..."])

        return menu

    def create_access_path_menu(self, _widget):

        self.__iconDict = {}  # resetting icon dictionary

        self.hideContextMenuIcons = True

        menu = QMenu(_widget)

        menu.aboutToHide.connect(self.restoreIcons)

        self.addActionToContextMenu(menu, self.actions["Get XML Element Value (Clipboard)"])

        self.addActionToContextMenu(menu, self.actions["Set XML Element Value (Clipboard)"])

        menu.addSeparator()

        self.addActionToContextMenu(menu, self.actions["XML Access Path to Clipboard"])

        #         menu.addAction(self.actions["Add To Scan..."])

        return menu

    def __serializerEdit(self):

        se = SerializerEdit(self.treeWidget)

        resource = self.treeWidget.getCurrentResource()

        se.setupDialog(resource)

        if se.exec_():
            se.modifySerializerResource(resource)

            projItem = self.treeWidget.getProjectParent(self.treeWidget.currentItem())

            self.markProjectDirty(projItem)

    def __convertXMLToPython(self):

        print("CONVERTING XML TO PYTHON")

        print("self.xmlFileToConvert=", self.xmlFileToConvert)

        if self.xmlFileToConvert:

            cc3dXML2ObjConverter = XMLUtils.Xml2Obj()

            root_element = cc3dXML2ObjConverter.Parse(self.xmlFileToConvert)

            dirToStoreTmpFile = os.path.dirname(self.xmlFileToConvert)

            tmpFilePath = os.path.join(dirToStoreTmpFile, 'tmp.py')

            tmpFilePath = os.path.abspath(tmpFilePath)  # normalizing the path

            configureSimFcnBody = cc3dPythonGen.generate_configure_sim_fcn_body(root_element, tmpFilePath)

            configureSimFcnBody += '\n'

            self.__ui.newFile()

            editor = self.__ui.getCurrentEditor()

            editor.insertAt(configureSimFcnBody, 0, 0)

            lexer = self.__ui.guessLexer("tmp.py")

            if lexer[0]:
                editor.setLexer(lexer[0])

            self.__ui.setEditorProperties(editor)

            self.xmlFileToConvert = None

    @staticmethod
    def check_if_in_project(pdh, file_path: str) -> bool:
        """
        Checks if a given file is belongs to a project(represented by pdh)
        :param pdh: project data handle
        :param file_path:
        :return:
        """
        cc3d_simulation_data = pdh.cc3dSimulationData
        file_path = file_path.strip()

        if cc3d_simulation_data.pythonScript.strip() == file_path:
            return True
        if cc3d_simulation_data.pythonScriptResource.path.strip() == file_path:
            return True
        if cc3d_simulation_data.xmlScript.strip() == file_path:
            return True

        for key, resource in cc3d_simulation_data.resources.items():
            if resource.path == file_path:
                return True

        return False

    def find_project_for_file(self, file_path: str):
        """
        Scans all open projects in the CCC3D Project panel to locate a project that a given file (file_path)
        belongs to. If nothing can be found it returns None

        :param file_path:
        :return:
        """

        pdh_found = None

        for pdh in self.projectDataHandlers.values():
            if self.check_if_in_project(pdh=pdh, file_path=file_path):
                pdh_found = pdh
                break

        return pdh_found

    @staticmethod
    def is_main_python_script(pdh, file_path):
        """
        checks if a given file is main Python script
        :param pdh:
        :param file_path:
        :return:
        """
        return pdh.cc3dSimulationData.pythonScript.strip() == file_path

    @staticmethod
    def is_python_resource_script(pdh, file_path):
        """
        checks if a given file is Python resource script (usually a steppable script)
        :param pdh:
        :param file_path:
        :return:
        """
        for key, resource in pdh.cc3dSimulationData.resources.items():
            if resource.path == file_path:
                return True

        return False

    def open_file_in_editor(self, file_path):
        """
        Opens file in editor and returns editor object
        :param file_path:
        :return:
        """
        self.__ui.loadFile(file_path)
        editor = self.__ui.getCurrentEditor()
        return editor

    def insert_steppable_template(self, main_python_script_editor, steppable_script_editor):
        """
        Inserts template steppable code into editors - main python and steppable editors
        :param main_python_script_editor:
        :param steppable_script_editor:
        :return:
        """
        tw = self.treeWidget

        main_python_script_path = self.__ui.getEditorFileName(main_python_script_editor)
        steppable_script = self.__ui.getEditorFileName(steppable_script_editor)

        basename_for_import, ext = os.path.splitext(os.path.basename(steppable_script))

        sgd = SteppableGeneratorDialog(tw)

        sgd.mainScriptLB.setText(main_python_script_path)

        if not sgd.exec_():
            return

        steppeble_name = str(sgd.steppebleNameLE.text())

        if not steppeble_name.lower().endswith('steppable'):
            # append Steppable to new steppable in case user does not do that
            steppeble_name += 'Steppable'

        frequency = sgd.freqSB.value()

        steppable_type = "Generic"

        if sgd.genericLB.isChecked():
            steppable_type = "Generic"

        elif sgd.mitosisRB.isChecked():
            steppable_type = "Mitosis"

        elif sgd.clusterMitosisRB.isChecked():
            steppable_type = "ClusterMitosis"

        elif sgd.runBeforeMCSRB.isChecked():
            steppable_type = "RunBeforeMCS"

        extra_fields = []

        if sgd.scalarCB.isChecked():
            extra_fields.append("Scalar")

        if sgd.scalarCellLevelCB.isChecked():
            extra_fields.append("ScalarCellLevel")

        if sgd.vectorCB.isChecked():
            extra_fields.append("Vector")

        if sgd.vectorCellLevelCB.isChecked():
            extra_fields.append("VectorCellLevel")

        # adding steppable
        # will instantiate steppable templates only when needed
        if not self.steppableTemplates:
            self.steppableTemplates = SteppableTemplates()

        # figuring out if we need to add import header for steppable newly generated code
        steppable_import_header = self.steppableTemplates.generate_steppable_import_header()
        header_import_regex_list = self.steppableTemplates.get_steppable_header_import_regex_list()

        add_import_header = False
        for regex in header_import_regex_list:
            line_num, indent_level = self.find_regex_occurrence(
                regex=regex, script_editor_window=steppable_script_editor, find_first=True)

            if line_num < 0:
                add_import_header = True
                break

        entry_line, indentation_level = self.find_entry_point_for_steppable_registration(main_python_script_editor)

        steppable_code = self.steppableTemplates.generate_steppable_code(steppeble_name, frequency, steppable_type,
                                                                         extra_fields)
        if add_import_header:
            steppable_code = steppable_import_header + steppable_code

        if steppable_code == "":
            QMessageBox.warning(tw, "Problem With Steppable Generation", "Could not generate steppable")

            return

        max_line_idx = steppable_script_editor.lines()

        col = steppable_script_editor.lineLength(max_line_idx - 1)

        steppable_script_editor.insertAt(steppable_code, max_line_idx, col)

        steppable_script_editor.ensureLineVisible(max_line_idx + 20)

        # Registration of steppable

        if not main_python_script_editor:
            QMessageBox.warning(tw, "Problem with Main Python script",
                                "Please edit python main script to register steppable . "
                                "Could not open main Python script")
            return

        if entry_line == -1:
            QMessageBox.warning(tw, "Please check Python main script",
                                "Please edit python main script to register steppable . "
                                "Could not determine where to put steppeble registration code ")
            return

        steppable_registration_code = self.steppableTemplates.generate_steppable_registration_code(
            steppeble_name, frequency, basename_for_import, indentation_level,
            main_python_script_editor.indentationWidth())

        if indentation_level == -1:
            QMessageBox.warning(tw, "Possible indentation problem",

                                "Please edit python main script position properly steppable registration code ")

        main_python_script_editor.insertAt(steppable_registration_code, entry_line, 0)

        main_python_script_editor.ensureLineVisible(max_line_idx + 10)
        # steppableScriptEditorWindow
        print("ENTRY LINE FOR REGISTRATION OF STEPPABLE IS ", entry_line)

    def add_steppable_cc3d_python(self):
        """
        Inserts template steppable code into Steppable script and also registers newly created steppable in the main
        Python script. Handles clicks from CC3D Python menu

        :return:
        """

        current_file_path = self.__ui.getCurrentDocumentName()
        pdh = self.find_project_for_file(file_path=current_file_path)
        if pdh is None:
            QMessageBox.warning(self.treeWidget, "Could not find open open cc3d project",
                                f"Could find open cc3d project containing <br> "
                                f"{current_file_path}")
            return

        main_python_script = pdh.cc3dSimulationData.pythonScript.strip()

        steppable_script = None

        if self.is_python_resource_script(pdh=pdh, file_path=current_file_path):
            steppable_script = current_file_path
        else:
            try:
                # in this case we are picking first available python file from the project that is not
                # a main script.
                candidate_steppable_script = list(pdh.cc3dSimulationData.resources.values())[0].path
                if candidate_steppable_script != main_python_script:
                    steppable_script = candidate_steppable_script
            except (IndexError, AttributeError):
                pass

        if steppable_script is None:
            # we could not locate steppable file in the project
            # we will exit - in the future we could create steppable file and add it to the project in such situation
            QMessageBox.warning(self.treeWidget, "Could not open file",
                                f"Could not open steppable file")

            return

        if not main_python_script:
            # we could not locate main python script
            QMessageBox.warning(self.treeWidget, "Could not locate python main script",
                                f"Make sure your simulation has main python script")
            return

        main_python_script_editor = self.open_file_in_editor(file_path=main_python_script)
        steppable_script_editor = self.open_file_in_editor(file_path=steppable_script)

        self.insert_steppable_template(main_python_script_editor=main_python_script_editor,
                                       steppable_script_editor=steppable_script_editor)

    def add_steppable(self):
        """
        Inserts template steppable code into Steppable script and also registers newly created steppable in the main
        Python script.Handles right-clicks (context) from CC3D Project panel
        :return:
        """

        # curItem here points to Python resource file meaning it is a viable file to paste steppable
        tw = self.treeWidget
        cur_item = tw.currentItem()
        proj_item = tw.getProjectParent(cur_item)

        if not proj_item:
            return

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            return

        main_python_script_path = pdh.cc3dSimulationData.pythonScriptResource.path

        # check if the file to which we are trying to add Steppable is Python resource
        item_full_path = str(tw.getFullPath(cur_item))

        main_script_editor_window = None
        steppable_script_editor_window = None

        if main_python_script_path != "":

            self.openFileInEditor(main_python_script_path)
            editor = self.__ui.getCurrentEditor()

            if str(self.__ui.getCurrentDocumentName()) == main_python_script_path:
                main_script_editor_window = editor

        self.openFileInEditor(item_full_path)

        editor = self.__ui.getCurrentEditor()
        if str(self.__ui.getCurrentDocumentName()) == item_full_path:
            steppable_script_editor_window = editor

        if not steppable_script_editor_window:
            QMessageBox.warning(tw, "File Open Problem", "Could not open steppable file in Twedit++5-CC3D")
            return

        self.insert_steppable_template(main_python_script_editor=main_script_editor_window,
                                       steppable_script_editor=steppable_script_editor_window)

    def find_regex_occurrence(self, regex, script_editor_window, find_first=True):
        """
        Finds first regex occurrence in the editor window
        :param regex: {compiled regex}
        :param script_editor_window:{editor window}
        :param find_first:{bool} flag whether we look for first or last occurence
        :return: {tuple} line, column of regex occurence
        """

        if not script_editor_window:
            return -1, - 1

        last_line = script_editor_window.lines() - 1

        if find_first:
            line_sequence = range(last_line)
        else:
            line_sequence = range(last_line, -1, -1)

        for line_idx in line_sequence:

            line_text = script_editor_window.text(line_idx)

            main_loop_regex_found = re.match(regex, line_text)

            if main_loop_regex_found:

                print("Indentation for mainLoop line is: ", script_editor_window.indentation(
                    line_idx), " indentation width=", script_editor_window.indentationWidth())

                indentation_level = script_editor_window.indentation(
                    line_idx) // script_editor_window.indentationWidth()

                if script_editor_window.indentation(line_idx) % script_editor_window.indentationWidth():
                    # problems with indentation will used indentation 0 and inform user about the issue
                    indentation_level = -1

                return line_idx, indentation_level

        return -1, -1

    def find_entry_point_for_steppable_registration(self, main_script_editor_window):

        main_loop_regex = re.compile('^[\s]*CompuCellSetup\.run()')

        return self.find_regex_occurrence(regex=main_loop_regex,
                                          script_editor_window=main_script_editor_window,
                                          find_first=False)

    def showProjectPanel(self, _flag):

        """

            THIS SLOT WILL BE CALLED MULTIPLE TIMES AS IT IS LINKED TO TWO DIFFERENT SIGNALS - THIS IS NOT A PROBLEM IN THIS PARTICULAR CASE THOUGH

        """

        print("showProjectPanel CALLED ", _flag)

        if _flag:

            self.cc3dProjectDock.show()

        else:

            self.cc3dProjectDock.hide()

        if self.actions["Show Project Panel"].isChecked() != _flag:
            self.actions["Show Project Panel"].setChecked(_flag)

    def __runInPlayer(self):

        tw = self.treeWidget
        cur_item = tw.currentItem()

        proj_item = tw.getProjectParent(cur_item)

        if not proj_item:

            number_ofprojects = self.treeWidget.topLevelItemCount()

            if number_ofprojects == 1:

                proj_item = self.treeWidget.topLevelItem(0)

            elif number_ofprojects > 1:

                QMessageBox.warning(self.treewidget, "Please Select Project",

                                    "Please first click inside project that you wish to open in the PLayer and try again")

            else:
                return

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            return

        project_full_path = pdh.cc3dSimulationData.path

        # get CompuCell3D Twedit Plugin - it allows to start CC3D from twedit
        cc3d_plugin = self.__ui.pm.get_active_plugin("PluginCompuCell3D")

        if not cc3d_plugin:
            return

        cc3d_plugin.startCC3D(project_full_path)

    def __newCC3DProject(self):

        tw = self.treeWidget
        nsw = NewSimulationWizard(tw)

        if nsw.exec_():
            nsw.generateNewProject()

    def __displayProperties(self):

        tw = self.treeWidget
        proj_item = tw.getProjectParent(tw.currentItem())

        if not proj_item:
            return

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            return

        try:
            ild = tw.projects[qt_obj_hash(proj_item)]
        except LookupError:
            return

        try:
            resource = ild.itemToResource[qt_obj_hash(tw.currentItem())]
        except LookupError:
            return

        print("resource=", resource)

        ip = ItemProperties(self.treeWidget)

        ip.setResourceReference(resource)

        ip.updateUi()

        if ip.exec_():

            print("Changes were made")

            dirtyFlagLocal = False

            if resource.module != str(ip.moduleLE.text()):
                dirtyFlagLocal = True

            if resource.origin != str(ip.originLE.text()):
                dirtyFlagLocal = True

            if resource.copy != ip.copyCHB.isChecked():
                dirtyFlagLocal = True

            resource.module = str(ip.moduleLE.text())

            resource.origin = str(ip.originLE.text())

            resource.copy = ip.copyCHB.isChecked()

            print("resource=", resource)

            print("copy=", resource.copy)

            # set dirtyFlag to True

            try:

                self.treeWidget.projects[qt_obj_hash(proj_item)].dirtyFlag = dirtyFlagLocal

            except LookupError as e:

                pass



        else:

            print("No Changes were made")

    def __openXMLPythonInEditor(self):

        tw = self.treeWidget

        projItem = tw.getProjectParent(tw.currentItem())

        if not projItem:
            return

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            return

        print("__openXMLPythonInEditor pdh.cc3dSimulationData.xmlScript=", pdh.cc3dSimulationData.xmlScript)

        print("__openXMLPythonInEditor pdh.cc3dSimulationData.xmlScriptResource.path=",
              pdh.cc3dSimulationData.xmlScriptResource.path)

        # in order to do deeper level expansion we first have to expand top level

        projItem.setExpanded(True)

        if pdh.cc3dSimulationData.xmlScript != "":

            self.openFileInEditor(pdh.cc3dSimulationData.xmlScript)

            xmlItem = tw.getItemByText(projItem, "XML Script")

            if xmlItem:
                xmlItem.setExpanded(True)

        if pdh.cc3dSimulationData.pythonScript != "":

            self.openFileInEditor(pdh.cc3dSimulationData.pythonScript)

            pythonItem = tw.getItemByText(projItem, "Main Python Script")

            if pythonItem:
                pythonItem.setExpanded(True)

        for path, resource in pdh.cc3dSimulationData.resources.items():

            if resource.type == "Python":

                self.openFileInEditor(path)

                pythonItem = tw.getItemByText(projItem, "Python")

                if pythonItem:
                    pythonItem.setExpanded(True)

        return

    def openFileInEditor(self, _fileName=""):

        if _fileName == "":
            return

        tw = self.treeWidget

        proj_item = tw.getProjectParent(tw.currentItem())

        if not proj_item:
            return

        ild = None

        try:

            ild = tw.projects[qt_obj_hash(proj_item)]

        except LookupError as e:

            pass

        if _fileName != "":

            # we will check if current tab before and after opening new document
            # are the same (meaning an attempt to open same document twice)

            self.__ui.loadFile(_fileName)

            current_document_name = self.__ui.getCurrentDocumentName()

            current_editor = self.__ui.getCurrentEditor()

            current_editor.registerCustomContextMenu(self.createParameterScanMenu(current_editor))

            # check if opening of document was successful

            if current_document_name == _fileName:

                # next we check if _fileName is already present in self.projectLinkedTweditTabs as
                # a value and linked to tab different than currentTabWidget

                # this happens when user opens _fileName from project widget,
                # renames it in Twedit and then attempts to open _fileName again from project widget

                if ild:

                    tab_references_to_remove = []

                    for tabWidget, path in ild.projectLinkedTweditTabs.items():

                        if path == _fileName and tabWidget != current_editor:
                            tab_references_to_remove.append(tabWidget)

                    for tab in tab_references_to_remove:

                        try:

                            del ild.projectLinkedTweditTabs[tab]

                        except LookupError as e:

                            pass

                    # insert current tab and associate it with _fileName -

                    # if projectLinkedTweditTabs[currentTabWidget] is already present we will
                    # next statement is ignored - at most it changes value projectLinkedTweditTabs[currentTabWidget]

                    ild.projectLinkedTweditTabs[current_editor] = _fileName

    def __openInEditor(self):

        tw = self.treeWidget

        projItem = tw.getProjectParent(tw.currentItem())

        if not projItem:
            return

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            return

            # fileName=pdh.cc3dSimulationData.path

        idl = None

        try:

            idl = tw.projects[qt_obj_hash(projItem)]

        except LookupError as e:

            pass

        fileName = tw.getFullPath(tw.currentItem())

        if fileName != "":
            self.openFileInEditor(fileName)

    def closeProjectUsingProjItem(self, _projItem=None):

        if not _projItem:
            return

        tw = self.treeWidget

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(_projItem)]

        except LookupError as e:

            return

        fileName = pdh.cc3dSimulationData.path

        # check if project is dirty

        dirtyFlag = False

        try:

            dirtyFlag = self.treeWidget.projects[qt_obj_hash(_projItem)].dirtyFlag

        except LookupError as e:

            pass

        if dirtyFlag:

            ret = QMessageBox.warning(self.treeWidget, "Save Project Changes?",

                                      "Project was modified.<br>Do you want to save changes?",

                                      QMessageBox.Yes | QMessageBox.No)

            if ret == QMessageBox.Yes:
                self.__save_cc3d_project()

        # ask users if they want to save unsaved documents associated with the project

        # close tabs associated with the project

        idl = None

        try:

            idl = tw.projects[qt_obj_hash(_projItem)]

        except LookupError as e:

            pass

        for tab in list(idl.projectLinkedTweditTabs.keys()):
            index = tab.panel.indexOf(tab)

            self.__ui.closeTab(index, True, tab.panel)

            # self.__ui.closeTab(index)

        # remove self.treeWidget.projects[_projItem],self.treeWidget.projects[fileName] and self.projectDataHandlers[_projItem]from dictionaries

        try:

            del self.projectDataHandlers[qt_obj_hash(_projItem)]

            del self.treeWidget.projects[qt_obj_hash(_projItem)]

            del self.treeWidget.projects[fileName]



        except LookupError as e:

            pass

        for i in range(tw.topLevelItemCount()):

            if tw.topLevelItem(i) == _projItem:
                tw.takeTopLevelItem(i)

                break

                # tw.removeChild(projItem)

    def __closeProject(self):

        tw = self.treeWidget

        projItem = tw.getProjectParent(tw.currentItem())

        pathToRemove = ""

        for path, projItemLocal in self.openProjectsDict.items():

            if projItemLocal == projItem:
                pathToRemove = path

        try:

            del self.openProjectsDict[pathToRemove]

        except LookupError as e:

            pass

        self.closeProjectUsingProjItem(projItem)

    def markProjectDirty(self, projItem):

        try:

            self.treeWidget.projects[qt_obj_hash(projItem)].dirtyFlag = True

        except LookupError as e:

            pass

    def __removeResources(self):

        tw = self.treeWidget
        proj_item = tw.getProjectParent(tw.currentItem())

        if not proj_item:
            return

        ret = QMessageBox.warning(tw, "Delete Selected Items?",
                                  "Are you sure you want to delete selected items?<br>This cannot be undone.<br> "
                                  "Proceed?",
                                  QMessageBox.Yes | QMessageBox.No)

        if ret == QMessageBox.No:
            return

        try:
            ild = self.treeWidget.projects[qt_obj_hash(proj_item)]
        except LookupError:
            return

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            return

        selection = self.treeWidget.selectedItems()

        # divide the selection into type level items (e.g. items like Main Python Script, Python ,PIF File etc.)
        # and leaf items (i.e files)

        type_items = []

        leaf_items = []

        for item_tmp in selection:

            if proj_item == item_tmp.parent():
                type_items.append(item_tmp)
            elif item_tmp != proj_item:
                leaf_items.append(item_tmp)

        # first process leaf items - remove them from the project
        for item_tmp in leaf_items:
            parent = item_tmp.parent()
            pdh.cc3dSimulationData.removeResource(ild.getFullPath(item_tmp))

            if ild.getResourceName(item_tmp) == 'CC3DSerializerResource':
                pdh.cc3dSimulationData.removeSerializerResource()

            ild.removeItem(item_tmp)

            parent.removeChild(item_tmp)

            if not parent.childCount() and parent not in type_items:
                ild.removeItem(parent)

                proj_item.removeChild(parent)

        # process type_items
        for item_tmp in type_items:
            children_list = []

            for i in range(item_tmp.childCount()):
                children_list.append(item_tmp.child(i))

            for child_item in children_list:
                # pdh.cc3dSimulationData.removeResource(ild.getFullPath(item_tmp))
                pdh.cc3dSimulationData.removeResource(ild.getFullPath(child_item))
                ild.removeItem(child_item)
                item_tmp.removeChild(child_item)

            if ild.getResourceName(item_tmp) == 'CC3DSerializerResource':
                pdh.cc3dSimulationData.removeSerializerResource()

            elif ild.getResourceName(item_tmp) == 'CC3DParameterScanResource':
                pdh.cc3dSimulationData.removeParameterScanResource()

            ild.removeItem(item_tmp)

            proj_item.removeChild(item_tmp)

        # save project
        self.__save_cc3d_project()

        # # mark project as dirty
        # self.markProjectDirty(proj_item)

    def checkFileExtension(self, _extension="", _expectedExtensions=[]):

        if not len(_expectedExtensions):
            return ""

        if _extension in _expectedExtensions:

            return ""

        else:

            return _expectedExtensions[0]

    def __addResource(self):

        wz = NewFileWizard(self.treeWidget)
        if wz.exec_():
            name = wz.nameLE.text().strip()

            # dont allow empty file names
            if name == "":
                return

            file_name = os.path.basename(name)
            base, extension = os.path.splitext(file_name)
            location = str(wz.locationLE.text())

            if wz.customTypeCHB.isChecked():
                file_type = str(wz.customTypeLE.text())
            else:
                # have to replace it with dictionary
                file_type = str(wz.fileTypeCB.currentText())

                if file_type == "Main Python Script":
                    file_type = "PythonScript"
                elif file_type == "XML Script":
                    file_type = "XMLScript"
                elif file_type == "PIF File":
                    file_type = "PIFFile"
                elif file_type == "Python File":
                    file_type = "Python"
                elif file_type == "Concentration File":
                    file_type = "ScalarField"
                # check file extensions

                if file_type == "Python" or file_type == "PythonScript":
                    if extension == "":
                        name = name + '.py'
                    else:

                        suggested_extension = self.checkFileExtension(extension, ['.py', '.pyw'])

                        if suggested_extension != "":
                            ret = QMessageBox.warning(self.treeWidget, "Possible Extension Mismatch",
                                                      "Python script typically has extension <b>.py</b> .<br> "
                                                      "Your file has extension <b>%s</b> . "
                                                      "<br> Do you want to continue?" % extension,
                                                      QMessageBox.Yes | QMessageBox.No)

                            if ret == QMessageBox.No:
                                return

                if file_type == "XMLScript":

                    if extension == "":
                        name = name + '.xml'

                    else:
                        suggested_extension = self.checkFileExtension(extension, ['.xml'])
                        if suggested_extension != "":
                            ret = QMessageBox.warning(self.treeWidget, "Possible Extension Mismatch",
                                                      "XML script typically has extension <b>.xml</b> .<br> "
                                                      "Your file has extension <b>%s</b> . "
                                                      "<br> Do you want to continue?" % extension,
                                                      QMessageBox.Yes | QMessageBox.No)

                            if ret == QMessageBox.No:
                                return

                if file_type == "PIFFile":

                    if extension == "":
                        name = name + '.piff'
                    else:
                        suggested_extension = self.checkFileExtension(extension, ['.piff'])
                        if suggested_extension != "":
                            ret = QMessageBox.warning(self.treeWidget, "Possible Extension Mismatch",
                                                      "PIF File typically has extension <b>.piff</b> .<br> "
                                                      "Your file has extension <b>%s</b> ."
                                                      " <br> Do you want to continue?" % extension,
                                                      QMessageBox.Yes | QMessageBox.No)

                            if ret == QMessageBox.No:
                                return

            # extract project data handler
            tw = self.treeWidget
            proj_item = tw.getProjectParent(tw.currentItem())

            if not proj_item:
                return

            try:
                pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
            except LookupError:
                return

            # first check if location has not changed - this is a relative path w.r.t root of the simulation

            if location == "" or location == "Simulation" or location == "Simulation/":

                location = "Simulation"

                full_location = os.path.join(pdh.cc3dSimulationData.basePath, "Simulation")

            else:
                try:
                    full_location = os.path.join(pdh.cc3dSimulationData.basePath, str(location))
                    self.makeDirectory(full_location)
                except IOError:
                    print("COULD NOT MAKE DIRECTORY ", pdh.cc3dSimulationData.basePath)

                    QMessageBox.warning(self, "COULD NOT MAKE DIRECTORY",
                                        "Write permission error. You do not have write permissions to %s directory" % (
                                            pdh.cc3dSimulationData.basePath), QMessageBox.Ok)

                    return

            # check if a file exists in which case we have to copy it to current directory
            name = str(name)

            try:
                open(name)
                # if file exist we will copy it to the 'full_location' directory
                file_name = os.path.basename(name)
                resource_name = os.path.join(full_location, file_name)

                try:
                    shutil.copy(name, resource_name)
                except shutil.Error:
                    # ignore any copy errors
                    QMessageBox.warning(self.__ui, "COULD NOT COPY FILE",
                                        "Could not copy %s to %s . " % (name, full_location), QMessageBox.Ok)

            except IOError:
                # file does not exist
                try:
                    resource_name = os.path.join(full_location, name)
                    write_file(os.path.join(full_location, name), "")

                except IOError:
                    print("COULD NOT CREATE FILE")
                    QMessageBox.warning(self.__ui, "COULD NOT CREATE FILE",
                                        "Write permission error. You do not have write permissions to %s directory" % (
                                            full_location), QMessageBox.Ok)
                    return

            # Those 2 fcn calls have to be paired
            # attach new file to the project
            pdh.cc3dSimulationData.addNewResource(resource_name, file_type)
            # insert new file into the tree
            self.insertNewTreeItem(resource_name, file_type)

            # mark project as dirty
            # self.markProjectDirty(proj_item)

            # save project
            self.__save_cc3d_project()

    def __addSerializerResource(self):

        tw = self.treeWidget

        projItem = tw.getProjectParent(tw.currentItem())

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            print("could not find simulation data handler for this item")

            return

        if pdh.cc3dSimulationData.serializerResource:
            QMessageBox.warning(tw, "Serializer is already defined",

                                "You cannot have more than one serializer per simulation")

            return

        se = SerializerEdit(self.treeWidget)

        resource = self.treeWidget.getCurrentResource()

        print('resource=', resource)

        # se.setupDialog(resource)

        if se.exec_():
            pdh.cc3dSimulationData.addNewSerializerResource()  # adding empty serializer resource

            se.modifySerializerResource(pdh.cc3dSimulationData.serializerResource)

            projItem = self.treeWidget.getProjectParent(self.treeWidget.currentItem())

            self.markProjectDirty(projItem)

            # insert new file into the tree

            self.insertNewGenericResourceTreeItem(pdh.cc3dSimulationData.serializerResource)

    def insertNewGenericResourceTreeItem(self, _resource):

        projItem = self.treeWidget.getProjectParent(self.treeWidget.currentItem())

        if not projItem:
            return

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            return

        pd = pdh.cc3dSimulationData  # project data

        ild = None

        try:

            ild = self.treeWidget.projects[qt_obj_hash(projItem)]

        except:

            print("COULD NOT FIND PROJECT DATA")

            return

        if _resource.resourceName == 'CC3DSerializerResource':

            item = QTreeWidgetItem(projItem)

            item.setText(0, "Serializer")

            item.setIcon(0, QIcon(':/icons/save-simulation.png'))

            try:

                ild.insertnewGenericResource(item, _resource)

            except LookupError as e:

                # print "pd.resources[resourceName]=",pd.resources[resourceName]

                pass



        elif _resource.resourceName == 'CC3DParameterScanResource':

            # make new branch to store this item

            item = QTreeWidgetItem(projItem)

            item.setText(0, "ParameterScan")

            item.setIcon(0, QIcon(':/icons/scan_32x32.png'))

            item1 = QTreeWidgetItem(item)

            item1.setText(0, os.path.basename(_resource.path))

            try:

                ild.insertnewGenericResource(item, _resource)

                ild.insertNewItem(item1,

                                  _resource)  # file path has to be entered into ild using insertNewItem to enable proper behavior of tree widget                



            except LookupError as e:

                # print "pd.resources[resourceName]=",pd.resources[resourceName]

                pass

    def insertNewTreeItem(self, resourceName, fileType):

        # first find the node where to insert new item

        projItem = self.treeWidget.getProjectParent(self.treeWidget.currentItem())

        if not projItem:
            return

        fileNameBase = os.path.basename(resourceName)

        print("resourceName=", resourceName)

        print("fileType=", fileType)

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            return

        pd = pdh.cc3dSimulationData  # project data

        ild = None

        try:

            ild = self.treeWidget.projects[qt_obj_hash(projItem)]

        except:

            print("COULD NOT FIND PROJECT DATA")

            return

        if fileType == "PythonScript":

            typeItem = self.findTypeItemByName("Main Python Script")

            # we will replace Python script with new one

            if typeItem:

                ild.removeItem(typeItem.child(0))

                typeItem.removeChild(typeItem.child(0))

                pythonScriptItem = QTreeWidgetItem(typeItem)

                pythonScriptItem.setText(0, fileNameBase)

                ild.insertNewItem(pythonScriptItem, pd.pythonScriptResource)



            else:  # make new branch to store this item 

                pythonScriptItem = QTreeWidgetItem(projItem)

                pythonScriptItem.setText(0, "Main Python Script")

                pythonScriptItem.setIcon(0, QIcon(':/icons/python-icon.png'))

                pythonScriptItem1 = QTreeWidgetItem(pythonScriptItem)

                pythonScriptItem1.setText(0, fileNameBase)

                ild.insertNewItem(pythonScriptItem1, pd.pythonScriptResource)





        elif fileType == "XMLScript":

            typeItem = self.findTypeItemByName("XML Script")

            # we will replace XML script with new one

            if typeItem:

                ild.removeItem(typeItem.child(0))

                typeItem.removeChild(typeItem.child(0))

                xmlScriptItem = QTreeWidgetItem(typeItem)

                xmlScriptItem.setText(0, fileNameBase)

                ild.insertNewItem(xmlScriptItem, pd.xmlScriptResource)



            else:  # make new branch to store this item 

                xmlScriptItem = QTreeWidgetItem(projItem)

                xmlScriptItem.setText(0, "XML Script")

                xmlScriptItem.setIcon(0, QIcon(':/icons/xml-icon.png'))

                xmlScriptItem1 = QTreeWidgetItem(xmlScriptItem)

                xmlScriptItem1.setText(0, fileNameBase)

                ild.insertNewItem(xmlScriptItem1, pd.xmlScriptResource)



        elif fileType == "PIFFile":

            typeItem = self.findTypeItemByName("PIF File")

            # we will do not replace PIF File with new one - just add another one

            if typeItem:

                # check if new path  exists in this branch

                for i in range(typeItem.childCount()):

                    if str(ild.getFullPath(typeItem.child(i))) == str(resourceName):
                        return

                pifFileItem = QTreeWidgetItem(typeItem)

                pifFileItem.setText(0, fileNameBase)

                # check if full path exist in this branch

                try:

                    ild.insertNewItem(pifFileItem, pd.resources[resourceName])

                except LookupError as e:

                    pass



            else:  # make new branch to store this item 

                pifFileItem = QTreeWidgetItem(projItem)

                pifFileItem.setText(0, "PIF File")

                pifFileItem.setIcon(0, QIcon(':/icons/pifgen_64x64.png'))

                pifFileItem1 = QTreeWidgetItem(pifFileItem)

                pifFileItem1.setText(0, fileNameBase)

                print("PIF FILE RESOURCE=", os.path.abspath(resourceName))

                try:

                    ild.insertNewItem(pifFileItem1, pd.resources[os.path.abspath(resourceName)])

                except LookupError as e:

                    # print "pd.resources[resourceName]=",pd.resources[resourceName]

                    pass

        else:

            typeItem = self.findTypeItemByName(fileType)

            if typeItem:

                # check if new path  exists in this branch

                for i in range(typeItem.childCount()):

                    if str(ild.getFullPath(typeItem.child(i))) == str(resourceName):
                        return

                item = QTreeWidgetItem(typeItem)

                item.setText(0, fileNameBase)

                try:

                    ild.insertNewItem(item, pd.resources[resourceName])

                except LookupError as e:

                    pass



            else:  # make new branch to store this item 

                item = QTreeWidgetItem(projItem)

                item.setText(0, fileType)

                item1 = QTreeWidgetItem(item)

                item1.setText(0, fileNameBase)

                print("PIF FILE RESOURCE=", os.path.abspath(resourceName))

                try:

                    ild.insertNewItem(item1, pd.resources[os.path.abspath(resourceName)])

                except LookupError as e:

                    # print "pd.resources[resourceName]=",pd.resources[resourceName]

                    pass

    def makeDirectory(self, fullDirPath):

        """

            This fcn attmpts to make directory or if directory exists it will do nothing

        """

        # dirName=os.path.dirname(fullDirPath)

        try:

            mkpath(fullDirPath)

        except:

            raise IOError

        return

    def __save_cc3d_project(self):
        """
        writes cc3d project to disk
        :return: None
        """

        cur_item = self.treeWidget.currentItem()

        proj_item = self.treeWidget.getProjectParent(cur_item)

        if not proj_item:
            return

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            return

        file_name = pdh.cc3dSimulationData.path
        pdh.write_cc3d_file_format(file_name)

        # set dirtyFlag to False
        try:
            self.treeWidget.projects[qt_obj_hash(proj_item)].dirtyFlag = False
        except LookupError:
            pass

    def __saveCC3DProjectAs(self):

        tw = self.treeWidget

        currentProjectDir = os.path.dirname(str(self.configuration.setting("RecentProject")))

        # going one level up to open dialog in the correct location

        currentProjectDir = os.path.dirname(currentProjectDir)

        projectDirName, _ = QFileDialog.getSaveFileName(self.__ui, "Save CC3D Project...", currentProjectDir, '', '',

                                                        QFileDialog.DontConfirmOverwrite)

        projectDirName = str(projectDirName)

        # projectDirName can have extension because we are using getSaveFile, so we get rid of extension here

        projectDirName, extension = os.path.splitext(projectDirName)

        projectCoreName = os.path.basename(projectDirName)

        if str(projectDirName) == "":
            return

        curItem = self.treeWidget.currentItem()

        projItem = self.treeWidget.getProjectParent(curItem)

        if not projItem:

            numberOfprojects = self.treeWidget.topLevelItemCount()

            if numberOfprojects == 1:

                projItem = self.treeWidget.topLevelItem(0)

            elif numberOfprojects > 1:

                QMessageBox.warning(self.treewidget, "Please Select Project",

                                    "Please first click inside project that you wish to save and try again")

            else:

                return

        pdh = None

        try:

            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]

        except LookupError as e:

            return

            # note, a lot of the code here was written assuming we will not need to reopen the project

        # however it turns out that proper handling of such situation would require more refactoring so I decided to

        # close new project and reopen it (users will need to open their files) 

        # Once we refactor this plugin we can always add proper handling without requiring project to be reopened

        csd = pdh.cc3dSimulationData

        #         print 'csd.xmlScriptResource.path=',csd.xmlScriptResource.path

        fn2ew = self.__ui.getFileNameToEditorWidgetMap()

        if os.path.exists(projectDirName):

            ret = QMessageBox.warning(tw, "Directory or File %s Already Exists" % (projectCoreName),

                                      "Please choose different name for the project. Directory or file %s already exists" % (

                                          os.path.join(projectDirName, projectCoreName)), QMessageBox.Ok)

            return

        else:

            #             print 'will make new directory ', projectDirName

            os.makedirs(projectDirName)

            os.makedirs(os.path.join(projectDirName, 'Simulation'))

        coreCsdResourceNames = ['xmlScriptResource', 'pythonScriptResource', 'pifFileResource', 'windowScriptResource',

                                'serializerResource', 'parameterScanResource']

        resourceList = [getattr(csd, resourceName) for resourceName in coreCsdResourceNames]

        for resourceKey, resource in csd.resources.items():
            resourceList.append(resource)

        for resource in resourceList:

            if not resource: continue

            if resource.path.strip() == '': continue

            resourceBaseName = os.path.basename(resource.path)

            newResourcePath = os.path.join(os.path.abspath(projectDirName), 'Simulation', resourceBaseName)

            #             print 'resource=',resource

            if resource.path in list(fn2ew.keys()):

                # this means the editor with the resource is open

                editor = fn2ew[resource.path]

                self.__ui.saveFile(_fileName=newResourcePath,

                                   _editor=editor)  # if project resource is open we save it in the new location                

            else:

                # this means the editor with the resource is not open so we simply copy files

                shutil.copy(resource.path, newResourcePath)

            old_resource_path = resource.path

            resource.path = 'Simulation/' + resourceBaseName

            try:
                del pdh.cc3dSimulationData.resources[old_resource_path]
                pdh.cc3dSimulationData.resources[resource.path] = resource
            except LookupError:
                print('could not find ', old_resource_path, ' in pdh.cc3dSimulationData.resources')

        cc3d_project_file_name = os.path.join(projectDirName, projectCoreName + '.cc3d')
        pdh.write_cc3d_file_format(cc3d_project_file_name)

        # after the project has been saved we need to update path to .cc3d fine and basePath
        csd.path = cc3d_project_file_name
        csd.basePath = os.path.dirname(csd.path)

        self.closeProjectUsingProjItem(projItem)
        self.openCC3Dproject(cc3d_project_file_name)

    def __goToProjectDirectory(self):

        tw = self.treeWidget

        cur_item = self.treeWidget.currentItem()

        proj_item = self.treeWidget.getProjectParent(cur_item)

        if not proj_item:

            number_ofprojects = self.treeWidget.topLevelItemCount()

            if number_ofprojects == 1:

                proj_item = self.treeWidget.topLevelItem(0)

            elif number_ofprojects > 1:

                QMessageBox.warning(self.treewidget,
                                    "Please Select Project",
                                    "Please first click inside project that you would like "
                                    "to open in file manager try again")
            else:
                return

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            return

        csd = pdh.cc3dSimulationData
        QDesktopServices.openUrl(QUrl.fromLocalFile(csd.basePath))

    def __zip_project(self):

        """
        Zips project directory

        :return:

        """

        cur_item = self.treeWidget.currentItem()
        proj_item = self.treeWidget.getProjectParent(cur_item)

        if not proj_item:
            number_of_projects = self.treeWidget.topLevelItemCount()

            if number_of_projects == 1:
                proj_item = self.treeWidget.topLevelItem(0)

            elif number_of_projects > 1:
                QMessageBox.warning(self.treewidget, "Please Select Project",
                                    "Please first click inside project that you wish to save and try again")

            else:
                return

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(proj_item)]
        except LookupError:
            return

        csd = pdh.cc3dSimulationData

        proposed_project_zip_name = csd.basePath + '.zip'

        while True:

            zipped_filename_tmp, _ = QFileDialog.getSaveFileName(self.__ui,
                                                                 "Save Zipped CC3D Project As...",
                                                                 proposed_project_zip_name,
                                                                 "*.zip")

            if zipped_filename_tmp:

                if not os.path.isfile(zipped_filename_tmp):
                    zipped_filename = zipped_filename_tmp
                    break

            else:
                return None

        zip_archive_core_name, ext = os.path.splitext(zipped_filename)
        shutil.make_archive(zip_archive_core_name, ext[1:], csd.basePath)

        return zipped_filename

    def openCC3Dproject(self, proj_file_name):

        proj_exist = True

        proj_file_name_path = Path(proj_file_name)
        if not proj_file_name_path.exists():
            QMessageBox.warning(self.treeWidget, 'Project file missing',
                                f'Project file <br> {proj_file_name} is missing')
            self.__ui.remove_item_from_configuration_string_list(self.configuration, "RecentProjects", proj_file_name)
            return

        self.__ui.add_item_to_configuration_string_list(self.configuration, "RecentProjects", proj_file_name)

        # extract file directory name and add it to settings
        dir_name = os.path.abspath(os.path.dirname(str(proj_file_name)))
        self.__ui.add_item_to_configuration_string_list(self.configuration, "RecentProjectDirectories", dir_name)

        try:
            self.openProjectsDict[proj_file_name]
        except LookupError:

            proj_exist = False

        if proj_exist:
            proj_item = self.openProjectsDict[proj_file_name]
            self.treeWidget.setCurrentItem(proj_item)
            return

        proj_item = QTreeWidgetItem(self.treeWidget)

        proj_item.setIcon(0, QIcon(':/icons/cc3d_64x64_logo.png'))

        # store a reference to data handler in a dictionary

        self.projectDataHandlers[qt_obj_hash(proj_item)] = CC3DSimulationDataHandler(None)

        self.projectDataHandlers[qt_obj_hash(proj_item)].read_cc3_d_file_format(proj_file_name)

        # we read manually the content of the parameter spec file

        # todo - reimplement reading of parameter scan spec on project open
        # if self.projectDataHandlers[qt_obj_hash(proj_item)].cc3dSimulationData.parameterScanResource:
        #     self.projectDataHandlers[
        #         qt_obj_hash(proj_item)].cc3dSimulationData.parameterScanResource.readParameterScanSpecs()

        self.__populateCC3DProjectWidget(proj_item, proj_file_name)

        self.configuration.setSetting("RecentProject", proj_file_name)

        self.openProjectsDict[proj_file_name] = proj_item

    def showOpenProjectDialogAndLoad(self, _dir=''):
        """
        Displays CC3D project open dialog and actually opens a CC3D project
        :param _dir: default directory to open file dialog to
        :return:
        """

        allowed_extensions = "*.cc3d ; *.zip"

        file_name, _ = QFileDialog.getOpenFileName(self.__ui, "Open CC3D file...", _dir, allowed_extensions)
        file_name_path = Path(file_name)
        if file_name_path.suffix in ['.zip']:
            unzipper = Unzipper(ui=self.__ui)
            file_name = unzipper.unzip_project(file_name_path)

        # this happens when e.g. during unzipping of cc3d project we could not identify uniquely
        # a file or if we skip opening altogether

        if file_name is None or str(file_name) == '':
            return

        # normalizing filename
        file_name = os.path.abspath(str(file_name))

        self.openCC3Dproject(file_name)

    def __openCC3DProject(self):
        """
        Action that opens file dialog to open .cc3d project. Pre-populates file with most recent project open
        :return:
        """

        current_file_path = str(self.configuration.setting("RecentProject"))
        self.showOpenProjectDialogAndLoad(current_file_path)

    def open_most_recent_cc3d_project(self):
        current_file_path = str(self.configuration.setting("RecentProject"))
        if not len(current_file_path):
            self.showOpenProjectDialogAndLoad()
            return

        current_proj_pth = Path(current_file_path)
        if current_proj_pth.exists():
            self.openCC3Dproject(str(current_proj_pth))

    def findTypeItemByName(self, _typeName):

        proj_item = self.treeWidget.getProjectParent(self.treeWidget.currentItem())

        if not proj_item:
            return None

        for i in range(proj_item.childCount()):

            child_item = proj_item.child(i)

            print("childItem.text(0)=", child_item.text(0), " _typeName=", _typeName)

            if str(_typeName) == str(child_item.text(0)):
                return child_item

        return None

    def __populateCC3DProjectWidget(self, projItem, fileName):

        try:
            pdh = self.projectDataHandlers[qt_obj_hash(projItem)]
        except LookupError:
            return

        file_name_base = os.path.basename(fileName)

        ild = ItemLookupData()

        self.treeWidget.projects[fileName] = ild

        self.treeWidget.projects[qt_obj_hash(projItem)] = ild

        # store a reference to data handler in a dictionary

        projItem.setText(0, file_name_base)

        pd = pdh.cc3dSimulationData  # project data

        if pd.pythonScript != "":
            python_script_item = QTreeWidgetItem(projItem)

            python_script_item.setText(0, "Main Python Script")

            python_script_item.setIcon(0, QIcon(':/icons/python-icon.png'))

            python_script_item1 = QTreeWidgetItem(python_script_item)

            python_script_item1.setText(0, os.path.basename(pd.pythonScript))

            # python_script_item1.setIcon(0,QIcon(':/icons/python-icon.png'))

            ild.insertNewItem(python_script_item1, pd.pythonScriptResource)

        if pd.xmlScript != "":
            xml_script_item = QTreeWidgetItem(projItem)

            xml_script_item.setText(0, "XML Script")

            xml_script_item.setIcon(0, QIcon(':/icons/xml-icon.png'))

            xml_script_item1 = QTreeWidgetItem(xml_script_item)

            xml_script_item1.setText(0, os.path.basename(pd.xmlScript))

            # xml_script_item1.setIcon(0,QIcon(':/icons/xml-icon.png'))

            ild.insertNewItem(xml_script_item1, pd.xmlScriptResource)

        if pd.pifFile != "":
            pif_file_item = QTreeWidgetItem(projItem)

            pif_file_item.setText(0, "PIF File")

            pif_file_item.setIcon(0, QIcon(':/icons/pifgen_64x64.png'))

            pif_file_item1 = QTreeWidgetItem(pif_file_item)

            pif_file_item1.setText(0, os.path.basename(pd.pifFile))

            ild.insertNewItem(pif_file_item1, pd.pifFileResource)

        resource_types = {}

        # Resources
        for resource_key, resource in pd.resources.items():

            if resource.type in list(resource_types.keys()):

                parent_item = resource_types[resource.type]

                new_resource_item = QTreeWidgetItem(parent_item)
                new_resource_item.setText(0, os.path.basename(resource.path))
                ild.insertNewItem(new_resource_item, resource)

            else:

                new_resource_item = QTreeWidgetItem(projItem)

                new_resource_item.setText(0, resource.type)

                if resource.type == "Python":
                    new_resource_item.setIcon(0, QIcon(':/icons/python-icon.png'))

                # inserting parent element for given resource type to dictionary

                resource_types[resource.type] = new_resource_item

                new_resource_item1 = QTreeWidgetItem(new_resource_item)

                new_resource_item1.setText(0, os.path.basename(resource.path))

                ild.insertNewItem(new_resource_item1, resource)

        # serialization data
        if pd.serializerResource:
            serializer_item = QTreeWidgetItem(projItem)

            serializer_item.setText(0, "Serializer")

            serializer_item.setIcon(0, QIcon(':/icons/save-simulation.png'))

            ild.insertnewGenericResource(serializer_item, pd.serializerResource)

        if pd.parameterScanResource:

            ps_resource = pd.parameterScanResource

            # make new branch to store this item 

            item = QTreeWidgetItem(projItem)
            item.setText(0, "ParameterScan")
            item.setIcon(0, QIcon(':/icons/scan_32x32.png'))
            item1 = QTreeWidgetItem(item)
            item1.setText(0, os.path.basename(ps_resource.path))
            try:
                # file path has to be entered into ild using insertNewItem to enable proper behavior of tree widget
                ild.insertnewGenericResource(item, ps_resource)
                ild.insertNewItem(item1, ps_resource)
            except LookupError:
                pass
