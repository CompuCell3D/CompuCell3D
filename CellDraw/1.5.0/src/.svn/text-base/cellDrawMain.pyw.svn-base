#!/usr/bin/env python
#

# ------------------------------------------------------------
# 2011 - Mitja: CellDraw (1.6.0 in the works) new features:
#   - scene bundles (*.cc3s + "Resources/")
#   - separate cell types (get them from CC3D prefs)
#     from cell regions
# ------------------------------------------------------------


# ------------------------------------------------------------
# 2011 - Mitja: CellDraw (1.5.0) new features:
#   - PIF_Generator becomes CellDraw
#   - moving all globals set by Control Panel GUI elements
#     to the cdPreferences.py file, so that they're saved
#     with the preferences, automatically.
#   - removing GUI from preferences window and place to
#     front panels/windows GUI
# ------------------------------------------------------------




# ------------------------------------------------------------
# 2011 - Mitja: (1.4.0) new features:
#   - further simplified GUI layout with control panel & PIFF table separate from main window
#   - main window has image layer on top of cell scene editor
#   - cut/copy/paste of cell regions
#   - outline corner resizing of regions
# ------------------------------------------------------------




# ------------------------------------------------------------
# 2011 - Mitja: (1.3.0) new features:
#   - new GUI layout with image layer on top of cell scene editor
#   - cut/copy/paste of cell regions
#   - outline corner resizing of regions
# ------------------------------------------------------------


# ------------------------------------------------------------
# 2010 - Mitja: (1.2.1) new features:
#   better PIFF input, interactive polygon drawing for regions
# ------------------------------------------------------------


# ------------------------------------------------------------
# 2010 - Mitja: (1.2) now this is going to be a primarily vector-based PIFF scene generator
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2010 - Mitja: (1.1) trying to build upon the existing code and GUI
#   1. changing pixmap scan, from sampling on a spaced grid
#      to sampling each in the input image
#   2. cleaning input for naming regions of cells
#   3. PIFF output from 1. and 2. above
#   4. adding external widget CDTableOfTypes from file cdTableOfTypes.py
#      to edit region names and add cell types per region (TODO for PIFF output)
#   5. filters (user selectable) to not save white- and black-colored regions
#   6. rasterizer to create (10 x 10) cells in PIFF file
# ------------------------------------------------------------

# ------------------------------------------------------------
# PIF_Generator 1.0 - written by ??? in 2008 (or 2009???)
# The original code contained no comments and no code documentation at all.
# ------------------------------------------------------------



# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
import sys
import os

def setVTKPaths():
    from os import environ
    import string
    platform=sys.platform
    if platform=='win32':
        # the VTK paths seem to be necessary on MS-Windows for CC3D,
        #   no need for them in CellDraw:
        # sys.path.append(environ["VTKPATH"])
        # sys.path.append(environ["VTKPATH1"])
        sys.path.append(environ["PYTHON_DEPS_PATH"])
        # sys.path.append(environ["SIP_PATH"])
        # sys.path.append(environ["SIP_UTILS_PATH"])
        # the PLAYERPATH wasn't necessary on MS-Windows for CC3D,
        #   there seem to be a need for it in CellDraw:
        sys.path.append(environ["PLAYERPATH"])
#   else:
#      swig_path_list=string.split(environ["VTKPATH"])
#      for swig_path in swig_path_list:
#         sys.path.append(swig_path)

# print "PATH=",sys.path
setVTKPaths()
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--



import sys, pdb, random

import math # for sqrt(), abs(), etc.

# 2011 - Mitja: to set environment variables for VTK use, better write this into a shell startup file:
# setenv MI_VTK_PATH ..../vtk/VTK_5.4.2_bin_and_build
# setenv PYTHONPATH ${MI_VTK_PATH}/Wrapping/Python:${MI_VTK_PATH}/bin
# setenv DYLD_LIBRARY_PATH ${MI_VTK_PATH}/bin
# 2011 - Mitja: to set environment variables for VTK use, the following is not enough
#    because it doesn't set the DYLD_LIBRARY_PATH:
# sys.path.append("..../vtk/VTK_5.4.2_bin_and_build/Wrapping/Python")
# sys.path.append("..../vtk/VTK_5.4.2_bin_and_build/bin")

# testing DICOM files input is commented out in PIF Generator version 1.4.0 until fix:
import vtk  # for vtkDICOMImageReader() so that we can read DICOM files
            # and for vtkDICOMImageReader() to read multi-page TIFF images

# 2010 - Mitja: importing from PyQt4 here only works because - either:
#   A. both PyQt4 and sip are already put in the system-wide Python install
# or:
#   B1. PyQt4 is in form of package within the directory where this script is run
#      (as it happens with CC3D, where the PyQt4 distribution is a directory
#       within the "player/" directory, where the main CC3D Python player script
#       gets started) *and*
#   B2. sip is also within the directory where this script is run
#       Both these requirements (B1. and B2.) are OK also if we place symlinks
#       from the current directory to wherever PyQt4 and sip are installed.
from PyQt4 import QtCore, QtGui

# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--



from ui_cellDrawGUI import Ui_CellDrawGUI

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

# 2010 - Mitja: external class for interactive table of region/cell type data:
from cdTableOfTypes import CDTableOfTypes

# 2010 - Mitja: external class for saving PIFF files from a graphics scene:
from cdSceneRasterizer import CDSceneRasterizer

# 2010 - Mitja: external class for creating/displaying a QGraphicsScene based on the input image:
from cdDiagramScene import CDDiagramSceneMainWidget

# 2010 - Mitja: external class for user preferences/settings:
from cdPreferences import CDPreferences

# 2010 - Mitja: simple external class for drawing a progress bar widget:
from cdWaitProgressBar import CDWaitProgressBar

# 2010 - Mitja: external class for scene bundle:
from cdSceneBundle import CDSceneBundle



# ----------------------------------------------------------------------
# 2010- Mitja: this is the main application window for the CellDraw
# ----------------------------------------------------------------------
class PIFInputMainWindow(QtGui.QMainWindow):


    # ------------------------------------------------------------------
    def __init__(self, parent=None):
    # main window and application initialization:
    # ------------------------------------------------------------------
        QtGui.QMainWindow.__init__(self, parent)
       
        self.theMainWindow = self
#
#         # 2010 - Mitja: apparently, the application is not set up yet:
#         self.finishedSetup = False
#         # 2010 - Mitja: in PIF Generator 1.0, the so-called "application setup" was
#         #   considered complete only after a pixmap has been loaded from a file and
#         #   analyzed, in the collectColors() function called by the openImage() function.
#         #   In PIF Generator 1.2.x and newer, we set finishedSetup to True
#         #   when the table widget and related color globals are done initializing.


        #  __init__ (1) - set up the main window's GUI,
        #  as built in Qt Designer and converted into python:
        # ---------------------------------------------------------
        #
        # 2010 - Mitja: the entire GUI for this application is built in "Qt Designer"
        #  which outputs a file with .ui extension, then converted to a python file using
        #  the PyQt "" command, thus: "pyuic4 cellDrawGUI.ui -o ui_cellDrawGUI.py".
        self.ui = Ui_CellDrawGUI()
        self.ui.setupUi(self)


        # 2010 - Mitja: set icons for application and window
        if sys.platform=='win32':
            # -->  -->  --> mswat code added for MS Windows --> -->  -->
            QApplication.setWindowIcon(QIcon("CellDraw/icons/pifgen_64x64.png"))
            # <--  <--  <-- mswat code added for MS Windows <-- <--  <--
        elif sys.platform=='darwin':
            QApplication.setWindowIcon(QIcon(":/icons/CellDraw.png"))
            self.setWindowIcon(QIcon(":/icons/CellDraw.png"))


#
#         # 2010 - Mitja: hide the dock widget containing zoom etc. pixmap-related functionalities,
#         #   to be implemented in the scene instead:
#         self.ui.futureFunctionalitiesDockWidget.hide()

        # 2010 - Mitja: hide the input controls dock widget containing picking functionalities,
        #   since below we move its (pixmap-related prefs) content widget to the preferences QDialog:
        self.ui.inputControlsContentsWidget.hide()
        self.ui.inputControlsDockWidget.setWidget(None)
        self.ui.inputControlsDockWidget.hide()



        #  __init__ (2) - assign preferred values to some globals:
        # ---------------------------------------------------------
        #
        # 2010 - Mitja:
        #    read some persistent-value globals from the preferences file on disk, if it already exists.
        #    Pass 'self' so that the main window becomes the parent window of the CDPreferences dialog:
        self.cdPreferences = CDPreferences(self.theMainWindow)
        self.cdPreferences.hide()
        self.cdPreferences.readPreferencesFromDisk()

        # explicitly connect the "cdPreferencesChangedSignal()" signal from the
        #   cdPreferences object, to our "slot" method "handlePreferencesChanged()"
        #   so that it will respond to any change in preferences:
        answer = self.connect(self.cdPreferences, \
                              QtCore.SIGNAL("cdPreferencesChangedSignal()"), \
                              self.handlePreferencesChanged )

        # 2010 - Mitja: some global variables are defined here:
        self.scaleFactor = 1.0
#         self.scaleXBox = 1.0
#         self.scaleYBox = 1.0
        self.sameScaleXYCheckBox = False

        # 2010 - Mitja: add functionalities for ignoring white / black regions when saving the PIFF file:
        self.ignoreWhiteRegionsForPIF = False
        self.ignoreBlackRegionsForPIF = False

        # 2010 - Mitja: add functionality for saving the PIFF file from the rasterized data:
        #    This was used for PIFF generation from a QPixmap. Since we now go through a polygon-based scene,
        #    the PIFRasterizer-related code is not necessary here anymore:
        # self.thePIFMustBeSavedFromRasterizedData = False

        # 2010 - Mitja: add functionality for saving the PIFF file directly from the GraphicsScene:
        #    (this global was previously used for testing overlaying the rasterized image on the top of the original image)
        self.thePIFMustBeSavedFromTheGraphicsScene = False


        # 2010 - Mitja: add functionality for saving PIFF metadata:
        self.savePIFMetadata = False

        # 2010 - Mitja: add functionality for picking a color region:
        self.pickColorRegion = False
        #               this always remains True since we only pick colors as paths (polygons) now:
        self.pickColorAsPath = True


        #  __init__ (3) - show the main window's GUI:
        # ---------------------------------------------------------
        #
        # on Mac OS X, make the window look just a bit less non-native:
        #   setUnifiedTitleAndToolBarOnMac() takes the toolbar placed in the TopToolBarArea
        #   and unify it with the main window's title bar, but it *doesn't* do the same
        #   with toolbars placed elsewhere (why? patchy Qt implementation?)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.show()
        self.raise_()
        self.move(300, 60)


        self.setMinimumSize(256, 256)
        # setGeometry is inherited from QWidget, taking 4 arguments:
        #   x,y  of the top-left corner of the QWidget, from top-left of screen
        #   w,h  of the QWidget
        self.resize(590,590)

        # the following is only useful to fix random placement at initialization
        #   *if* we use this panel as stand-alone, without including it in windows etc.
        #   These are X,Y *screen* coordinates (INCLUDING menu bar, etc.),
        #   where X,Y=0,0 is the top-left corner of the screen:
        pos = self.pos()
        pos.setX(280)
        pos.setY(30)
        self.move(pos)



        # 2010 - Mitja:  pixmap-related picking functionalities widget go in a QToolBox,
        #    into the input pixmap widget (in older code this was in the input controls dock widget)
        self.ui.inputControlsContentsWidget.show()
        self.ui.inputControlsContentsWidget.setPalette(QtGui.QPalette(QtGui.QColor(QtCore.Qt.lightGray)))
        self.ui.inputControlsContentsWidget.setAutoFillBackground(True)
        self.cdPreferences.addMorePreferencesWidget(self.ui.inputControlsContentsWidget)

#         self.inputControlsToolBox = QtGui.QToolBox()
#         self.inputControlsToolBox.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Ignored))
#         self.inputControlsToolBox.setMinimumWidth(self.ui.inputControlsContentsWidget.minimumWidth() + 32)
#         self.inputControlsToolBox.addItem(self.ui.inputControlsContentsWidget, "Input Image Controls")
#
#         self.inputImageWindow.layout().addWidget(self.inputControlsToolBox)

        # 2010 - Mitja: removing MDI from the QMainWindow altogether:
        #   there could be a centralWidget containing all QWidget panes, in theCentralLayout:
        # 2010 - Mitja: add the newly created inputImageWindow to the window's mdiArea
        #    (the mdiArea is defined in Qt Designer-generated files) :
        # self.ui.mdiArea.addSubWindow(self.inputImageWindow)
        # self.theCentralLayout.addWidget(self.inputImageWindow, 2, 1)
        # 2010 - Mitja: but we'll use a QSplitter to place the input image widget (label)
        #   together with the scene widget both in the central widget area of the main window.



        #  __init__ (5) - assign preferred values to some more globals:
        # --------------------------------------------------------------
        #
       
        # TODO: remove all these widget-baesd values from the UI file, until the UI file is not necessary anymore

        # 2010 - Mitja: some controls within the QMainWindow are set here:
#         self.ui.xDimEdit.setReadOnly(True)
#         self.ui.yDimEdit.setReadOnly(True)
#         self.ui.xScaleEdit.setReadOnly(True)
#         self.ui.yScaleEdit.setReadOnly(True)
        # 2010 - Mitja: HIDE all this since it's not used in the new code:
#         self.ui.xDimEdit.hide()
#         self.ui.yDimEdit.hide()
#         self.ui.xScaleEdit.hide()
#         self.ui.yScaleEdit.hide()
#         self.ui.scaleXBox.hide()
#         self.ui.scaleYBox.hide()
#         self.ui.sameScale_XY_checkBox.hide()
#         self.ui.label.hide()
#         self.ui.label_2.hide()
#         self.ui.label_3.hide()
#         self.ui.label_4.hide()
#         self.ui.label_5.hide()
#         self.ui.label_9.hide()
#         self.ui.label_10.hide()
#         self.ui.label_10.hide()
       
        # 2010 - Mitja: this area provides some help for the input image:
        # self.ui.textBrowser.hide()



        # 2010 - Mitja: some more global variables are defined here:

        # JUJu JUJU JUJU TODO TODO TODO TODO check how these globals are used:
        # ALL have to be set from the scene and the table now, and NOT from the pixmap
        # (those functions ought to be marked "image or pixmap only - legacy - unused")

        self.colorIds = [0]       # colorIds = a list of all RGBA values of colors present in scene items
        self.colorDict = {}       # colorDict = a dict of all region names, one for each RGBA color: name(color)
        self.comboDict = {}       # comboDict = a dict (unused?) of region INTs, one for each RGBA color: int(color)
        # 2010 - Mitja: (1.1) support for data from pixmap to external table widget:
        self.regionsTableDict = {}
        # 2010 - Mitja: (1.2) support for region names from external table widget:
        self.nameToColorDict = {} # nameToColorDict = a dict of all region colors, one for each region name: color(name)



        #  __init__ (6) - prepare a default empty lBoringPixMap:
        # -----------------------------------------------------------------------------
        #
        # 2010 - Mitja: at start, only show a boring single-color pixmap as the input image:
        lBoringPixMap = QtGui.QPixmap(self.cdPreferences.pifSceneWidth, self.cdPreferences.pifSceneHeight)
        lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )


        #  __init__ (7) - set up the main editable graphics scene widget:
        # -----------------------------------------------------------------------------
        #
        # 2010 - Mitja: external class for creating/displaying a QGraphicsScene based on the input image:
        self.graphicsSceneWindow = CDDiagramSceneMainWidget(self)

        # connect theSceneRasterizerWidget to the only instance of the cdPreferences object:
        self.graphicsSceneWindow.setPreferencesObject(self.cdPreferences)

        self.graphicsSceneWindow.scene.setSceneRect(QtCore.QRectF(0, 0, self.cdPreferences.pifSceneWidth, self.cdPreferences.pifSceneHeight))
        self.graphicsSceneWindow.updateSceneRectSize()


        # DIH continue cleaning up GUI here:
        #
        # 2010 - Mitja - create a CDSceneRasterizer object (including its visible widget)
        self.theSceneRasterizerWidget = CDSceneRasterizer(self)
        self.theSceneRasterizerWidget.hide()

        # connect theSceneRasterizerWidget to the only instance of the cdPreferences object:
        self.theSceneRasterizerWidget.setPreferencesObject(self.cdPreferences)

        # 2011 - Mitja: add input image handling directly to the QGraphicsScene,
        #   into a cdImageLayer to be drawn as foreground/overlay to the scene items:
        self.graphicsSceneWindow.cdImageLayer.width = self.cdPreferences.pifSceneWidth
        self.graphicsSceneWindow.cdImageLayer.height = self.cdPreferences.pifSceneHeight
        self.graphicsSceneWindow.cdImageLayer.pifSceneWidth = self.cdPreferences.pifSceneWidth
        self.graphicsSceneWindow.cdImageLayer.pifSceneHeight = self.cdPreferences.pifSceneHeight




        #  __init__ (8) - set up the GUI to control table of regions values:
        # -----------------------------------------------------------------------------
        #

        # 2010 - Mitja added a second window providing an interactive table
        #   containing all regions/colors found in the pixmap. If "self" is passed as parameter,
        #   when the QApplication exits, it'll signal this window to close as well:
        self.theTableOfTypes = CDTableOfTypes(self.graphicsSceneWindow)


        # explicitly connect the "regionsTableChangedSignal()" signal from the
        #   theTableOfTypes object, to our "slot" (i.e. handler) method
        #   so that it will respond to any change in table contents:
        answer = self.connect(self.theTableOfTypes, \
                              QtCore.SIGNAL("regionsTableChangedSignal()"), \
                              self.handleRegionsTableWidgetChanged )

        # prepare the default dict of regions and cell types:
        # each entry consists of a region's QColor, region's name, subtable for cell types, list of cell sizes and whether it's in use in the cell scene:
        self.regionsTableDict = dict({ 1: [ QtGui.QColor(QtCore.Qt.green), "green", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.green), "greenType", 1.0, 100]  ]   ], \
                                       2: [ QtGui.QColor(QtCore.Qt.blue), "blue", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.blue), "blueType", 1.0, 100]  ]   ], \
                                       3: [ QtGui.QColor(QtCore.Qt.red), "red", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.red), "redType", 1.0, 100]  ]   ], \
                                       4: [ QtGui.QColor(QtCore.Qt.darkYellow), "darkYellow", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.darkYellow), "darkYellowType", 1.0, 100]  ]   ], \
                                       5: [ QtGui.QColor(QtCore.Qt.lightGray), "lightGray", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.lightGray), "lightGrayType", 1.0, 100]  ]   ], \
                                       6: [ QtGui.QColor(QtCore.Qt.magenta), "magenta", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.magenta), "magentaType", 1.0, 100]  ]   ], \
                                       7: [ QtGui.QColor(QtCore.Qt.darkBlue), "darkBlue", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.darkBlue), "darkBlueType", 1.0, 100]  ]   ], \
                                       8: [ QtGui.QColor(QtCore.Qt.cyan), "cyan", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.cyan), "cyanType", 1.0, 100]  ]   ], \
                                       9: [ QtGui.QColor(QtCore.Qt.darkGreen), "darkGreen", [10, 10, 1], 0, \
                                                [  [QtGui.QColor(QtCore.Qt.darkGreen), "darkGreenType", 1.0, 100]  ]   ]   }  )


        # from CC3D colors = [QtCore.Qt.green, QtCore.Qt.blue, QtCore.Qt.red, QtCore.Qt.darkYellow, QtCore.Qt.lightGray, QtCore.Qt.magenta, QtCore.Qt.darkBlue, QtCore.Qt.cyan, QtCore.Qt.darkGreen]

        self.theTableOfTypes.setRegionsDict(self.regionsTableDict)
        self.theTableOfTypes.populateTableWithRegionsDict()











        self.graphicsSceneWindow.cdImageLayer.setImageLoadedFromFile(False)

        # now that both the regionsTableDict and the graphicsSceneWindow are initialized, add starting blank data:
        self.setImageGlobals( lBoringPixMap.toImage() )





        #  __init__ (9) - set global variable default values,
        #     and their related GUI widget defaults:
        # -----------------------------------------------------------------------------
        #
        # 2010 - Mitja: finally, manually set some defaults in GUI widgets,
        #   to make sure they're at their most commonly used state:
        self.ui.ignoreWhiteRegions_checkBox.setChecked(True)
        self.setIgnoreWhiteRegions(True)

        self.ui.ignoreBlackRegions_checkBox.setChecked(True)
        self.setIgnoreBlackRegions(True)

        self.ui.graphicsScenePIFF_checkBox.setChecked(True)
        self.setSavePIFFromGraphicsScene(True)

        self.ui.pickColorRegion_checkBox.setChecked(True)
        self.setPickColorRegion(True)

        self.ui.graphicsScenePIFF_saveMetadataCheckbox.setChecked(False)
        self.setSavePIFMetadata(False)



        #  __init__ (10) - connect application signals and slots:
        # -----------------------------------------------------------------------------
        #
        # 2010 - Mitja: the createActions() function is defined down below,
        #      and it connects signals and slots together:
        self.createActions()


        # explicitly connect the "signalVisibilityPIFRegionTable()" signal from the
        #   graphicsSceneWindow object, to our "slot" (i.e. handler) method
        #   so that it will respond to request to show/hide the PIFF region table:
        self.graphicsSceneWindow.signalVisibilityPIFRegionTable.connect( \
            self.handleTogglePIFRegionTableWindow )


        # connect the slot/callback handlers in cdPreferences() to signals:
        self.cdPreferences.connectSignalsToHandlers()


        #  __init__ (11) - finally set the main window,
        #                  then show a modal QDialog for the graphics scene preferences:
        # -----------------------------------------------------------------------------

        # 2011 - Mitja: do version checking of Qt and PyQt libraries, and stop
        #   running CellDraw if the installed versions are too old to support our code:
        self.versionWarning()

        self.setCentralWidget(self.graphicsSceneWindow)
       
        # if we get here, it means that we can use the available Qt and PyQt libraries:
        self.showPreferencesDialog(False)


        # if we get here, it means that we can use the available Qt and PyQt libraries:
        self.showPreferencesDialog(False)

    # ------------------------------------------------------------------
    # ----- end of init() -----
    # ------------------------------------------------------------------




    # ----------------------------------------------------------------------
    # 2011- Mitja: this is a little warning QDialog to stop running if the
    #   installed Qt and PyQt versions are too old to support our code
    # ----------------------------------------------------------------------
    def versionWarning(self):
        # we need at least PyQt 4.7.0 which means
        #   a value for PyQt4.QtCore.PYQT_VERSION of at least 263936 == 0x040700
        self.minPyQt = 263936
        self.minPyQtStr = "4.7.0"
        # we need at least Qt 4.6.2
        self.minQtStr = "4.6.2"

        # print "-------------------------------------------------------------------------------"
        # print "-------------------------------------------------------------------------------"
        # print "CellDraw using PyQt version PYQT_VERSION_STR =", PyQt4.QtCore.PYQT_VERSION_STR
        # print "CellDraw using PyQt version PYQT_VERSION =", PyQt4.QtCore.PYQT_VERSION
        # print "-------------------------------------------------------------------------------"
        # print "CellDraw using Qt runtime version qVersion()=", QtCore.qVersion()
        # print "CellDraw using Qt compile-time version QT_VERSION_STR=", QtCore.QT_VERSION_STR
        # print "-------------------------------------------------------------------------------"
        # print "-------------------------------------------------------------------------------"

        if (self.minPyQt > PyQt4.QtCore.PYQT_VERSION) or (self.minQtStr > QtCore.qVersion()):
            lVersionWarning = QtGui.QMessageBox.critical( self, \
                "CellDraw", \
                "CellDraw needs the following Qt and PyQt libraries to run:\n\n Qt %s or newer\n PyQt %s or newer\n\nDetected versions are:\n\n Qt runtime version: %s\n Qt compile-time version: %s\n PyQt version: %s (%s = 0x%06x).\n\nPlease contact your system administrator or the source where you obtained CellDraw.\n\nThis program will now exit." % \
                (self.minQtStr, self.minPyQtStr, QtCore.QT_VERSION_STR, QtCore.qVersion(), PyQt4.QtCore.PYQT_VERSION_STR, PyQt4.QtCore.PYQT_VERSION, PyQt4.QtCore.PYQT_VERSION) )
            sys.exit()

        print "___ - DEBUG ----- CDPreferences: versionWarning(): done"



    # ---------------------------------------------------------
    # 2010 - Mitja - this brings up the preferences dialog:
    # ---------------------------------------------------------
    def showPreferencesDialog(self, pShowMorePrefs=True):
        self.cdPreferences.setModal(True)
        if pShowMorePrefs is True:
            self.cdPreferences.setWindowTitle("CellDraw - Preferences")
        else:
            self.cdPreferences.setWindowTitle("CellDraw - Set Cell Scene Dimensions")
        self.cdPreferences.showMorePrefs(pShowMorePrefs)
        self.cdPreferences.show()
        self.cdPreferences.raise_()





    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle "content change" events
    #    (AKA signals) arriving from the object cdPreferences
    # ------------------------------------------------------------------
    def handlePreferencesChanged(self):
        print "handlePreferencesChanged(self) -- # SLOT function for the signal cdPreferencesChangedSignal() from cdPreferences"
        #
        #   here we retrieve the updated values from preferences and update CellDraw globals:
        self.cdPreferences.readPreferencesFromDisk()
        # 2010 - Mitja: get some default global values from the preferences:
        self.graphicsSceneWindow.cdImageLayer.pifSceneWidth = self.cdPreferences.pifSceneWidth
        self.graphicsSceneWindow.cdImageLayer.pifSceneHeight = self.cdPreferences.pifSceneHeight

        # propagate the change upstream, to all data structures!
        # here we update the PIFF graphics scene dimensions to the newly changed width/height values:
        self.graphicsSceneWindow.scene.setSceneRect(QtCore.QRectF(0, 0, \
                             self.cdPreferences.pifSceneWidth, self.cdPreferences.pifSceneHeight))
        self.graphicsSceneWindow.scene.mySceneUnits = self.cdPreferences.pifSceneUnits
        self.graphicsSceneWindow.updateSceneRectSize()
       
        # if and *only* if there is no image loaded from a picture file on disk,
        #   then resize the blank placeholder image accordingly:
        if self.graphicsSceneWindow.cdImageLayer.imageLoadedFromFile is False:
            lBoringPixMap = QtGui.QPixmap(self.cdPreferences.pifSceneWidth, self.cdPreferences.pifSceneHeight)
            lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.transparent) )

            # now confirm that the cdImageLayer does not contain an actual image loaded from a file:
            self.graphicsSceneWindow.cdImageLayer.setImageLoadedFromFile(False)
   
            # now that both the regionsTableDict and the graphicsSceneWindow are initialized, add starting blank data:
            self.setImageGlobals( lBoringPixMap.toImage() )



        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()

        print "    self.cdPreferences. "
        print "    self.cdPreferences. "
        print "    self.cdPreferences. "
        print "    self.cdPreferences. "
        print "    self.cdPreferences. "
        print "      self.graphicsSceneWindow.cdImageLayer.height =", self.graphicsSceneWindow.cdImageLayer.height
        print "      self.graphicsSceneWindow.cdImageLayer.width =", self.graphicsSceneWindow.cdImageLayer.width
        print "    self.cdPreferences. "
        print "      self.cdPreferences.pifSceneWidth =", self.cdPreferences.pifSceneWidth
        print "      self.cdPreferences.pifSceneHeight =", self.cdPreferences.pifSceneHeight
        print "      self.cdPreferences.pifSceneDepth =", self.cdPreferences.pifSceneDepth
        print "      self.cdPreferences.pifSceneUnits =", self.cdPreferences.pifSceneUnits
        print "    self.cdPreferences. "
        print "    self.cdPreferences. "
        print "    self.cdPreferences. "
        print "    self.cdPreferences. "
        print "___ - DEBUG ----- PIFInputMainWindow: handlePreferencesChanged() done."






    # ------------------------------------------------------------------
    # 2011 Mitja - this is a slot method to handle "toggle" events
    #    (AKA signals) arriving to hide or show the CDTableOfTypes
    # ------------------------------------------------------------------
    def handleTogglePIFRegionTableWindow(self, pString="Toggle"):
        if pString == "Toggle":       
            if self.theTableOfTypes.isHidden():
#                 print "self.theTableOfTypes.show() now!"
                self.theTableOfTypes.show()
            else:
#                 print "self.theTableOfTypes.hide() now!"
                self.theTableOfTypes.hide()
        elif pString == "Hide":
#             print "self.theTableOfTypes.hide() now!"
            self.theTableOfTypes.hide()
        elif pString == "Show":
#             print "self.theTableOfTypes.show() now!"
            self.theTableOfTypes.show()
        else:
            print "handleTogglePIFRegionTableWindow does not know what to do!"

        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()


    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle "content change" events
    #    (AKA signals) arriving from the object CDTableOfTypes
    # ------------------------------------------------------------------
    def handleRegionsTableWidgetChanged(self):
        # SLOT function for the signal "regionsTableChangedSignal() from CDTableOfTypes"
        #
        #   here we retrieve the table contents to update regionsTableDict:
        self.regionsTableDict = self.theTableOfTypes.getRegionsDict()
        lKeys = self.regionsTableDict.keys()
        print "2010 DEBUG: in handleRegionsTableWidgetChanged() the regionsTableDict is :"
        print "                  regionsTableDict keys =", lKeys
        for i in xrange(len(self.regionsTableDict)):
            print "                  regionsTableDict: i, lKeys[i], regionsTableDict[keys[i]], self.regionsTableDict[lKeys[i]][0].rgba() = ", \
                  i, lKeys[i], self.regionsTableDict[lKeys[i]], self.regionsTableDict[lKeys[i]][0].rgba()
            # update all global data structures to the new values just provided by the external table widget:
            # colorIds remains the same, since the external table doesn't change colors:
            #    self.colorIds
            # colorDict needs to be updated to new names:
            self.colorDict[self.regionsTableDict[lKeys[i]][0].rgba()] = self.regionsTableDict[lKeys[i]][1]
            # comboDict remains the same, since the external table doesn't change colorIds values:
            #    self.comboDict = {}

            # nameToColorDict = a dict of all region colors, one for each region name: color(name)
            self.nameToColorDict[self.regionsTableDict[lKeys[i]][1]] = self.regionsTableDict[lKeys[i]][0].rgba()
            # nameToColorDict also has to contain all type colors, one for each type name: color(name)
            for j in xrange(len(self.regionsTableDict[lKeys[i]][4])):
                print " -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_     at i =", i, ", j =", j, "self.regionsTableDict[lKeys[i]][4][j] =", \
                      self.regionsTableDict[lKeys[i]][4][j]
                self.nameToColorDict[ self.regionsTableDict[lKeys[i]][4][j][1] ] = self.regionsTableDict[lKeys[i]][4][j][0].rgba()

        print "2010 DEBUG: handleRegionsTableWidgetChanged() with globals:", \
            "\n ___ - DEBUG ----- self.colorIds =", self.colorIds, \
            "\n ___ - DEBUG ----- self.colorDict =", self.colorDict, \
            "\n ___ - DEBUG ----- self.comboDict =", self.comboDict, \
            "\n ___ - DEBUG ----- self.nameToColorDict =", self.nameToColorDict, \
            "\n ___ - DEBUG ----- self.regionsTableDict =", self.regionsTableDict
        print "___ - DEBUG ----- PIFInputMainWindow: handleRegionsTableWidgetChanged() ....."


        # propagate the change upstream, to all data structures!

        # update the theSceneRasterizerWidget with new region data:
        self.theSceneRasterizerWidget.setRegionsDict(self.regionsTableDict)

        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()

        print "___ - DEBUG ----- PIFInputMainWindow: handleRegionsTableWidgetChanged() done."



    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # now define fuctions that handle input from GUI checkbox widgets:
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle "content change" events
    #    (AKA signals) arriving from self.ui.ignoreWhiteRegions_checkBox
    # ------------------------------------------------------------------
    def setIgnoreWhiteRegions(self, pCheckBox):
        self.ignoreWhiteRegionsForPIF = pCheckBox
        self.theSceneRasterizerWidget.setIgnoreWhiteRegions(self.ignoreWhiteRegionsForPIF)
        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()
        print ">>>>>>>>>>>>>>>>>>>>>>>> PIFInputMainWindow.ignoreWhiteRegionsForPIF is now =", self.ignoreWhiteRegionsForPIF

    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle "content change" events
    #    (AKA signals) arriving from self.ui.ignoreBlackRegions_checkBox
    # ------------------------------------------------------------------
    def setIgnoreBlackRegions(self, pCheckBox):
        self.ignoreBlackRegionsForPIF = pCheckBox
        self.graphicsSceneWindow.scene.update()
        self.theSceneRasterizerWidget.setIgnoreBlackRegions(self.ignoreBlackRegionsForPIF)
        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()
        print ">>>>>>>>>>>>>>>>>>>>>>>> PIFInputMainWindow.ignoreBlackRegionsForPIF is now =", self.ignoreBlackRegionsForPIF


    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle "content change" events
    #    (AKA signals) arriving from self.ui.graphicsScenePIFF_checkBox
    # ------------------------------------------------------------------
    def setSavePIFFromGraphicsScene(self, pCheckBox):
        self.thePIFMustBeSavedFromTheGraphicsScene = pCheckBox
        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()
        print ">>>>>>>>>>>>>>>>>>>>>>>> PIFInputMainWindow.thePIFMustBeSavedFromTheGraphicsScene is now =", self.thePIFMustBeSavedFromTheGraphicsScene



    # ------------------------------------------------------------------
    # 2010 Mitja - this is a slot method to handle "content change" events
    #    (AKA signals) arriving from self.ui.pickColorRegion_checkBox
    # ------------------------------------------------------------------
    def setPickColorRegion(self, pCheckBox):
        self.pickColorRegion = pCheckBox

        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()

        print ">>>>>>>>>>>>>>>>>>>>>>>> PIFInputMainWindow.pickColorRegion is now =", self.pickColorRegion


    # ------------------------------------------------------------------
    # 2010 - Mitja: add functionality for saving PIFF metadata:
    # 2010 Mitja - this is a slot method to handle "content change" events
    #    (AKA signals) arriving from self.ui.graphicsScenePIFF_saveMetadataCheckbox
    # ------------------------------------------------------------------
    def setSavePIFMetadata(self, pCheckBox):
        self.savePIFMetadata = pCheckBox

        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()

        self.theSceneRasterizerWidget.setSavePIFMetadata(self.savePIFMetadata)
        print ">>>>>>>>>>>>>>>>>>>>>>>> PIFInputMainWindow.savePIFMetadata is now =", self.savePIFMetadata




    # ---------------------------------------------------------
    # collectAllColorsInTable() collects all colors present in table widget,
    #     as source for possible PIFF regions (colors).  We'll have TODO to synchronize the
    #     diagram/scene and the table widget for adding colors.
    # ---------------------------------------------------------
    def collectAllColorsInTable(self):
        # 2010 - Mitja: this function collects *all* color values in the external table widget

        #   here we retrieve the table contents to update the global regionsTableDict:
        self.regionsTableDict = self.theTableOfTypes.getRegionsDict()
       
        lKeys = self.regionsTableDict.keys()
        print "2010 DEBUG:   in   collectAllColorsInTable() the regionsTableDict is :"
        print "                   regionsTableDict keys =", lKeys

        # this is how the regionsTableDict used to be created:
        # self.regionsTableDict[k] = [QtGui.QColor(lColor), self.colorDict[lColor], \
        #                            [ [QtGui.QColor(lColor).darker(150), "type"+str(k), 1.0, 100] ] ]

        for i in xrange(len(self.regionsTableDict)):
            lColor = self.regionsTableDict[lKeys[i]][0].rgba()
            print "                  regionsTableDict: i, lKeys[i], regionsTableDict[keys[i]], self.regionsTableDict[lKeys[i]][0].rgba() = ", \
                  i, lKeys[i], self.regionsTableDict[lKeys[i]], lColor
            #
            # update all global data structures to the new values just provided by the external table widget:
            #
            # 2010 - Mitja: add new color to the global colorIDs (list of colors) :
            if lColor not in self.colorIds:
                self.colorIds.append(lColor)

            # 2010 - Mitja: add new color to the global colorDict (dict of colors<->names) :
            #   (in the original code, all these were assigned the name "empty")
            self.colorDict[lColor] = self.regionsTableDict[lKeys[i]][1]

            # nameToColorDict = a dict of all region colors, one for each region name: color(name)
            self.nameToColorDict[self.regionsTableDict[lKeys[i]][1]] = self.regionsTableDict[lKeys[i]][0].rgba()
            # nameToColorDict also has to contain all type colors, one for each type name: color(name)
            for j in xrange(len(self.regionsTableDict[lKeys[i]][4])):
                self.nameToColorDict[ self.regionsTableDict[lKeys[i]][4][j][1] ] = self.regionsTableDict[lKeys[i]][4][j][0].rgba()


        # 2010 - Mitja: for unknown reason, the global "colorIds" is initialized with
        #   a single element containing the value 0, which now needs to be removed:
        try:
            self.colorIds.remove(0)
        except:
            print ">>>>>>>>>>>>>>>>>>>>>>>> self.colorIds does not include 0 :", self.colorIds


        # 2010 - Mitja: add content to some widgets in the GUI,
        #   and to even more global lists:
        i = 0
        for key in self.colorDict.keys():
            # 2010 - Mitja: add a count to yet another global comboDict (dict of colors<->ints ?) :
            self.comboDict[key] = i
            i += 1

#
#         # 2010 - Mitja: the so-called "application setup" is considered complete now:
#         self.finishedSetup = True

        # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
        self.graphicsSceneWindow.scene.update()

        print "2010 DEBUG: collectAllColorsInTable() DONE with globals:", \
              "\n self.colorIds =", self.colorIds, \
              "\n self.colorDict =", self.colorDict, \
              "\n self.comboDict =", self.comboDict, \
              "\n self.nameToColorDict =", self.nameToColorDict, \
              "\n self.regionsTableDict =", self.regionsTableDict
    # end of def collectAllColorsInTable(self)
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    def openImage(self):

        print "2010 DEBUG: openImage() STARTING with globals:", \
              "\n self.colorIds =", self.colorIds, \
              "\n self.colorDict =", self.colorDict, \
              "\n self.comboDict =", self.comboDict, \
              "\n self.regionsTableDict =", self.regionsTableDict

        # 2010 - Mitja: according to Qt documentation, the tr() function returns a
        #   translated version of its input parameter, optionally based on a
        #   disambiguation string and value of n for strings containing plurals;
        #   otherwise it returns the input parameter itself if no appropriate
        #   translated string is available:
        lFileName = QtGui.QFileDialog.getOpenFileName(self, \
            self.tr("CellDraw - Open Image File"), \
            QtCore.QDir.currentPath())

        if not lFileName.isEmpty():

            # 2010 - Mitja: load the file's data into a QImage object:
            lImage = QtGui.QImage(lFileName)

            if lImage.isNull():
                QtGui.QMessageBox.warning( self, self.tr("CellDraw"), \
                    self.tr("Cannot open the image file: " \
                    "%1").arg(lFileName) )
            else:
                print "2010 DEBUG: _,.- ~*'`'*~-.,_ openImage() got [", lFileName, "]"

                # now confirm that the cdImageLayer contains an actual image loaded from a file:
                self.graphicsSceneWindow.cdImageLayer.setImageLoadedFromFile(True)

                # update all input-image-related globals:
                self.setImageGlobals( lImage )

                # 2010 - Mitja: also pass the loaded image as background brush for the external QGraphicsScene window:
                #
                # but this TILES the image onto the "brush" background image.
                # instead, draw the pixmap onto an image, and use that image for the brush....:
                #  brushes are always tiled.
                #    You could create an image of (width(),height()),
                #    draw your pixmap in the center of it and use that for the brush.
                #    This way it will tile, but only once.
                #    Besure to update the pixmap on resize!
                lThePath,lTheFileName = os.path.split(str(lFileName))
                print "2010 DEBUG 2010 DEBUG now calling self.graphicsSceneWindow.updateBackgroundImage(lTheFileName, lImage) with ", lTheFileName, lImage
                self.graphicsSceneWindow.updateBackgroundImage(lTheFileName, lImage)

            # 2011 - Mitja: and ask for a redraw of the cdImageLayer:
            self.graphicsSceneWindow.scene.update()

    # end of def openImage(self)
    # ---------------------------------------------------------




    # ---------------------------------------------------------
    # 2010 - Mitja: "setImageGlobals()" sets image-related globals every time a new image
    #   is loaded from a picture image file from disk.
    #   It derives from the original "openImage()" function which has been now split into
    #   the new "openImage()" function which only obtains an image from a file, and this
    #   setImageGlobals() which deals with updating globals from a new image's data.
    # ---------------------------------------------------------
    #   Mandatory input parameter to setImageGlobals() : one QImage object.
    # ---------------------------------------------------------
    def setImageGlobals(self, pImage):

        # 2010 - Mitja: note that (as per Qt documentation) a QLabel may contain a QPixmap object, but not a QImage.
        #   Thus the image instance we assign to cdImageLayer from pImage is a separate object:
        self.graphicsSceneWindow.cdImageLayer.setImage(pImage)


        # 2010 - Mitja: sample the color of every pixel in the pixmap
        #  then store all found colors in the global "colorDict"
        #  (in the original code this was done on a sampling grid, by
        #   calling self.collectColors() instead)
        #
        # Colors are collected from the external table now
        #  TODO they ought to be maybe updated if adding more colors to the polygon scene?
        self.collectAllColorsInTable()


#         # 2010 - Mitja: calling updateActions() simply enables zooming in/out
#         #   on the pixmap, by activating these two entries in the "View" menu:
#         # TODO2 - this function call has to be updated to actions related to the polygon scene now!
#         self.updateActions()




        print "2010 DEBUG: _,.- ~*'`'*~-.,_ setImageGlobals() with image: ", str(pImage), " ... done."



    # ---------------------------------------------------------
    def fileOpenImage_Callback(self):
        self.openImage()

    # ---------------------------------------------------------
    # 2010 - Mitja: the fileSavePIFFromScene_Callback() function is activated by
    #   the action_Export_PIFF from the "File" menu and its keyboard shortcut:
    # ---------------------------------------------------------
    def fileSavePIFFromScene_Callback(self):

        print "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() starting."

        # 2010 - Mitja: setup local variables for file saving:
        lToBeSavedFileExtension = QtCore.QString("piff")
        lToBeSavedInitialPath = QtCore.QDir.currentPath() + self.tr("/untitled.") + lToBeSavedFileExtension

        # 2010 - Mitja: save PIFF file directly from the GraphicsScene:
        #
        if self.thePIFMustBeSavedFromTheGraphicsScene is True:
            # (thePIFMustBeSavedFromTheGraphicsScene was used (previously, in PIF Generator 1.1)
            # for testing the overlaying of the rasterized image on the top of the original image)

            # update the theSceneRasterizerWidget with new data:
            #
            #   1. retrieve the PIFF table of types contents to update regionsTableDict:
            self.handleRegionsTableWidgetChanged()
            self.theSceneRasterizerWidget.setRegionsDict(self.regionsTableDict)
            self.theSceneRasterizerWidget.setRasterWidth(self.cdPreferences.piffFixedRasterWidth)
            self.theSceneRasterizerWidget.setIgnoreWhiteRegions(self.ignoreWhiteRegionsForPIF)
            self.theSceneRasterizerWidget.setIgnoreBlackRegions(self.ignoreBlackRegionsForPIF)
            #   2. update the graphics scene data:
            self.theSceneRasterizerWidget.setInputGraphicsScene(self.graphicsSceneWindow.scene)


            # this used to be set to the width&height of the scene contents.
            # But we now set PIFF width&height values in preferences, separately from the scene contents, so we don't set it thus:
            # self.theSceneRasterizerWidget.resize(self.graphicsSceneWindow.scene.width(), self.graphicsSceneWindow.scene.height())
            # Instead we set it from preferences:
            self.theSceneRasterizerWidget.resize((100+self.cdPreferences.pifSceneWidth), (100+self.cdPreferences.pifSceneHeight))

            self.theSceneRasterizerWidget.show()
            self.theSceneRasterizerWidget.raise_()

            # the class global keeping track of the selected PIFF generation mode:
            #    CDConstants.PIFFSaveWithFixedRaster,
            #    CDConstants.PIFFSaveWithOneRasterPerRegion,
            #    CDConstants.PIFFSaveWithPotts = range(3)
            if self.cdPreferences.piffGenerationMode == CDConstants.PIFFSaveWithFixedRaster:
                # ----------------------------------------------------
                # 2010 - Mitja: generate PIFF content with *** fixed *** size cells:
                #   (set all PIFF cell regions to be generated with same-sized square cells,
                #    as from the raster size set in CellDraw preferences)
                # ----------------------------------------------------
   
                self.theSceneRasterizerWidget.rasterizeSceneToFixedSizeRaster()

                fileName = QtGui.QFileDialog.getSaveFileName(self, self.tr("CellDraw - Save fixed-raster PIFF as"),
                                       lToBeSavedInitialPath,
                                       self.tr("%1 files (*.%2);;All files (*)")
                                           .arg(lToBeSavedFileExtension.toUpper())
                                           .arg(lToBeSavedFileExtension))
                if fileName.isEmpty():
                    self.theSceneRasterizerWidget.hide()
                    return False
                    print "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() fixed-size raster PIFF failed."
                else:
                    # DIH:
                    self.theSceneRasterizerWidget.savePIFFFileFromFixedSizeRaster(fileName)
                    self.theSceneRasterizerWidget.hide()
                    print "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() fixed-size raster PIFF done."
                    return True

            elif self.cdPreferences.piffGenerationMode == CDConstants.PIFFSaveWithPotts:
                # ----------------------------------------------------
                # saving to PIFF different-sized cells for each region:
                # ----------------------------------------------------

                # here we use the Potts algorithm to generate each region of cells:
                self.theSceneRasterizerWidget.computePottsModelAndSavePIF()
                self.theSceneRasterizerWidget.hide()
                print "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() Potts-generated PIFF done."
                return True
                # ----------------------------------------------------------------

            else:
                # ----------------------------------------------------
                # here we rasterize each object separately, *** region-raster *** ("variable") size cells:
                # ----------------------------------------------------

                # 2011 - Mitja: first step towards 3D PIFF scenes:
                self.theSceneRasterizerWidget.rasterizeSceneToRegionRasters()

                fileName = QtGui.QFileDialog.getSaveFileName(self, self.tr("CellDraw - Save region-raster PIFF as"),
                                       lToBeSavedInitialPath,
                                       self.tr("%1 files (*.%2);;All files (*)")
                                           .arg(lToBeSavedFileExtension.toUpper())
                                           .arg(lToBeSavedFileExtension))
                if fileName.isEmpty():
                    self.theSceneRasterizerWidget.hide()
                    print "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() region-raster PIFF failed."
                    return False
                else:
                    # this used to be the call to rasterize region-raster cells from each region:
                    #   self.theSceneRasterizerWidget.attemptAtRasterizingVariableSizeCellsAndSavePIF(fileName)
                    # this used to be another old-style region raster ("variable")-sized cell rasteization:
                    #   self.theSceneRasterizerWidget.rasterizeVarSizedCellRegionsAndSavePIF(fileName)
                    self.theSceneRasterizerWidget.savePIFFFileFromRegionRasters(fileName)
                    self.theSceneRasterizerWidget.hide()
                    print "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() region-raster PIFF done."
                    return True
                # ----------------------------------------------------------------
               

        else:
            # ----------------------------------------------------------------
            fileName = QtGui.QFileDialog.getSaveFileName(self, self.tr("CellDraw - Save per-pixel PIFF as"),
                                   lToBeSavedInitialPath,
                                   self.tr("%1 files (*.%2);;All files (*)")
                                       .arg(lToBeSavedFileExtension.toUpper())
                                       .arg(lToBeSavedFileExtension))
            if fileName.isEmpty():
                self.theSceneRasterizerWidget.hide()
                print "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() pixel-based PIFF failed."
                return False
            else:
                # 2010 - Mitja: this was originally calling the saveFile function:
                #   self.saveFile(fileName)
                self.savePIFFileFromAllPixels(fileName)
                self.theSceneRasterizerWidget.hide()
                print "___ - DEBUG ----- PIFInputMainWindow: fileSavePIFFromScene_Callback() pixel-based PIFF done."
                return True




    # ---------------------------------------------------------
    # 2010 - Mitja:
    #     savePIFFileFromAllPixels() samples the pixmap at every pixel, and then saves
    #     a PIFF file with region labels taken from the table (table labels part TODO)
    # ---------------------------------------------------------
    def savePIFFileFromAllPixels(self, fileName):
        # (the savePIFFileFromAllPixels() function is based
        #    on the original saveFile() function from PIF Generator 1.0)

        #   2011 - TODOTODOTODO place this in cdImageLayer too!!!
       
        file = QtCore.QFile(fileName)
        if not file.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            QtGui.QMessageBox.warning(self, self.tr("CellDraw"), \
                self.tr("Saving per-pixel PIFF file...\mcan not write file %1:\n%2.").arg(fileName).arg(file.errorString()))
            print "___ - DEBUG ----- PIFInputMainWindow: savePIFFileFromAllPixels() failed."
            return False

        # 2010 - Mitja: TODO TODO TODO
        #   here we retrieve the table contents to an update regionsTableDict:
        self.regionsTableDict = self.theTableOfTypes.getRegionsDict()
        lKeys = self.regionsTableDict.keys()
        print "2010 DEBUG DEBUG DEBUG DEBUG: in savePIFFileFromAllPixels() the regionsTableDict is :"
        print "                  regionsTableDict keys =", lKeys
        for i in xrange(len(self.regionsTableDict)):
            print "                  regionsTableDict: i, lKeys[i], regionsTableDict[keys[i]] = ", \
                  i, lKeys[i], self.regionsTableDict[lKeys[i]]




        # 2010 - Mitja: open a QTextStream, i.e. an "interface for reading and writing text"
        lOutputStream = QtCore.QTextStream(file)

        # 2010 - Mitja: show the user that the application is busy (while writing to a file)
        #  by changing the mouse cursor to a "wait" shape:
        # 2011 - Mitja: this doesn't always restore to normal (on different platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # 2010 - Mitja: sample and save *all* cell values from the pixmap image
        #   (not just those at "fixedRasterWidth" intervals)
        lSampleInterval = 1
        lWidth = self.graphicsSceneWindow.cdImageLayer.width
        lHeight = self.graphicsSceneWindow.cdImageLayer.height

        lCellID = 0

        # 2010 - Mitja: python's xrange function is more appropriate for large loops
        #   since it generates integers (the range function generates lists instead)
        for i in xrange(0, lWidth, lSampleInterval):
            for j in xrange(0, lHeight, lSampleInterval):
                xoffset = i
                yoffset = j

                # 2010 - Mitja: sample the pixmap to obtain the color value at i,j:
                lColor = self.graphicsSceneWindow.cdImageLayer.image.pixel(xoffset,yoffset)

                lCellType = self.colorDict[lColor]
                # 2010 - Mitja: the simplest type of save function is implemented as
                #   "save each pixel as another cell type" :
                xmin = int(i)
                xmax = int(i)
                ymin = int(j)
                ymax = int(j)
                # xmin = (int)(i*self.scaleXBox)
                # xmax = xmin+lSampleInterval-1
                # ymin = (int)(j*self.scaleYBox)
                # ymax = ymin+lSampleInterval-1

                # 2010 - Mitja: add functionalities for ignoring white / black regions when saving the PIFF file:
                if  (self.ignoreWhiteRegionsForPIF == True) and (lColor == QtGui.QColor(QtCore.Qt.white).rgba()):
            # print ">>>>>>>>>>>>>>>>>>>>>>>> lColor, QtGui.QColor(QtCore.Qt.white) =", lColor, QtGui.QColor(QtCore.Qt.white).rgba()
                    pass # do nothing
                elif (self.ignoreBlackRegionsForPIF == True) and (lColor == QtGui.QColor(QtCore.Qt.black).rgba()) :
                    # print ">>>>>>>>>>>>>>>>>>>>>>>> lColor, QtGui.QColor(QtCore.Qt.black) =", lColor, QtGui.QColor(QtCore.Qt.black).rgba()
                    pass # do nothing
                else :
                    lOutputStream << "%s %s %s %s %s %s 0 0\n"%(lCellID,lCellType,xmin,xmax,ymin,ymax)
                    lCellID +=1

        print "___ - DEBUG ----- PIFInputMainWindow: savePIFFileFromAllPixels() done."

        # 2010 - Mitja: stop showing that the application is busy (while writing to a file)
        #   and undo the last setOverrideCursor(), i.e. set the mouse cursor to what it was before:
        # 2011 - Mitja: this doesn't always restore to normal (on different platrforms?) so we don't change cursor for now:
        # QtGui.QApplication.restoreOverrideCursor()


    # end of def savePIFFileFromAllPixels(self, fileName)
    # ---------------------------------------------------------






    # ---------------------------------------------------------
    # 2010 - Mitja: callback for menu item to "Save Scene"
    # ---------------------------------------------------------
    def fileSaveScene_Callback(self):
        print "2010 DEBUG: fileSaveScene_Callback() STARTING with globals:", \
              "\n self.colorIds =", self.colorIds, \
              "\n self.colorDict =", self.colorDict, \
              "\n self.comboDict =", self.comboDict, \
              "\n self.regionsTableDict =", self.regionsTableDict

        self.graphicsSceneWindow.saveSceneFile()


    # end of def fileSaveScene_Callback(self)
    # ---------------------------------------------------------





    # ---------------------------------------------------------
    # 2011 - Mitja: callback for menu item for "New Scene"
    # ---------------------------------------------------------
    def fileNewScene_Callback(self):
        print "2011 DEBUG: fileNewScene_Callback() STARTING with globals:", \
              "\n self.colorIds =", self.colorIds, \
              "\n self.colorDict =", self.colorDict, \
              "\n self.comboDict =", self.comboDict, \
              "\n self.regionsTableDict =", self.regionsTableDict

        self.graphicsSceneWindow.newSceneFile()


    # end of def fileNewScene_Callback(self)
    # ---------------------------------------------------------




    # ---------------------------------------------------------
    # 2010 - Mitja: callback for menu item to "Open Scene"
    # ---------------------------------------------------------
    def fileOpenScene_Callback(self):
        CDConstants.printOut(  "2010 DEBUG: fileOpenScene_Callback() STARTING with globals:"+ \
              "\n self.colorIds ="+ str(self.colorIds)+ \
              "\n self.colorDict ="+ str(self.colorDict)+ \
              "\n self.comboDict ="+ str(self.comboDict)+ \
              "\n self.regionsTableDict ="+ str(self.regionsTableDict), CDConstants.DebugAll )

        lFileDialog = QtGui.QFileDialog(self.theMainWindow)
        lFilters = "Scene Bundle (*."+CDConstants.SceneBundleFileExtension+");;Scene File (*.pifScene)"

        #         lFilters.append("Scene Bundle (*.cc3s)")
        #         lFilters.append("Scene File (*.pifScene)")
        #         # lFilters.append("any file (*)")

        lFileDialog.setNameFilters(lFilters);
        lFileName = lFileDialog.getOpenFileName(self, self.tr("CellDraw - Open Scene"), \
            QtCore.QDir.currentPath(), self.tr(lFilters) )

        if not lFileName.isEmpty():

            # check if the user selected a .cc3s file or a .pifScene file:
            lFileExtension = os.path.splitext(str(lFileName))[1]
            CDConstants.printOut(  "fileOpenScene_Callback() : lFileExtension = " + str(lFileExtension), CDConstants.DebugAll )

            if (lFileExtension == "."+CDConstants.SceneBundleFileExtension):

                self.cdSceneBundle = CDSceneBundle(self.theMainWindow)
                if (self.cdSceneBundle.openSceneBundleFile(lFileName) == True):
                    lTheSceneFileName = self.cdSceneBundle.getSceneFileName()
                    if (lTheSceneFileName != ""):
                        CDConstants.printOut(  "fileOpenScene_Callback() : lTheSceneFileName = " + str(lTheSceneFileName), CDConstants.DebugAll )
                        self.graphicsSceneWindow.openSceneFile(lTheSceneFileName)
                    lThePIFFFileName = self.cdSceneBundle.getPIFFFileName()
                    if (lThePIFFFileName != ""):
                        CDConstants.printOut(  "fileOpenScene_Callback() : lThePIFFFileName = " + str(lThePIFFFileName), CDConstants.DebugAll )
                        self.openPIFFFile(lThePIFFFileName)
                
            elif (lFileExtension == ".pifScene"):
                self.graphicsSceneWindow.openSceneFile(lFileName)

#         CDConstants.printOut(  "fileOpenScene_Callback() : lFileDialog.selectedNameFilter() = " + str(lFileDialog.selectedNameFilter()), CDConstants.DebugAll )
#         CDConstants.printOut(  "fileOpenScene_Callback() : lFileDialog.selectedFiles()[0] = " + str(lFileDialog.selectedFiles()[0]), CDConstants.DebugAll )
#         CDConstants.printOut(  "os.path.splitext(str(lFileName))[1] = " + str(os.path.splitext(str(lFileName))[1]), CDConstants.DebugAll )

    # end of def fileOpenScene_Callback(self)
    # ---------------------------------------------------------


    # ---------------------------------------------------------
    # 2010 - Mitja: callback for menu item to "Open PIFF"
    # ---------------------------------------------------------
    def openPIFFile_Callback(self):
        CDConstants.printOut(  "2011 DEBUG: openPIFFile_Callback() STARTING with globals:"+ \
              "\n self.colorIds ="+ str(self.colorIds)+ \
              "\n self.colorDict ="+ str(self.colorDict)+ \
              "\n self.comboDict ="+ str(self.comboDict)+ \
              "\n self.regionsTableDict ="+ str(self.regionsTableDict), CDConstants.DebugAll )

        lFileName = QtGui.QFileDialog.getOpenFileName(self, \
            self.tr("CellDraw - Open PIFF File"), \
            QtCore.QDir.currentPath(), self.tr("*.piff"))

        if not lFileName.isEmpty():
            self.openPIFFFile(lFileName)

    # end of def openPIFFile_Callback(self)
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    # 2010 - Mitja: Open a PIFF file and parse it into an image:
    # ---------------------------------------------------------
    def openPIFFFile(self, pFileName):
        CDConstants.printOut(  "2011 DEBUG: openPIFFFile("+str(pFileName)+") STARTING.", CDConstants.DebugExcessive )

        if not QtCore.QString(pFileName).isEmpty():
            # 2010 - Mitja: load the file's data into a QImage object:
            lThePIFInputFile = QtCore.QFile(pFileName)
            lFileOK = lThePIFInputFile.open(QtCore.QIODevice.ReadOnly)

            if lFileOK is False:
                QtGui.QMessageBox.warning( self, self.tr("CellDraw"), \
                    self.tr("Cannot open the PIFF file:\n" \
                    "%1").arg(pFileName) )
            else:

                # show a panel containing a progress bar:
                lProgressBarPanel = CDWaitProgressBar("Loading image data from PIFF file, pass 1 of 2.", 100, self.theMainWindow)
                lProgressBarPanel.show()

                lThePIFText = QtCore.QTextStream(lThePIFInputFile).readAll()
                # print " "
                # print " QtCore.QTextStream(lThePIFInputFile).readAll() = lThePIFText =", lThePIFText
                lThePIFText.replace("\r\n", "\n")
                lThePIFText.replace("\r", "\n")
                lThePIFTextList = lThePIFText.split("\n")
                lThePIFTextNumberOfLines = len(lThePIFTextList)
                lProgressBarPanel.setRange(0,lThePIFTextNumberOfLines)
                # print " "
                # print "  lThePIFTextList =", lThePIFTextList
                # print " "
                i = 0
               
                # check PIFF coordinate boundaries:
                #
                xMin = +9999
                xMax = -9999
                yMin = +9999
                yMax = -9999
                for lThePIFTextLine in lThePIFTextList:
                    lProgressBarPanel.setValue(i)
                    # if we have a lThePIFTextLine, it doesn't necessarily follow
                    #   that it's a well-formed PIFF line... so we better use "try - except" :
                    try:                   
                        # print "lThePIFTextLine", i, "is <",  lThePIFTextLine, ">"
                        thePIFlineList = lThePIFTextLine.split(" ", QtCore.QString.SkipEmptyParts)
                        pifXMin = int(thePIFlineList[2])
                        pifXMax = int(thePIFlineList[3])
                        pifYMin = int(thePIFlineList[4])
                        pifYMax = int(thePIFlineList[5])
           
                        if ( pifXMin < xMin ):
                            xMin = pifXMin
                        if ( pifYMin < yMin ):
                            yMin = pifYMin
                        if ( pifXMax > xMax ):
                            xMax = pifXMax
                        if ( pifYMax > yMax ):
                            yMax = pifYMax
                        # print "xMin =", xMin, " yMin =", yMin, "xMax =", xMax, " yMax =", yMax
                        j = 0
                        for word in thePIFlineList:
                            # print "word", j, "is <",  word, "> is <", thePIFlineList[j], ">"
                            j = j + 1                       
                        i = i + 1
                    except:
                        # we got exception in parsing a PIFF line, just do nothing.
                        pass

                # now that we got PIFF boundaries, build an image from the PIFF:
                #

                lBackgroundRect = QtCore.QRectF( QtCore.QRect(0, 0, xMax, yMax) )
                lPixmap = QtGui.QPixmap(lBackgroundRect.width(), lBackgroundRect.height())
                lPixmap.fill( QtGui.QColor(QtCore.Qt.white) )
                lPainter = QtGui.QPainter(lPixmap)

                # close the first panel containing a progress bar:
                lProgressBarPanel.maxProgressBar()
                lProgressBarPanel.accept()

                # show a second panel containing a progress bar:
                lProgressBarPanel = CDWaitProgressBar("Loading image data from PIFF file, pass 2 of 2.", 100, self.theMainWindow)
                lProgressBarPanel.show()
                lProgressBarPanel.setRange(0,lThePIFTextNumberOfLines)

                i = 0
                lNameToColorDictKeys = self.nameToColorDict.keys()
                for lThePIFTextLine in lThePIFTextList:
                    lProgressBarPanel.setValue(i)
                    # if we have a lThePIFTextLine, it doesn't necessarily follow
                    #   that it's a well-formed PIFF line... so we better use "try - except" :
                    try:                   
                        thePIFlineList = lThePIFTextLine.split(" ", QtCore.QString.SkipEmptyParts)
                        pifXMin = int(thePIFlineList[2])
                        pifXMax = int(thePIFlineList[3])
                        # pifYMin = int(thePIFlineList[4])
                        # pifYMax = int(thePIFlineList[5])
                        # invert Y values here, from RHS to LHS:
                        pifYMin = ((yMax-1) - int(thePIFlineList[5]))
                        pifYMax = ((yMax-1) - int(thePIFlineList[4]))
                        lTmpRect = QtCore.QRectF( QtCore.QRect(pifXMin, pifYMin, pifXMax-pifXMin+1, pifYMax-pifYMin+1) )
                        lTmpPixmap  = QtGui.QPixmap( lTmpRect.width(), lTmpRect.height() )
                        # if the PIFF line's cell type name is in our dict, assign the related color, otherwise keep in a new different color:
                        lColor = QtCore.Qt.darkCyan
                        for j in xrange(len(self.nameToColorDict)):
                            # print "lNameToColorDictKeys[j] is thePIFlineList[1] =", lNameToColorDictKeys[j], thePIFlineList[1]
                            if (str(lNameToColorDictKeys[j]) == str(thePIFlineList[1])):
                                # print "YUHUHU lNameToColorDictKeys[j], self.nameToColorDict[lNameToColorDictKeys[j]],  QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[j]]) =", lNameToColorDictKeys[j], self.nameToColorDict[lNameToColorDictKeys[j]],  QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[j]])
                                lColor = QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[j]])
                        lTmpPixmap.fill( lColor )
                        lPainter.drawPixmap(QtCore.QPoint(pifXMin,pifYMin), lTmpPixmap)
                        i = i + 1
                    except:
                        # we got exception in parsing a PIFF line, just do nothing.
                        pass
                lPainter.end()

                # create another pixmap of the same size as the graphics scene rectangle, and fill it with the chosen background pattern:
                image = lPixmap.toImage()

                lProgressBarPanel.maxProgressBar()
                lProgressBarPanel.accept()


                print "2010 DEBUG: _,.- ~*'`'*~-.,_ openPIFFFile() converted [", pFileName, "] PIFF into image."

                # now confirm that the cdImageLayer contains an actual image loaded from a file:
                self.graphicsSceneWindow.cdImageLayer.setImageLoadedFromFile(True)

                self.setImageGlobals( image )
               
                # 2010 - Mitja: also pass the loaded image as background brush for the external QGraphicsScene window:
                #
                # but this TILES the image onto the "brush" background image.
                # instead, draw the pixmap onto an image, and use that image for the brush....:
                #  brushes are always tiled.
                #    You could create an image of (width(),height()),
                #    draw your pixmap in the center of it and use that for the brush.
                #    This way it will tile, but only once.
                #    Besure to update the pixmap on resize!
                lThePath,lTheFileName = os.path.split(str(pFileName))
                print "2010 DEBUG 2010 DEBUG now calling self.graphicsSceneWindow.updateBackgroundImage(lTheFileName, image) = (", lTheFileName, image, ")."
                self.graphicsSceneWindow.updateBackgroundImage(lTheFileName, image)


    # end of def openPIFFFile(self)
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    # 2010 - Mitja: callback for menu item to "Open DICOM File"
    # ---------------------------------------------------------
    # simple DICOM files import:
    def openDICOMFile_Callback(self):
   
        try:
            vtk.vtkDICOMImageReader()
        except:
            lVtkMissingWarning = QtGui.QMessageBox.critical( self, \
            "CellDraw", \
            "CellDraw needs VTK libraries to read DICOM files.\n\nIf you need to open DICOM files in CellDraw, please contact your system administrator or the source where you obtained CellDraw.")
            return


        print "-.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-"
        print "2010 DEBUG: openDICOMFile_Callback() STARTING with globals:", \
              "\n self.colorIds =", self.colorIds, \
              "\n self.colorDict =", self.colorDict, \
              "\n self.comboDict =", self.comboDict, \
              "\n self.regionsTableDict =", self.regionsTableDict

        lFileName = QtGui.QFileDialog.getOpenFileName(self, self.tr("CellDraw - Open DICOM File"), \
                       QtCore.QDir.currentPath(), self.tr("*.dcm"))

        if not lFileName.isEmpty():
            # 2010 - Mitja: load the file's data into a VTK object:
#             lThePIFInputFile = QtCore.QFile(lFileName)
#             lFileOK = lThePIFInputFile.open(QtCore.QIODevice.ReadOnly)

            reader = vtk.vtkDICOMImageReader()
            reader.SetFileName(str(lFileName))
            reader.Update() # <-- important <-- otherwise the VTK pipeline is not updated <--
            image_point_data = reader.GetOutput().GetPointData()
            print "-.-  -.-  -.-  -.-  GetPointData == image_point_data:   -.-  -.-  -.-  -.-"
            print image_point_data
           
            scalarptr = reader.GetOutput().GetScalarPointer(1, 1, 0)
            print "-.-  -.-  -.-  -.-  GetScalarPointer == scalarptr:   -.-  -.-  -.-  -.-"
            print scalarptr
            floatsomething = reader.GetOutput().GetScalarComponentAsFloat(1, 1, 0, 0)
            print "-.-  -.-  -.-  -.-  GetScalarComponentAsFloat == floatsomething:   -.-  -.-  -.-  -.-"
            print floatsomething

            dimensions = reader.GetOutput().GetDimensions()
            xMax = dimensions[0]
            yMax = dimensions[1]
            zMax = dimensions[2]
            print "-.-  -.-  -.-  -.-  GetDimensions:   -.-  -.-  -.-  -.-"
            print xMax, yMax


            pixelDataRangeMin, pixelDataRangeMax = reader.GetOutput().GetScalarRange()
            pixelDataRange = pixelDataRangeMax - pixelDataRangeMin
            print "-.-  -.-  -.-  -.-  GetScalarRange:   -.-  -.-  -.-  -.-"
            print pixelDataRangeMin, pixelDataRangeMax, pixelDataRange
            print "-.-"
            scalars = reader.GetOutput().GetPointData().GetScalars()
            components = scalars.GetNumberOfComponents()
            print "-.-  -.-  -.-  -.-  GetNumberOfComponents:   -.-  -.-  -.-  -.-"
            print components
            print "-.-  -.-  -.-  -.-  GetScalars:   -.-  -.-  -.-  -.-"
            print scalars

            # print reader.GetOutput()
#             print reader.GetOutput().Dimensions()
#             print reader.GetOutput().Scalars()
           
            lBackgroundRect = QtCore.QRectF( QtCore.QRect(0, 0, xMax, yMax) )
            lPixmap = QtGui.QPixmap(lBackgroundRect.width(), lBackgroundRect.height())
            lPixmap.fill( QtGui.QColor(QtCore.Qt.white) )
            lPainter = QtGui.QPainter(lPixmap)

            # show a panel containing a progress bar:
            lProgressBarPanel = CDWaitProgressBar("Loading image data from DICOM file...", xMax, self.theMainWindow)
            # show() and raise_() have to be called here:
            lProgressBarPanel.show()
            # lProgressBarPanel.raise_()

            if (components == 1) :
                # read in monochromatic data from the 2D image:
                for i in xrange(xMax):
                    lProgressBarPanel.advanceProgressBar()
                    # print i
                    for j in xrange(yMax):
                        newFloatPixelValueRaw = reader.GetOutput().GetScalarComponentAsFloat(i, j, 0, 0)
                        newFloatPixelValueNorm = 255.0 * ( \
                                (newFloatPixelValueRaw - pixelDataRangeMin) / pixelDataRange )
                        # print newFloatPixelValueRaw, newFloatPixelValueNorm, pixelDataRangeMin, pixelDataRangeMax, pixelDataRange
                        # tuple [0] =
                        # QRgb color = qRgba(tuple[0], tuple[1], tuple[2], tuple[3]);
                        lColor = QtGui.QColor(newFloatPixelValueNorm, newFloatPixelValueNorm, newFloatPixelValueNorm)
                        lPen = QtGui.QPen(lColor)
                        lPen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
                        lPainter.setPen(lPen)
                        lPainter.drawPoint(i, j)
            lProgressBarPanel.maxProgressBar()
            lProgressBarPanel.accept()

#
#             lNameToColorDictKeys = self.nameToColorDict.keys()
#             for lThePIFTextLine in lThePIFTextList:
#                 # if we have a lThePIFTextLine, it doesn't necessarily follow
#                 #   that it's a well-formed PIFF line... so we better use "try - except" :
#                 try:                   
#                     thePIFlineList = lThePIFTextLine.split(" ", QtCore.QString.SkipEmptyParts)
#                     pifXMin = int(thePIFlineList[2])
#                     pifXMax = int(thePIFlineList[3])
#                     # pifYMin = int(thePIFlineList[4])
#                     # pifYMax = int(thePIFlineList[5])
#                     # invert Y values here, from RHS to LHS:
#                     pifYMin = ((yMax-1) - int(thePIFlineList[5]))
#                     pifYMax = ((yMax-1) - int(thePIFlineList[4]))
#                     lTmpRect = QtCore.QRectF( QtCore.QRect(pifXMin, pifYMin, pifXMax-pifXMin+1, pifYMax-pifYMin+1) )
#                     lTmpPixmap  = QtGui.QPixmap( lTmpRect.width(), lTmpRect.height() )
#                     # if the PIFF line's cell type name is in our dict, assign the related color, otherwise keep black:
#                     lColor = QtCore.Qt.black
#                     for i in xrange(len(self.nameToColorDict)):
#                         # print "lNameToColorDictKeys[i] is thePIFlineList[1] =", lNameToColorDictKeys[i], thePIFlineList[1]
#                         if (str(lNameToColorDictKeys[i]) == str(thePIFlineList[1])):
#                             # print "YUHUHU lNameToColorDictKeys[i], self.nameToColorDict[lNameToColorDictKeys[i]],  QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[i]]) =", lNameToColorDictKeys[i], self.nameToColorDict[lNameToColorDictKeys[i]],  QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[i]])
#                             lColor = QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[i]])
#                     lTmpPixmap.fill( lColor )
#                     lPainter.drawPixmap(QtCore.QPoint(pifXMin,pifYMin), lTmpPixmap)
#                 except:
#                     # we got exception in parsing a PIFF line, just do nothing.
#                     pass
            lPainter.end()

            # create another pixmap of the same size as the graphics scene rectangle, and fill it with the chosen background pattern:
            image = lPixmap.toImage()



            print "2010 DEBUG: _,.- ~*'`'*~-.,_ openDICOMFile_Callback() converted [", lFileName, "] PIFF into image."

            # now confirm that the cdImageLayer contains an actual image loaded from a file:
            self.graphicsSceneWindow.cdImageLayer.setImageLoadedFromFile(True)

            self.setImageGlobals( image )
           
            # 2010 - Mitja: also pass the loaded image as background brush for the external QGraphicsScene window:
            #
            # but this TILES the image onto the "brush" background image.
            # instead, draw the pixmap onto an image, and use that image for the brush....:
            #  brushes are always tiled.
            #    You could create an image of (width(),height()),
            #    draw your pixmap in the center of it and use that for the brush.
            #    This way it will tile, but only once.
            #    Besure to update the pixmap on resize!
            lThePath,lTheFileName = os.path.split(str(lFileName))
            print "2010 DEBUG 2010 DEBUG now calling self.graphicsSceneWindow.updateBackgroundImage(lTheFileName, image) = (", lTheFileName, image, ")."
            self.graphicsSceneWindow.updateBackgroundImage(lTheFileName, image)
        print "2010 DEBUG: openDICOMFile_Callback() ENDING."
        print "-.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-"


    # end of def openDICOMFile_Callback(self)
    # ---------------------------------------------------------













    # ---------------------------------------------------------
    # 2010 - Mitja: callback for menu item to "Open TIFF Multi-page File"
    # ---------------------------------------------------------
    # simple TIFF multi-page file import:
    def openTIFFMultiPageFile_Callback(self):
   
        try:
            vtk.vtkTIFFReader()
        except:
            lVtkMissingWarning = QtGui.QMessageBox.critical( self, \
            "CellDraw", \
            "CellDraw needs VTK libraries to read multi-page TIFF files.\n\nIf you need to open TIFF files in CellDraw, please contact your system administrator or the source where you obtained CellDraw.")
            return



        CDConstants.printOut( "-.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-", CDConstants.DebugAll )
        CDConstants.printOut(  "2010 DEBUG: openTIFFMultiPageFile_Callback() STARTING with globals:" +
              "\n self.colorIds ="+str(self.colorIds)+
              "\n self.colorDict ="+str(self.colorDict)+
              "\n self.comboDict ="+str(self.comboDict)+
              "\n self.regionsTableDict ="+str(self.regionsTableDict), CDConstants.DebugAll )
        CDConstants.printOut( "     1     ", CDConstants.DebugAll )

#         lFileName = QtGui.QFileDialog.getOpenFileName(self, self.tr("CellDraw - Open TIFF Multi-Page File"), \
#                        QtCore.QDir.currentPath(), self.tr("*.tif"))

        lFileDialog = QtGui.QFileDialog(self.theMainWindow)
        lFilters = "TIFF (*.tiff);;TIF (*.tif)"
        lFileDialog.setNameFilters(lFilters);
        lFileName = lFileDialog.getOpenFileName(self, self.tr("CellDraw - Open TIFF Multi-Page File"), \
            QtCore.QDir.currentPath(), self.tr(lFilters) )

        CDConstants.printOut( "     2     lFileName = " + str(lFileName), CDConstants.DebugAll )

        if not lFileName.isEmpty():
            # 2010 - Mitja: load the file's data into a VTK object:
            lThePIFInputFile = QtCore.QFile(lFileName)
            lFileOK = lThePIFInputFile.open(QtCore.QIODevice.ReadOnly)

            CDConstants.printOut( "     3     lThePIFInputFile = " + str(lThePIFInputFile) + " lFileOK = " + str(lFileOK), CDConstants.DebugAll )
            lTheTIFFReader = vtk.vtkTIFFReader()
            CDConstants.printOut( "     4     lTheTIFFReader = " + str(lTheTIFFReader), CDConstants.DebugAll )
            lTheTIFFReader.SetFileName(str(lFileName))
            CDConstants.printOut( "     5     lTheTIFFReader = " + str(lTheTIFFReader), CDConstants.DebugAll )
            lTheTIFFReaderOutput = lTheTIFFReader.GetOutput()
            CDConstants.printOut( "     6     lTheTIFFReaderOutput = " + str(lTheTIFFReaderOutput), CDConstants.DebugAll )
            lTheTIFFReader.Update() # <-- important <-- otherwise the VTK pipeline is not updated <--
            CDConstants.printOut( "     7     lTheTIFFReaderOutput after Update() = " + str(lTheTIFFReaderOutput), CDConstants.DebugAll )
            image_point_data = lTheTIFFReader.GetOutput().GetPointData()
            CDConstants.printOut( "-.-  -.-  -.-  -.-  GetPointData == image_point_data:   -.-  -.-  -.-  -.-", CDConstants.DebugAll )
            print image_point_data
           
            scalarptr = lTheTIFFReader.GetOutput().GetScalarPointer(1, 1, 0)
            CDConstants.printOut( "-.-  -.-  -.-  -.-  GetScalarPointer == scalarptr:   -.-  -.-  -.-  -.-", CDConstants.DebugAll )
            print scalarptr
            floatsomething = lTheTIFFReader.GetOutput().GetScalarComponentAsFloat(1, 1, 0, 0)
            CDConstants.printOut( "-.-  -.-  -.-  -.-  GetScalarComponentAsFloat == floatsomething:   -.-  -.-  -.-  -.-", CDConstants.DebugAll )
            print floatsomething

            dimensions = lTheTIFFReader.GetOutput().GetDimensions()
            xMax = dimensions[0]
            yMax = dimensions[1]
            zMax = dimensions[2]
            CDConstants.printOut( "-.-  -.-  -.-  -.-  GetDimensions:   -.-  -.-  -.-  -.-", CDConstants.DebugAll )
            print xMax, yMax


            pixelDataRangeMin, pixelDataRangeMax = lTheTIFFReader.GetOutput().GetScalarRange()
            pixelDataRange = pixelDataRangeMax - pixelDataRangeMin
            CDConstants.printOut( "-.-  -.-  -.-  -.-  GetScalarRange:   -.-  -.-  -.-  -.-", CDConstants.DebugAll )
            print pixelDataRangeMin, pixelDataRangeMax, pixelDataRange
            CDConstants.printOut( "-.-", CDConstants.DebugAll )
            scalars = lTheTIFFReader.GetOutput().GetPointData().GetScalars()
            components = scalars.GetNumberOfComponents()
            CDConstants.printOut( "-.-  -.-  -.-  -.-  GetNumberOfComponents:   -.-  -.-  -.-  -.-", CDConstants.DebugAll )
            print components
            CDConstants.printOut( "-.-  -.-  -.-  -.-  GetScalars:   -.-  -.-  -.-  -.-", CDConstants.DebugAll )
            print scalars

            # print lTheTIFFReader.GetOutput()
#             print lTheTIFFReader.GetOutput().Dimensions()
#             print lTheTIFFReader.GetOutput().Scalars()
           
            lBackgroundRect = QtCore.QRectF( QtCore.QRect(0, 0, xMax, yMax) )
            lPixmap = QtGui.QPixmap(lBackgroundRect.width(), lBackgroundRect.height())
            lPixmap.fill( QtGui.QColor(QtCore.Qt.white) )
            lPainter = QtGui.QPainter(lPixmap)

            # show a panel containing a progress bar:
            lProgressBarPanel = CDWaitProgressBar("Loading image data from TIFF multi-page file...", xMax, self.theMainWindow)
            # show() and raise_() have to be called here:
            lProgressBarPanel.show()
            # lProgressBarPanel.raise_()

            if (components == 1) :
                # read in monochromatic data from the 2D image:
                for i in xrange(xMax):
                    lProgressBarPanel.advanceProgressBar()
                    # print i
                    for j in xrange(yMax):
                        newFloatPixelValueRaw = lTheTIFFReader.GetOutput().GetScalarComponentAsFloat(i, j, 0, 0)
                        newFloatPixelValueNorm = 255.0 * ( \
                                (newFloatPixelValueRaw - pixelDataRangeMin) / pixelDataRange )
                        # print newFloatPixelValueRaw, newFloatPixelValueNorm, pixelDataRangeMin, pixelDataRangeMax, pixelDataRange
                        # tuple [0] =
                        # QRgb color = qRgba(tuple[0], tuple[1], tuple[2], tuple[3]);
                        lColor = QtGui.QColor(newFloatPixelValueNorm, newFloatPixelValueNorm, newFloatPixelValueNorm)
                        lPen = QtGui.QPen(lColor)
                        lPen.setCosmetic(True) # cosmetic pen = width always 1 pixel wide, independent of painter's transformation set
                        lPainter.setPen(lPen)
                        lPainter.drawPoint(i, j)
            lProgressBarPanel.maxProgressBar()
            lProgressBarPanel.accept()

#
#             lNameToColorDictKeys = self.nameToColorDict.keys()
#             for lThePIFTextLine in lThePIFTextList:
#                 # if we have a lThePIFTextLine, it doesn't necessarily follow
#                 #   that it's a well-formed PIFF line... so we better use "try - except" :
#                 try:                   
#                     thePIFlineList = lThePIFTextLine.split(" ", QtCore.QString.SkipEmptyParts)
#                     pifXMin = int(thePIFlineList[2])
#                     pifXMax = int(thePIFlineList[3])
#                     # pifYMin = int(thePIFlineList[4])
#                     # pifYMax = int(thePIFlineList[5])
#                     # invert Y values here, from RHS to LHS:
#                     pifYMin = ((yMax-1) - int(thePIFlineList[5]))
#                     pifYMax = ((yMax-1) - int(thePIFlineList[4]))
#                     lTmpRect = QtCore.QRectF( QtCore.QRect(pifXMin, pifYMin, pifXMax-pifXMin+1, pifYMax-pifYMin+1) )
#                     lTmpPixmap  = QtGui.QPixmap( lTmpRect.width(), lTmpRect.height() )
#                     # if the PIFF line's cell type name is in our dict, assign the related color, otherwise keep black:
#                     lColor = QtCore.Qt.black
#                     for i in xrange(len(self.nameToColorDict)):
#                         # print "lNameToColorDictKeys[i] is thePIFlineList[1] =", lNameToColorDictKeys[i], thePIFlineList[1]
#                         if (str(lNameToColorDictKeys[i]) == str(thePIFlineList[1])):
#                             # print "YUHUHU lNameToColorDictKeys[i], self.nameToColorDict[lNameToColorDictKeys[i]],  QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[i]]) =", lNameToColorDictKeys[i], self.nameToColorDict[lNameToColorDictKeys[i]],  QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[i]])
#                             lColor = QtGui.QColor(self.nameToColorDict[lNameToColorDictKeys[i]])
#                     lTmpPixmap.fill( lColor )
#                     lPainter.drawPixmap(QtCore.QPoint(pifXMin,pifYMin), lTmpPixmap)
#                 except:
#                     # we got exception in parsing a PIFF line, just do nothing.
#                     pass
            lPainter.end()

            # create another pixmap of the same size as the graphics scene rectangle, and fill it with the chosen background pattern:
            image = lPixmap.toImage()



            CDConstants.printOut( "2010 DEBUG: _,.- ~*'`'*~-.,_ openTIFFMultiPageFile_Callback() converted ["+str(lFileName)+"] TIFF into image.", CDConstants.DebugAll )

            # now confirm that the cdImageLayer contains an actual image loaded from a file:
            self.graphicsSceneWindow.cdImageLayer.setImageLoadedFromFile(True)

            self.setImageGlobals( image )
           
            # 2010 - Mitja: also pass the loaded image as background brush for the external QGraphicsScene window:
            #
            # but this TILES the image onto the "brush" background image.
            # instead, draw the pixmap onto an image, and use that image for the brush....:
            #  brushes are always tiled.
            #    You could create an image of (width(),height()),
            #    draw your pixmap in the center of it and use that for the brush.
            #    This way it will tile, but only once.
            #    Besure to update the pixmap on resize!
            lThePath,lTheFileName = os.path.split(str(lFileName))
            CDConstants.printOut( "2010 DEBUG 2010 DEBUG now calling self.graphicsSceneWindow.updateBackgroundImage(lTheFileName, image) = ("+str(lTheFileName)+" "+str(image)+").", CDConstants.DebugAll )
            self.graphicsSceneWindow.updateBackgroundImage(lTheFileName, image)
        CDConstants.printOut( "2010 DEBUG: openTIFFMultiPageFile_Callback() ENDING.", CDConstants.DebugAll )
        CDConstants.printOut( "-.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-  -.-", CDConstants.DebugAll )


    # end of def openTIFFMultiPageFile_Callback(self)
    # ---------------------------------------------------------





    # ---------------------------------------------------------
    def createActions(self):

        self.connect(self.ui.action_Exit, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("close()"))

        # 2011 - Mitja: add setting a new pifScene file, to clear our PIFF scene:
        self.connect(self.ui.action_New_Scene, QtCore.SIGNAL("triggered()"), self.fileNewScene_Callback)

        # 2011 - Mitja: add opening a pifScene file, to rebuild our PIFF scene from previously saved data:
        self.connect(self.ui.action_Open_Scene, QtCore.SIGNAL("triggered()"), self.fileOpenScene_Callback)

        self.connect(self.ui.action_Open_Image, QtCore.SIGNAL("triggered()"), self.fileOpenImage_Callback)

        # 2011 - Mitja: add opening a DICOM file to read its data into a pixmap:
        # testing DICOM files input:
        self.connect(self.ui.action_Open_DICOM, QtCore.SIGNAL("triggered()"), self.openDICOMFile_Callback)

        # 2011 - Mitja: add opening a Multi-Page TIFF file to use its data to generate a volume of cells:
        self.connect(self.ui.action_Open_Multi_Page_TIFF, QtCore.SIGNAL("triggered()"), self.openTIFFMultiPageFile_Callback)



        # 2010 - Mitja: add opening a PIFF file to read its data into a pixmap:
        self.connect(self.ui.action_Open_PIFF, QtCore.SIGNAL("triggered()"), self.openPIFFile_Callback)

        # 2011 - Mitja: add saving to a pifScene file, so that we can work on it at a later time:
        self.connect(self.ui.action_Save_Scene, QtCore.SIGNAL("triggered()"), self.fileSaveScene_Callback)

        # 2010 - Mitja: someone had decided to name "printAct" an action which actually saved a file (what about using consistent naming?) but now it has been renamed to "action_Export_PIFF":
        self.connect(self.ui.action_Export_PIFF, QtCore.SIGNAL("triggered()"), self.fileSavePIFFromScene_Callback)

        # 2010 - Mitja: add a preferences action:
        self.connect(self.ui.action_Preferences, QtCore.SIGNAL("triggered()"), self.showPreferencesDialog)

        # 2010 - Mitja: add functionalities for ignoring white / black regions when saving the PIFF file:
        self.connect(self.ui.ignoreWhiteRegions_checkBox, QtCore.SIGNAL("toggled(bool)"), self.setIgnoreWhiteRegions)
        self.connect(self.ui.ignoreBlackRegions_checkBox, QtCore.SIGNAL("toggled(bool)"), self.setIgnoreBlackRegions)

        # 2010 - Mitja: add functionality for saving the PIFF file from the rasterized data:
        #    This was used for PIFF generation from a QPixmap. Since we now go through a polygon-based scene,
        #    the PIFRasterizer-related code is not necessary here anymore:
        # self.connect(self.ui.saveRasterizedData_checkBox, QtCore.SIGNAL("toggled(bool)"), self.setSavePIFFromRasterizedData)

        # 2010 - Mitja: add functionality for saving the PIFF file directly from the graphics scene:
        #   (this checkbox was previously used for testing overlaying the rasterized image on the top of the original image)
        self.connect(self.ui.graphicsScenePIFF_checkBox, QtCore.SIGNAL("toggled(bool)"), self.setSavePIFFromGraphicsScene)

        # 2010 - Mitja: add functionality for picking a color region:
        self.connect(self.ui.pickColorRegion_checkBox, QtCore.SIGNAL("toggled(bool)"), self.setPickColorRegion)

        # 2010 - Mitja: add functionality for saving PIFF metadata:
        self.connect(self.ui.graphicsScenePIFF_saveMetadataCheckbox, QtCore.SIGNAL("toggled(bool)"), self.setSavePIFMetadata)




    # ---------------------------------------------------------
    # end of class PIFInputMainWindow(QtGui.QMainWindow):
    # ---------------------------------------------------------




# ------------------------------------------------------------
# ------------------------------------------------------------
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    # 2010 - Mitja: the class PIFInputMainWindow actually implements this application's
    #   main GUI (not just the main window):
    cellDrawMainWindow = PIFInputMainWindow()

    # 2010 - Mitja: parse any command-line options to see if an image has to be
    #  opened right away, specified as "commandname -o filename"
    if len(sys.argv) > 1:
        if sys.argv[1] == '-o':
            file_name = sys.argv[2]
            cellDrawMainWindow.openImage(file_name)

    # 2010 - Mitja: QMainWindow.raise_() must be called after QMainWindow.show()
    #     otherwise the PyQt/Qt-based GUI won't receive foreground focus.
    #     It's a workaround for a well-known bug caused by PyQt/Qt on Mac OS X
    #     as shown here:
    #       http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    cellDrawMainWindow.show()
    cellDrawMainWindow.raise_()

    sys.exit(app.exec_())


# ------------------------------------------------------------
# ------------------------------------------------------------
# Local Variables:
# coding: US-ASCII
# End:
