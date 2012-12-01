#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

# the following import is about a resource file generated thus:
#     "pyrcc4 cdDiagramScene.qrc -o cdDiagramScene_rc.py"
# which requires the file cdDiagramScene.qrc to correctly point to the files in ":/images"
# only that way will the icons etc. be available after the import:
# import cdDiagramScene_rc


# ======================================================================
# 2012 - Mitja: a separate status bar class, to add to its default behavior.
#
#
#  NOTE:  QStatusBar has a well-documented Qt bug:
#
#    "QDockWidget statusTip missing from main window when floating"
#   it doesn't seem likely to be address by Qt developers anytime soon:
#    <http://bugreports.qt-project.org/browse/QTBUG-12054>
#  "This task is old and has been idle for a long time. It is therefore being
#   closed on the assumption that it will not be addressed in the foreseeable
#   future."
#
# ======================================================================
#
class CDStatusBar(QtGui.QStatusBar):

    # ------------------------------------------------------------
    def __init__(self,pParent=None):
        QtGui.QStatusBar.__init__(self, pParent)
        CDConstants.printOut( "___ - DEBUG ----- CDStatusBar: __init__() 0", CDConstants.DebugExcessive )

        # ----------------------------------------------------------------
        #
        # QStatusBar setup (1) - GUI setup for QStatusBar additional properties:
        #
        self.setSizeGripEnabled(True)
        self.layout().setMargin(0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.setPalette(QtGui.QPalette(QtGui.QColor(200,200,200)))
        self.setAutoFillBackground(True)
        __theStatusBarFont = QtGui.QFont()
        __theStatusBarFont.setWeight(QtGui.QFont.Light)
        self.setFont(__theStatusBarFont)
        self.__statusBarToolTipString = QtCore.QString("<strong>Status Bar</strong> - status tips, temporary messages and mode indicators are presented here.<br><small><small><br><u>Note</u>: When a Qt Dock Widget is <em>floating</em>, its status tips do not show up in the main window's status bar. It's only when the widget is docked that they do: <a href=\"http://bugreports.qt-project.org/browse/QTBUG-12054\">http://bugreports.qt-project.org/browse/QTBUG-12054</a>. To see a Dock Widget's status tips again, re-dock the Widget to the main window.</small></small>")
        self.setToolTip(self.__statusBarToolTipString)

        # ----------------------------------------------------------------
        #
        # QStatusBar setup (2) - prepare additional widget content for the ToolBar:
        #
        CDConstants.printOut( "___ - DEBUG ----- CDStatusBar: __init__() 1", CDConstants.DebugExcessive )

        # we add one small "logo" permanent widget to the far right corner of the toolbar,
        # and according to Qt documentation:
        #    "Permanently means that the widget may not be obscured by temporary messages."

        self.__thePermanentWidget = QtGui.QWidget(self)
        self.__thePermanentWidget.setLayout(QtGui.QVBoxLayout())
        self.__thePermanentWidget.layout().setMargin(0)
        self.__thePermanentWidget.layout().setSpacing(0)
        self.__thePermanentWidget.layout().setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
#         self.__thePermanentWidget.setPalette(QtGui.QPalette(QtGui.QColor(176,176,176)))
#         self.__thePermanentWidget.setAutoFillBackground(True)
        __statusBarLabelToolTipString = QtCore.QString("<small>CellDraw 1.6.0<br><br>An editing and conversion software tool for PIFF files, as used in CompuCell3D simulations.<br><br>CellDraw can be useful for creating PIFF files containing a high number of cells and cell types, either by drawing a scene containing cell regions in a paint program, and then discretize the drawing into a PIFF file, or by drawing the cell scenario directly in CellDraw.<br><br>More information at:<br><a href=\"http://www.compucell3d.org/\">http://www.compucell3d.org/</a></small>")
        self.__thePermanentWidget.setToolTip(__statusBarLabelToolTipString)
        __statusBarLabelPixmap = QtGui.QPixmap(':/icons/CellDraw.png').scaledToHeight(32)
        __statusBarLabel = QtGui.QLabel()
        __statusBarLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        __statusBarLabelTextString = QtCore.QString("<small><small><a href=\"http://compucell3d.org/\"><img src=\":/icons/CellDraw.png\" height=\"32\"><br>CellDraw</a></small></small>")
        __statusBarLabel.setText(__statusBarLabelTextString)
        __statusBarLabel.setFont(__theStatusBarFont)
#         __statusBarLabel.setPixmap(__statusBarLabelPixmap)
        
        self.__thePermanentWidget.layout().addWidget(__statusBarLabel)
# 
# 
#         self.miniAboutAction = QtGui.QAction(self.__thePermanentWidget)
#         self.miniAboutAction.setCheckable(True)
#         self.miniAboutAction.setChecked(True)
#         self.miniAboutAction.setIcon(QtGui.QIcon(':/icons/CellDraw.png'))
# #        self.miniAboutAction.setIconSize(QtCore.QSize(24, 24))
#         self.miniAboutAction.setIconText("Select")
#         self.miniAboutAction.setToolTip("Scene Layer - Select Tool")
#         self.miniAboutAction.setStatusTip("Scene Layer Select Tool: select a region in the Cell Scene")
# 
#         self.__thePermanentWidget.layout().addWidget(self.miniAboutAction)
# 
# 
# 

        # add __thePermanentWidget permanently to status bar,
        #   where the 2nd parameter is "stretch" (0 = minimum space necessary)
        self.addPermanentWidget(self.__thePermanentWidget, 0)
        self.__thePermanentWidget.show()
        self.show()


        CDConstants.printOut( "----- CDStatusBar.__init__(pParent=="+str(pParent)+") done. -----", CDConstants.DebugExcessive )

    # end of   def __init__(self,pParent=None)
    # ------------------------------------------------------------



    # ------------------------------------------------------------
    # add a widget permanently to the far right corner of the toolbar,
    # according to Qt documentation:
    #    "Permanently means that the widget may not be obscured by temporary messages."
    # ------------------------------------------------------------
    def addPermanentWidgetToStatusBar(self, pWidget):
        # the 2nd parameter is "stretch" (0 = minimum space necessary)
        self.addPermanentWidget(pWidget, 0)
        pWidget.show()
        self.show()


    # ------------------------------------------------------------
    # insert a widget towards the far right corner of the toolbar,
    # according to Qt documentation:
    #    "Permanently means that the widget may not be obscured by temporary messages."
    # ------------------------------------------------------------
    def insertPermanentWidgetInStatusBar(self, pIndex, pWidget):
        self.insertPermanentWidget(pIndex, pWidget)
#         pWidget.show()
        self.show()


    # ------------------------------------------------------------
    # remove a widget from the far right corner of the toolbar,
    # according to Qt documentation:
    #    "This function does not delete the widget but hides it.
    #     To add the widget again, you must call both the addWidget() and show() functions."
    # ------------------------------------------------------------
    def removeWidgetFromStatusBar(self, pWidget):
        self.removeWidget(pWidget)
        self.show()







# end class CDStatusBar(QtGui.QStatusBar)
# ======================================================================





# ------------------------------------------------------------
# just for testing:
# ------------------------------------------------------------
if __name__ == '__main__':
    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() 1", CDConstants.DebugAll )
    import sys

    app = QtGui.QApplication(sys.argv)

    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() 2", CDConstants.DebugAll )

    testQMainWindow = QtGui.QMainWindow()
    testQMainWindow.setGeometry(100, 100, 900, 500)

    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() 3", CDConstants.DebugAll )

    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() - testQMainWindow == "+str(testQMainWindow), CDConstants.DebugTODO )


    lTestStatusBarObject = CDStatusBar( testQMainWindow )
    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() - lTestStatusBarObject == "+str(lTestStatusBarObject), CDConstants.DebugTODO )
   
    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() - NOW testQMainWindow.setStatusBar(lTestStatusBarObject) ...", CDConstants.DebugTODO )


    lPrintOut = testQMainWindow.setStatusBar(lTestStatusBarObject)
    testQMainWindow.statusBar().show()
    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() - ... == "+str(lPrintOut), CDConstants.DebugTODO )

    testCentralWidget = QtGui.QLabel( testQMainWindow )
    testCentralWidget.setFrameShape( QtGui.QFrame.WinPanel )
    testCentralWidget.setFrameStyle( QtGui.QFrame.Panel | QtGui.QFrame.Raised )
    testCentralWidget.setLineWidth(5)
 
    lBoringPixMap = QtGui.QPixmap(480, 320)
    lBoringPixMap.fill( QtGui.QColor(QtCore.Qt.red) )
    testCentralWidget.setPixmap(lBoringPixMap)
    testCentralWidget.setToolTip("testCentralWidget toolTip")
    testCentralWidget.setStatusTip("testCentralWidget statusTip")


    testQMainWindow.setCentralWidget(QtGui.QLabel("*"))
#    testQMainWindow.setUnifiedTitleAndToolBarOnMac(False)

    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() 4", CDConstants.DebugAll )

    # 2010 - Mitja: QMainWindow.raise_() must be called after QMainWindow.show()
    #     otherwise the PyQt/Qt-based GUI won't receive foreground focus.
    #     It's a workaround for a well-known bug caused by PyQt/Qt on Mac OS X
    #     as shown here:
    #       http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    testQMainWindow.raise_()
    testQMainWindow.show()

    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() 5", CDConstants.DebugAll )

    sys.exit(app.exec_())

    CDConstants.printOut( "___ - DEBUG ----- CDStatusBar.__main__() 6", CDConstants.DebugAll )

# Local Variables:
# coding: US-ASCII
# End:
