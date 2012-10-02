#!/usr/bin/env python

from PyQt4 import QtGui, QtCore


# external class for selecting layer mode:
from cdModeSelectToolBar import CDModeSelectToolBar

# external class defining all global constants for CellDraw:
from cdConstants import CDConstants


# ======================================================================
# 2012 - Mitja: CDToolBars is a class for grouping and providing access
#    to QToolBar elements in the application
# ======================================================================
class CDToolBars(QtCore.QObject):

    # ------------------------------------------------------------
    def __init__(self,pParent):
        # we don't use the pParent parameter to pass it to QObject, 
        #  but directly as parent object for all the QToolBar objects instantiated here:
        super(CDToolBars, self).__init__(None)

        self.mainWindow = pParent
        
        # demo toolbar object:
        lOneToolbar = QtGui.QToolBar("QToolBar OneToolbar")
        print lOneToolbar
        # set all icons in this QToolBar to be of the same size:
        lOneToolbar.setIconSize(QtCore.QSize(24, 24))
        lOneToolbar.setObjectName("setObjectName OneToolbar")
        lOneToolbar.setToolTip("setToolTip OneToolbar")
        lOneToolbar.addWidget(QtGui.QLabel("*"))
        lOneToolbar.addWidget(QtGui.QLabel(" "))
        lOneToolbar.addWidget(QtGui.QLabel("_"))
        print self.mainWindow.addToolBar(QtCore.Qt.TopToolBarArea, lOneToolbar)
        # addToolBarBreak() places the next toolbar in the same toolbar area in a "new line"
        # print self.mainWindow.addToolBarBreak(QtCore.Qt.TopToolBarArea)



        # 2011 - Mitja: to control the "layer selection" for Cell Scene mode,
        #   we add a set of radio-buttons to the Control Panel:
        self.theModeSelectToolBar = CDModeSelectToolBar("CellDraw | Main ToolBar")
        print self.mainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.theModeSelectToolBar)
#         print self.mainWindow.addToolBarBreak(QtCore.Qt.TopToolBarArea)


        CDConstants.printOut( "----- CDToolBars(pParent=="+str(pParent)+").__init__() done. -----", CDConstants.DebugExcessive )

    # ------------------------------------------------------------------
    # ----- end of init() -----
    # ------------------------------------------------------------------



    # ------------------------------------------------------------
    # return the ID of the only checked button in the QButtonGroup:
    # ------------------------------------------------------------
    def getSelectedSceneMode(self):
        return self.selectedSceneMode


    # ------------------------------------------------------------
    # set a checked button in the QButtonGroup:
    # ------------------------------------------------------------
    def setSelectedSceneMode(self, pId):
        self.selectedSceneMode = pId
        self.sceneModeActionDict[self.selectedSceneMode].setChecked(True)


    # ------------------------------------------------------------
    # set the icon of the Image Layer selection button
    # ------------------------------------------------------------
    def setImageLayerButtonIcon(self, pIcon):
        self.imageLayerAction.setIcon(QtGui.QIcon( pIcon ))


    # ------------------------------------------------------------
    # register a callback handler function for the
    #   "pSignal()" signal:
    # ------------------------------------------------------------
    def registerSignalHandler(self, pSignal, pHandler):
        pSignal.connect( pHandler )








    #
    # register any callback handlers for specific toolbars:
    #



    # ------------------------------------------------------------
    # register the callback handler function for the
    #   "signalModeSelectToolbarChanged()" signal:
    # ------------------------------------------------------------
    def registerSignalHandlerForModeSelectToolbarChanges(self, pHandler):
        self.theModeSelectToolBar.signalModeSelectToolbarChanged.connect( pHandler )


# end class CDToolBars(QtGui.QObject)
# ======================================================================





# ------------------------------------------------------------
# just for testing:
# ------------------------------------------------------------
if __name__ == '__main__':
    CDConstants.printOut( "----- CDToolBars.__main__() 1", CDConstants.DebugTODO )
    import sys

    app = QtGui.QApplication(sys.argv)

    CDConstants.printOut( "----- CDToolBars.__main__() 2", CDConstants.DebugTODO )

    testQMainWindow = QtGui.QMainWindow()
    testQMainWindow.setGeometry(100, 100, 900, 500)
    testQMainWindow.setUnifiedTitleAndToolBarOnMac(False)

    CDConstants.printOut( "----- CDToolBars.__main__() 3", CDConstants.DebugTODO )

    print "testQMainWindow = ", testQMainWindow
    lTestToolBarsObject = CDToolBars(testQMainWindow)
    print "lTestToolBarsObject = ", lTestToolBarsObject
   
#     print "NOW testQMainWindow.addToolBarsToMainWindow() ..."
#     print lTestToolBarsObject.addToolBarsToMainWindow()

    CDConstants.printOut( "----- CDToolBars.__main__() 4", CDConstants.DebugTODO )

    # 2010 - Mitja: QMainWindow.raise_() must be called after QMainWindow.show()
    #     otherwise the PyQt/Qt-based GUI won't receive foreground focus.
    #     It's a workaround for a well-known bug caused by PyQt/Qt on Mac OS X
    #     as shown here:
    #       http://www.riverbankcomputing.com/pipermail/pyqt/2009-September/024509.html
    testQMainWindow.raise_()
    testQMainWindow.show()

    CDConstants.printOut( "----- CDToolBars.__main__() 5", CDConstants.DebugTODO )

    sys.exit(app.exec_())

    CDConstants.printOut( "----- CDToolBars.__main__() 6", CDConstants.DebugTODO )

# Local Variables:
# coding: US-ASCII
# End:
