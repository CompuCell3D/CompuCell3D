#!/usr/bin/env python
#
# CDSceneAssistant - starting assistant (or wizard) for CellDraw - Mitja 2011
#
# ------------------------------------------------------------

import sys     # for handling command-line arguments, could be removed in final version
import inspect # for debugging functions, could be removed in deployed version

import random  # for generating regions with semi-random-colors for different cell types

import math    # for sqrt()

# -->  -->  --> mswat code removed to run in MS Windows --> -->  -->
# -->  -->  --> mswat code removed to run in MS Windows --> -->  -->
# from PyQt4 import QtGui, QtCore, Qt
# <--  <--  <-- mswat code removed to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code removed to run in MS Windows <-- <--  <--

# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
# -->  -->  --> mswat code added to run in MS Windows --> -->  -->
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--
# <--  <--  <-- mswat code added to run in MS Windows <-- <--  <--



# 2011 - Mitja: external QWidget for selecting the PIFF Generation mode:
from cdControlPIFFGenerationMode import CDControlPIFFGenerationMode

# 2011 - Mitja: external QWidget to input x,y,z block cell dimensions:
from cdTableOfBlockCellSizes import CDTableOfBlockCellSizes

# 2011 - Mitja: external class defining all global constants for CellDraw:
from cdConstants import CDConstants

# 2010 - Mitja: external class for user preferences/settings:
from cdPreferences import CDPreferences

# the following import is about a resource file generated thus:
#     "pyrcc4 cdDiagramScene.qrc -o cdDiagramScene_rc.py"
# which requires the file cdDiagramScene.qrc to correctly point to the files in ":/images"
# only that way will the icons etc. be available after the import:
import cdDiagramScene_rc




# debugging functions, remove in final Panel version
def debugWhoIsTheRunningFunction():
    return inspect.stack()[1][3]
def debugWhoIsTheParentFunction():
    return inspect.stack()[2][3]

print "000a - DEBUG - mi __main__ xe 00a"


# ======================================================================
# a helper class for the small table-within table for each region
# ======================================================================
# note: this class emits a signal:
#
#         self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))
#
# ======================================================================
class PIFOneRegionTable(QtGui.QWidget):

    def __init__(self, pParent):
        # it is compulsory to call the parent's __init__ class right away:
        super(PIFOneRegionTable, self).__init__(pParent)

        # don't show this widget until it's completely ready:
        self.hide()

        # init - windowing GUI stuff:
        #   (for explanations see the miInitGUI() function below, in the CDSceneAssistant class)
        # this title probably will not show up anywhere,
        #   since the widget will be used within another widget's layout:
        self.setWindowTitle("PIFF Cell Types in One Region")

        #
        # QWidget setup - more windowing GUI setup:
        #

        miDialogsWindowFlags = QtCore.Qt.WindowFlags()
        # this panel is a so-called "Tool" (by PyQt and Qt definitions)
        #    we'd use the Tool type of window, except for this oh-so typical Qt bug:
        #    http://bugreports.qt.nokia.com/browse/QTBUG-6418
        #    i.e. it defines a system-wide panel which shows on top of *all* applications,
        #    even when this application is in the background.
        # miDialogsWindowFlags = QtCore.Qt.Tool
        #    so we use a plain QtCore.Qt.Window type instead:
        miDialogsWindowFlags = QtCore.Qt.Window
        #    add a peculiar WindowFlags combination to have no close/minimize/maxize buttons:
        miDialogsWindowFlags |= QtCore.Qt.WindowTitleHint
        miDialogsWindowFlags |= QtCore.Qt.CustomizeWindowHint
#        miDialogsWindowFlags |= QtCore.Qt.WindowMinimizeButtonHint
#        miDialogsWindowFlags |= QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(miDialogsWindowFlags)

        # 1. The widget is not modal and does not block input to other widgets.
        # 2. If widget is inactive, the click won't be seen by the widget.
        #    (it does NOT work as Qt docs says it would on Mac OS X: click-throughs don't get disabled)
        # 3. The widget can choose between alternative sizes for widgets to avoid clipping.
        # 4. The native Carbon size grip should be opaque instead of transparent.
        self.setAttribute(QtCore.Qt.NonModal  | \
                          QtCore.Qt.WA_MacNoClickThrough | \
                          QtCore.Qt.WA_MacVariableSize | \
                          QtCore.Qt.WA_MacOpaqueSizeGrip )

        # do not delete the window widget when the window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)



        # init - create central table, set it up and show it inside the panel:
        #   (for explanations see the miInitCentralTableWidget() function below,
        #    in the CDSceneAssistant class)
        self.oneRegionTable = QtGui.QTableWidget(self)
        self.oneRegionTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.oneRegionTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)
        self.oneRegionTable.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)
        # self.oneRegionTable.verticalHeader().hide()
        #
        self.oneRegionTable.setColumnCount(5)
        self.oneRegionTable.setRowCount(1)
        self.oneRegionTable.setHorizontalHeaderLabels( (" ", "Cell Type", "Amount", "Fraction", "Volume") )
        self.oneRegionTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setResizeMode(2, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setResizeMode(3, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.horizontalHeader().setResizeMode(4, QtGui.QHeaderView.Interactive)
        self.oneRegionTable.verticalHeader().setResizeMode(0, QtGui.QHeaderView.Interactive)
        #
        self.oneRegionTable.setLineWidth(1)
        self.oneRegionTable.setMidLineWidth(1)
        self.oneRegionTable.setFrameShape(QtGui.QFrame.Panel)
        self.oneRegionTable.setFrameShadow(QtGui.QFrame.Plain)

        # init - create region dict, populate it with empty values:
        self.oneRegionDict = dict()

        self.oneRegionDict = dict( { 1: [QtGui.QColor(64,64,64), "Condensing", 3, 100], \
                                     2: [QtGui.QColor(64,64,64), "NonCondensing", 4, 80]  } )

        # init - create pattern dict, it will be used to distinguish between different types in the region:
        self.patternDict = dict()
        self.patternDict = dict({ 0: QtCore.Qt.SolidPattern, \
                                  1: QtCore.Qt.Dense2Pattern, \
                                  2: QtCore.Qt.Dense4Pattern, \
                                  3: QtCore.Qt.Dense6Pattern, \
                                  4: QtCore.Qt.HorPattern, \
                                  5: QtCore.Qt.VerPattern, \
                                  6: QtCore.Qt.CrossPattern, \
                                  7: QtCore.Qt.BDiagPattern, \
                                  8: QtCore.Qt.FDiagPattern, \
                                  9: QtCore.Qt.DiagCrossPattern    } )
        # print "patternDict = ", self.patternDict

        self.populateOneRegionSubTable()

        # a separate "QDialogButtonBox" widget, to have "add" and "remove" buttons
        #   that allow the user to add/remove lines from the oneRegionTable:
        addTableRowButton = QtGui.QPushButton("+")
        removeTableRowButton = QtGui.QPushButton("-")
        self.buttonBox = QtGui.QDialogButtonBox()
        self.buttonBox.addButton(addTableRowButton, QtGui.QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton(removeTableRowButton, QtGui.QDialogButtonBox.RejectRole)
        # connects signals from buttons to "slot" methods:
        self.buttonBox.accepted.connect(self.handleAddTableRow)
        self.buttonBox.rejected.connect(self.handleRemoveTableRow)

        # place the sub-widget in a layout, assign the layout to the PIFOneRegionTable:
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(0)
        vbox.addWidget(self.oneRegionTable)
        vbox.addWidget(self.buttonBox)
        self.setLayout(vbox)
        self.layout().setAlignment(QtCore.Qt.AlignTop)

        # connect the "cellChanged" pyqtBoundSignal to a "slot" method
        #   so that it will respond to any change in table item contents:
        self.oneRegionTable.cellChanged[int, int].connect(self.handleTableCellChanged)

        self.show()

        # print "    - DEBUG ----- PIFOneRegionTable: __init__(): done"


    # ------------------------------------------------------------------
    # assign a dict parameter value to oneRegionDict:
    # ------------------------------------------------------------------
    def setOneRegionDict(self, pDict):
        self.oneRegionDict = pDict
        # print "___ - DEBUG ----- PIFOneRegionTable: setOneRegionDict() to", self.oneRegionDict, " done."


    # ------------------------------------------------------------------
    # rebuild the oneRegionDict global from its table contents:
    # ------------------------------------------------------------------
    def updateOneRegionDict(self):
        # print "___ - DEBUG DEBUG DEBUG ----- PIFOneRegionTable: self.updateOneRegionDict() from ", self.oneRegionDict

        # set how many rows are present in the table:
        lTypesCount = self.oneRegionTable.rowCount()

        # get the oneRegionDict keys in order to access elements:
        lKeys = self.oneRegionDict.keys()
       
        # add additional (non-editable) table column with percentages calculated at each table content update:
        lTotalAmounts = 0.0

        # parse each table rown separately to build a oneRegionDict entry:
        for i in xrange(lTypesCount):

            #   the key is NOT retrieved from the first oneRegionTable column,
            #   (since we don't show oneRegionDict keys in oneRegionTable anymore) :
            # lKey = self.oneRegionTable.item(i, 0)
            lKey = lKeys[i]

            # the color remains the same (table color is not editable (yet?) )
            lColor = self.oneRegionDict[lKey][0]
            # the region name is retrieved from the third oneRegionTable column:
            lTypeName = str ( self.oneRegionTable.item(i, 1).text() )
            # the amount (or "percentage" of total) for this cell type is retrieved from the third oneRegionTable column:
            lAmount = float ( self.oneRegionTable.item(i, 2).text() )
            # keep track of the total of all individual cell type amounts:
            lTotalAmounts = lTotalAmounts + lAmount
            # 2011 - add a Volume field for each cell type:
            lVolume = float ( self.oneRegionTable.item(i, 4).text() )

            # rebuild the i-th dict entry:
            self.oneRegionDict[lKey] = [ lColor, lTypeName, lAmount, lVolume ]


        # now fill the additional (non-editable) table column with percentages:
        # (prevent oneRegionTable from emitting any "cellChanged" signals when
        #   updating its content programmatically:)
        self.oneRegionTable.blockSignals(True)
        for i in xrange(lTypesCount):
            lKey = lKeys[i]
            # create a fourth QTableWidgetItem and place a value (i.e. cells amount in region) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg( self.oneRegionDict[lKey][2] / lTotalAmounts ) )
            # the region color table item shouldn't be selectable/editable:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 2 in the table:
            self.oneRegionTable.setItem(i, 3, lItem)

        # (allow oneRegionTable to emit "cellChanged" signals now that we're done
        #   updating its content programmatically:)
        self.oneRegionTable.blockSignals(False)


        # print "___ - DEBUG ----- PIFOneRegionTable: self.updateOneRegionDict() to ", self.oneRegionDict, " done."

    # ------------------------------------------------------------------
    # retrieve the up-to-date oneRegionDict for external use:
    # ------------------------------------------------------------------
    def getOneRegionDict(self):
        # first rebuild the oneRegionDict global from its table:
        self.updateOneRegionDict()
        # print "___ - DEBUG ----- PIFOneRegionTable: getOneRegionDict is ", self.oneRegionDict, " done."
        return self.oneRegionDict

    # ------------------------------------------------------------------
    # populate the one-region subtable with data from oneRegionDict
    # ------------------------------------------------------------------
    def populateOneRegionSubTable(self):
#        print "SIZE SIZE SIZE self.oneRegionTable.height() =", self.oneRegionTable.height()
#        print "SIZE SIZE SIZE self.oneRegionTable.parentWidget().height() =", self.oneRegionTable.parentWidget().height()

#        print "SIZE SIZE SIZE self.oneRegionTable.width() =", self.oneRegionTable.width()
#        print "SIZE SIZE SIZE self.oneRegionTable.parentWidget().width() =", self.oneRegionTable.parentWidget().width()
#        print "___ - DEBUG ----- PIFOneRegionTable: populateOneRegionSubTable() = ", \
#              self.oneRegionTable.rowCount()

        # prevent oneRegionTable from emitting any "cellChanged" signals when
        #   updating its content programmatically:
        self.oneRegionTable.blockSignals(True)

        # set how many rows are needed in the table:
        lTypesCount = len(self.oneRegionDict)
        self.oneRegionTable.setRowCount(lTypesCount)

        # set how many rows are present in the table:
#         print "___ - DEBUG ----- PIFOneRegionTable: populateOneRegionSubTable() = ", \
#               self.oneRegionTable.rowCount()

        # get the oneRegionDict keys in order to resize the table:
        lKeys = self.oneRegionDict.keys()
#         print "___ - DEBUG ----- PIFOneRegionTable: lKeys =", lKeys

        # add additional (non-editable) table column with percentages calculated at each table content update:
        lTotalAmounts = 0.0

        for i in xrange(lTypesCount):
            lKey = lKeys[i]

            #   before, we created a QTableWidgetItem **item**, set its string value to the dict key:
            # lItem = QtGui.QTableWidgetItem( QtCore.QString("%1").arg(lKey) )
            #   the table item containing the dict key ought not to be selected/edited:
            # lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            #   this string value *was* shown in column 0 of the table widget:
            # self.oneRegionTable.setItem(i, 0, lItem)

            # create a first QTableWidgetItem and place a swatch in it:
            lItem = QtGui.QTableWidgetItem()
            # (setting the background color like this would not be a good idea,
            #   for example because it can be confused with selection highlight colors:
            #   lItem.setBackground(QtGui.QBrush(self.oneRegionDict[lKey][0]))
            # this way is much better, it generates a rectangle in the same color,
            #   and we add patterns to the color so that it can distinguish types within regions:
            #   print "lKey,self.oneRegionDict[lKey] = ", lKey,self.oneRegionDict[lKey]
            lItem.setIcon(  self.createColorIcon( self.oneRegionDict[lKey][0] )  )
            # 2011 Mitja - remove the pattern for each cell type, so that colors can show better:
            # lItem.setIcon(  self.createColorIcon( self.oneRegionDict[lKey][0], self.patternDict[int(lKey) % 10] )  )
            # the region color table item shouldn't be selectable/editable:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 0 in the table:
            self.oneRegionTable.setItem(i, 0, lItem)

            # create a second QTableWidgetItem and place text (i.e. cell type name) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.oneRegionDict[lKey][1]) )
            # this goes to column 1 in the table:
            self.oneRegionTable.setItem(i, 1, lItem)

            # create a third QTableWidgetItem and place a value (i.e. cells amount in region) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.oneRegionDict[lKey][2]) )
            # this goes to column 2 in the table:
            self.oneRegionTable.setItem(i, 2, lItem)
            lTotalAmounts = lTotalAmounts + self.oneRegionDict[lKey][2]

            # create a fifth QTableWidgetItem and place a value (i.e. volume for this cell type) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.oneRegionDict[lKey][3]) )
            # this goes to column 4 in the table,
            #   because column 3 is just the automatically-computed percentage of cells per type:
            self.oneRegionTable.setItem(i, 4, lItem)



        for i in xrange(lTypesCount):
            lKey = lKeys[i]
            # create a fourth QTableWidgetItem and place a value (i.e. cells amount in region) from oneRegionDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg( self.oneRegionDict[lKey][2] / lTotalAmounts ) )
            # the region color table item shouldn't be selectable/editable:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 3 in the table:
            self.oneRegionTable.setItem(i, 3, lItem)
#
#
#         # distribute the available space according to the space requirement of each column or row:
#         # TODO TODO: all this widget layout semi/auto/resizing still @#$@*Y@ does not work...
#         w = 0
#         for i in xrange(self.oneRegionTable.columnCount()):
#             # print "column", i, "has width", self.oneRegionTable.columnWidth(i)
#             w = w + self.oneRegionTable.columnWidth(i)
#         w = w + self.oneRegionTable.verticalHeader().width()
#         w = w + self.oneRegionTable.verticalScrollBar().width()
#         # print "column and everything is", w
#
#         h = 0
#         for i in xrange(self.oneRegionTable.rowCount()):
#             # print "row", i, "has height", self.oneRegionTable.rowHeight(i)
#             h = h + self.oneRegionTable.rowHeight(i)
#         h = h + self.oneRegionTable.horizontalHeader().height()
#         h = h + self.oneRegionTable.horizontalScrollBar().height()
#         # print "column and everything is", h
# #        self.oneRegionTable.resize(w + 4, h + 4)
# #        self.oneRegionTable.parentWidget().resize(w + 4, h + 4)
#
# #        self.oneRegionTable.resizeRowsToContents()
        self.oneRegionTable.resizeColumnsToContents()


#        print "SIZE SIZE SIZE self.oneRegionTable.height() =", self.oneRegionTable.height()
#        print "SIZE SIZE SIZE self.oneRegionTable.parentWidget().height() =", self.oneRegionTable.parentWidget().height()

#        print "SIZE SIZE SIZE self.oneRegionTable.width() =", self.oneRegionTable.width()
#        print "SIZE SIZE SIZE self.oneRegionTable.parentWidget().width() =", self.oneRegionTable.parentWidget().width()

        # start with no table cell selected, and the user can then click to select:
        self.oneRegionTable.setCurrentCell(-1,-1)
        # self.oneRegionTable.parentWidget().resize(self.oneRegionTable.width(),self.oneRegionTable.height())

        # allow oneRegionTable to emit "cellChanged" signals now that we're done
        #   updating its content programmatically:
        self.oneRegionTable.blockSignals(False)

        # print "___ - DEBUG ----- PIFOneRegionTable: populateOneRegionSubTable() : done"

    # ------------------------------------------------------------------
    # remove the last row from the table, if there is more than one row:
    # ------------------------------------------------------------------
    def handleRemoveTableRow(self):
        # get the oneRegionTable keys in order to resize the table:
        lKeys = self.oneRegionDict.keys()
        lLength = len(self.oneRegionDict)
        # don't remove the last row in the dict:
        if (lLength > 1):
            lLastKey = lKeys[lLength-1]
            # remove the last entry in the oneRegionDict:
            del self.oneRegionDict[lLastKey]
            # update the table:
            self.populateOneRegionSubTable()

        # propagate the change upstream by emitting a signal, for example to parent objects:
        self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))

        # print "___ - DEBUG ----- PIFOneRegionTable: handleRemoveTableRow() : done"

    # ------------------------------------------------------------------
    # add a row to the table:
    # ------------------------------------------------------------------
    def handleAddTableRow(self):
        self.updateOneRegionDict()
        # print "___ - DEBUG ----- PIFOneRegionTable: handleAddTableRow FROM ", self.oneRegionDict

        # get the oneRegionTable keys in order to resize the table:
        lKeys = self.oneRegionDict.keys()
        lLength = len(self.oneRegionDict)
        # generate a new key for the oneRegionDict (just a sequential number)
        lNewKey = lKeys[lLength-1] + 1

        # generate a oneRegionDict entry:
        lThisRegionQColor = QtGui.QColor( self.oneRegionDict[lKeys[lKeys[lLength-1]]][0] )
        lNewColorR = lThisRegionQColor.red()
        lNewColorG = lThisRegionQColor.green()
        lNewColorB = lThisRegionQColor.blue()

#         lNewColorB = lNewColorB + 64
#         lNewColorR = lNewColorR + 64

        if (lNewColorR < 128) :
            lNewColorR = lNewColorR + int(random.random()*64.0)
        else:
            lNewColorR = lNewColorR - int(random.random()*64.0)

        if (lNewColorG < 128) :
            lNewColorG = lNewColorG + int(random.random()*64.0)
        else:
            lNewColorG = lNewColorG - int(random.random()*64.0)

        if (lNewColorB < 128) :
            lNewColorB = lNewColorB + int(random.random()*64.0)
        else:
            lNewColorB = lNewColorB - int(random.random()*64.0)

        # CONTINUA 
        lNewColor = QtGui.QColor( lNewColorR, \
                                  lNewColorG, \
                                  lNewColorB    )

        CDConstants.printOut( "DEBUG ----- CDSceneRasterizer: handleAddTableRow(): lNewColor = " + str(lNewColor), CDConstants.DebugAll )
        CDConstants.printOut( "DEBUG ----- CDSceneRasterizer: handleAddTableRow(): lNewColorR = " + str(lNewColorR), CDConstants.DebugAll )
        CDConstants.printOut( "DEBUG ----- CDSceneRasterizer: handleAddTableRow(): lNewColorG = " + str(lNewColorG), CDConstants.DebugAll )
        CDConstants.printOut( "DEBUG ----- CDSceneRasterizer: handleAddTableRow(): lNewColorB = " + str(lNewColorB), CDConstants.DebugAll )

        self.oneRegionDict[lNewKey] = [ lNewColor, "newType"+str(lNewKey+1),  0.0, 100 ]
        self.updateOneRegionDict()
        # print "___ - DEBUG ----- PIFOneRegionTable: handleAddTableRow TO ", self.oneRegionDict
        # update the table:
        self.populateOneRegionSubTable()

        # propagate the change upstream by emitting a signal, for example to parent objects:
        self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))

        # print "___ - DEBUG ----- PIFOneRegionTable: handleAddTableRow() done."

    # ------------------------------------------------------------------
    # generate an icon containing a rectangle in a specified color
    # ------------------------------------------------------------------
    def createColorIcon(self, pColor, pPattern = None):
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        lBrush = QtGui.QBrush(pColor)
        if pPattern is not None:
            lBrush.setStyle(pPattern)
        painter.fillRect(QtCore.QRect(0, 0, 32, 32), lBrush)
        painter.end()
        return QtGui.QIcon(pixmap)



    # ------------------------------------------------------------------
    # this is a slot method to handle "content change" events (AKA signals)
    #   from table items:
    # ------------------------------------------------------------------
    def handleTableCellChanged(self,pRow,pColumn):

        # from <http://lateral.netmanagers.com.ar/stories/BBS48.html> :
        #   "We could define a method in the Main class, and connect it to the
        #    itemChanged signal, but there is no need because we can use AutoConnect."
        #   "If you add to a class "Main" a method with a specific name,
        #    it will be connected to that signal. The name is on_objectname_signalname."

        # print "___ - DEBUG ----- CDSceneAssistant: handleTableCellChanged() pRow,pColumn =" , pRow,pColumn
        # update the dict:
        self.updateOneRegionDict()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))

        # print "___ - DEBUG ----- CDSceneAssistant: handleTableCellChanged()  done."

# end of class PIFOneRegionTable(QtGui.QWidget)
# ======================================================================





# ======================================================================
# a QWidget-based control panel, in application-specific panel style
# ======================================================================
# note: this class emits a signal:
#
#         self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))
#
class CDMainTableClass(QtGui.QWidget):

    def __init__(self, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDMainTableClass, self).__init__(pParent)


        # don't show this widget until it's completely ready:
        self.hide()

        # save the parent, whatever that may be:
        lParent = pParent

        #
        # init - windowing GUI stuff:
        #
        self.miInitGUI()



        # 2011 - Mitja: to control the "PIFF Generation mode",
        #   we add a set of radio-buttons:

        self.theControlsForPIFFGenerationMode = CDControlPIFFGenerationMode()

        self.layout().addWidget(self.theControlsForPIFFGenerationMode)

        #
        # init - create central table, set it up and show it inside the panel:
        #
        self.layout().addWidget(self.miInitCentralTableWidget())

        #
        # init - create region dict, populate it with empty values:
        #
        self.regionsDict = dict()
        self.debugRegionDict()
        self.populateTableWithRegionsDict()
        self.debugRegionDict()

        # the QObject.connect( QObject, SIGNAL(), QObject, SLOT() ) function
        #   creates a connection of the given type from the signal in the sender object
        #   to the slot method in the receiver object,
        #   and it returns true if the connection succeeds; otherwise it returns false.

#
#         #
#         # the QTableWidget type emits a signal named "itemSelectionChanged()"
#         #   whenever the selection changes:  TODO TODO TODO do we need this?
#         answer = self.connect(self.regionsTable, \
#                               QtCore.SIGNAL("itemSelectionChanged()"), \
#                               self.handleTableItemSelectionChanged )
#         # print "004 - DEBUG ----- CDMainTableClass: self.connect() =", answer

        # connect the "cellChanged" pyqtBoundSignal to a "slot" method
        #   so that it will respond to any change in table item contents:
        self.regionsTable.cellChanged[int, int].connect(self.handleTableCellChanged)


        # print "005 - DEBUG ----- CDMainTableClass: __init__(): done"


    # ------------------------------------------------------------------
    # define functions to initialize this panel:
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # init - windowing GUI stuff:
    # ------------------------------------------------------------------
    def miInitGUI(self):

        # how will the CDMainTableClass look like:
        self.setWindowTitle("Table of Types")

        self.setMinimumSize(540,322)
        # self.setMinimumSize(466,322)
        # setGeometry is inherited from QWidget, taking 4 arguments:
        #   x,y  of the top-left corner of the QWidget, from top-left of screen
        #   w,h  of the QWidget
        # NOTE: the x,y is NOT the top-left edge of the window,
        #    but of its **content** (excluding the menu bar, toolbar, etc.
        # self.setGeometry(750,60,480,480)

        # the following is only useful to fix random placement at initialization
        #   *if* we use this panel as stand-alone, without including it in windows etc.
        #   These are X,Y *screen* coordinates (INCLUDING menu bar, etc.),
        #   where X,Y=0,0 is the top-left corner of the screen:
        pos = self.pos()
        pos.setX(800)
        pos.setY(30)
        self.move(pos)
        self.show()


        # QHBoxLayout layout lines up widgets horizontally:
        self.tableWindowMainLayout = QtGui.QHBoxLayout()
        self.tableWindowMainLayout.setContentsMargins(2,2,2,2)
        self.tableWindowMainLayout.setSpacing(2)
        self.tableWindowMainLayout.setAlignment( \
            QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setLayout(self.tableWindowMainLayout)

        #
        # QWidget setup (2) - more windowing GUI setup:
        #

        miDialogsWindowFlags = QtCore.Qt.WindowFlags()
        # this panel is a so-called "Tool" (by PyQt and Qt definitions)
        #    we'd use the Tool type of window, except for this oh-so typical Qt bug:
        #    http://bugreports.qt.nokia.com/browse/QTBUG-6418
        #    i.e. it defines a system-wide panel which shows on top of *all* applications,
        #    even when this application is in the background.
        # miDialogsWindowFlags = QtCore.Qt.Tool
        #    so we use a plain QtCore.Qt.Window type instead:
        miDialogsWindowFlags = QtCore.Qt.Window
        #    add a peculiar WindowFlags combination to have no close/minimize/maxize buttons:
        miDialogsWindowFlags |= QtCore.Qt.WindowTitleHint
        miDialogsWindowFlags |= QtCore.Qt.CustomizeWindowHint
#        miDialogsWindowFlags |= QtCore.Qt.WindowMinimizeButtonHint
#        miDialogsWindowFlags |= QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(miDialogsWindowFlags)

        # 1. The widget is not modal and does not block input to other widgets.
        # 2. If widget is inactive, the click won't be seen by the widget.
        #    (it does NOT work as Qt docs says it would on Mac OS X: click-throughs don't get disabled)
        # 3. The widget can choose between alternative sizes for widgets to avoid clipping.
        # 4. The native Carbon size grip should be opaque instead of transparent.
        self.setAttribute(QtCore.Qt.NonModal  | \
                          QtCore.Qt.WA_MacNoClickThrough | \
                          QtCore.Qt.WA_MacVariableSize | \
                          QtCore.Qt.WA_MacOpaqueSizeGrip )

        # do not delete the window widget when the window is closed:
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
       
        # setWindowOpacity seems to work only if it's set after setting WindowFlags and attributes:
        self.setWindowOpacity(0.95)



    # ------------------------------------------------------------------
    # init (3) - central table, set up and show:
    # ------------------------------------------------------------------
    def miInitCentralTableWidget(self):
        # this assigns a table content to the "central widget" area of the window:
        self.regionsTable = QtGui.QTableWidget()
        # table item selection set to: "When the user selects an item,
        # any already-selected item becomes unselected, and the user cannot unselect
        # the selected item by clicking on it." :
        self.regionsTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        # clicking on a table item selects a single item:
        self.regionsTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)
        # actions which will initiate item editing: all of them (double-clicking, etc.) :
        self.regionsTable.setEditTriggers(QtGui.QAbstractItemView.AllEditTriggers)

        # don't show the table's rown numbers in the "verticalHeader" on the left side, since its
        #   numbering is inconsistent with the 1st column, which contains cell type ID numbers:
        # self.regionsTable.verticalHeader().hide()
        # show the table's verticalHeader, but remove all its labels:
        self.regionsTable.setHorizontalHeaderLabels( \
             (" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," ") )

        self.regionsTable.setColumnCount(6)
        self.regionsTable.setRowCount(6)
        self.regionsTable.setHorizontalHeaderLabels( \
             ("#", "Color", "Region", "Cell Size", "Use", "Cell Types in Region") )
        self.regionsTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Interactive)
        self.regionsTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Interactive)
        self.regionsTable.horizontalHeader().setResizeMode(2, QtGui.QHeaderView.Interactive)
        self.regionsTable.horizontalHeader().setResizeMode(3, QtGui.QHeaderView.Interactive)
        self.regionsTable.horizontalHeader().setResizeMode(4, QtGui.QHeaderView.Interactive)
        self.regionsTable.horizontalHeader().setResizeMode(5, QtGui.QHeaderView.Interactive)
        self.regionsTable.horizontalHeader().setStretchLastSection(True)
        # self.regionsTable.horizontalHeader().resizeSection(1, 180)
        self.regionsTable.show()

#        self.regionsTable.setMinimumSize(QtCore.QSize(245,0))
        self.regionsTable.setLineWidth(1)
        self.regionsTable.setMidLineWidth(1)
        self.regionsTable.setFrameShape(QtGui.QFrame.Panel)
        self.regionsTable.setFrameShadow(QtGui.QFrame.Plain)
        self.regionsTable.setObjectName("regionsTable")

        # -------------------------------------------
        # place the table in the Panel's vbox layout:
        # one cell information widget, containing a vbox layout, in which to place the table:
        tableContainerWidget = QtGui.QWidget()

        # this topFiller part is cosmetic and could safely be removed:
        topFiller = QtGui.QWidget()
        topFiller.setSizePolicy(QtGui.QSizePolicy.Expanding,
                QtGui.QSizePolicy.Expanding)

        # this infoLabel part is cosmetic and could safely be removed,
        #     unless useful info is provided here:
        self.infoLabel = QtGui.QLabel()
        self.infoLabel.setText("<i>colors</i> in the Cell Scene correspond to <i>cell region types</i>")
        self.infoLabel.setAlignment = QtCore.Qt.AlignCenter
        self.infoLabel.setLineWidth(3)
        self.infoLabel.setMidLineWidth(3)
        self.infoLabel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)

        # this bottomFiller part is cosmetic and could safely be removed:
        bottomFiller = QtGui.QWidget()
        bottomFiller.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        # create a layout and place all 'sub-widgets' in it:
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(0,0,0,0)
        # vbox.addWidget(topFiller)
        vbox.addWidget(self.regionsTable)
        vbox.addWidget(self.infoLabel)
        # vbox.addWidget(bottomFiller)
        # finally place the complete layout in a QWidget and return it:
        tableContainerWidget.setLayout(vbox)
        return tableContainerWidget




    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # now define fuctions that actually do something with data:
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # assign new dict content to the regionsDict global
    # ------------------------------------------------------------------
    def setRegionsDict(self, pDict):
        self.regionsDict = None
        self.regionsDict = pDict
        self.debugRegionDict()
        # print "___ - DEBUG ----- CDMainTableClass: self.setRegionDict() to ", self.regionsDict, " done."


    # ------------------------------------------------------------------
    # rebuild the regionsDict global from its PIFOneRegionTable objects:
    # ------------------------------------------------------------------
    def updateRegionsDict(self):
        # print "___ - DEBUG DEBUG DEBUG ----- CDMainTableClass: self.updateRegionsDict() from ", self.regionsDict

        # set how many rows are needed in the table:
        lRegionsCount = self.regionsTable.rowCount()

        # get the regionsDict keys in order to access elements:
        lKeys = self.getRegionsDictKeys()

        # parse each table rown separately to build a regionsDict entry:
        for i in xrange(lRegionsCount):
            # the key is NOT retrieved from the first regionsTable column:
            # lKey = self.regionsTable.item(i, 0)
            lKey = lKeys[i]
            # the color remains the same (table color is not editable (yet?) )
            lColor = self.regionsDict[lKeys[i]][0]

            # the region name is retrieved from the 3.rd regionsTable column:
            # print "___ - DEBUG DEBUG DEBUG ----- CDMainTableClass: self.updateRegionsDict() \n"
            # print "      i, self.regionsTable.item(i, 2) ", i, self.regionsTable.item(i, 2)
            lRegionName = str ( self.regionsTable.item(i, 2).text() )
#
#             # the region cell size is retrieved from the 4.th regionsTable column:
#             # print "___ - DEBUG DEBUG DEBUG ----- CDMainTableClass: self.updateRegionsDict() \n"
#             # print "      i, self.regionsTable.item(i, 3) ", i, self.regionsTable.item(i, 3)
#             lRegionCellSize = int ( self.regionsTable.item(i, 3).text() )

            # the region cell sizes are retrieved from the 4.th regionsTable column:
            CDConstants.printOut ( "___ - DEBUG DEBUG DEBUG ----- CDMainTableClass: self.updateRegionsDict() ", CDConstants.DebugVerbose )
            CDConstants.printOut ( "      i, self.regionsTable.cellWidget(i, 3) " + str(i) + " " + str( self.regionsTable.cellWidget(i, 3) ), CDConstants.DebugVerbose )
            lRegionBlockCellSizes = self.regionsTable.cellWidget(i, 3)

            # the region use is retrieved from the 5.th regionsTable column:
            # print "___ - DEBUG DEBUG DEBUG ----- CDMainTableClass: self.updateRegionsDict() \n"
            # print "      i, self.regionsTable.item(i, 4) ", i, self.regionsTable.item(i, 4)
            lRegionInUse = int ( self.regionsTable.item(i, 4).text() )

            # the cell types dict for each region is obtained from the PIFOneRegionTable widget
            #   which is in the 6.th  regionsTable column:
            lOneRegionTableWidget = self.regionsTable.cellWidget(i, 5)

            # rebuild the i-th dict entry:
            self.regionsDict[lKey] = [ lColor, lRegionName, \
                                       lRegionBlockCellSizes.getOneRegionDict(), \
                                       lRegionInUse, \
                                       lOneRegionTableWidget.getOneRegionDict() ]

        # print "___ - DEBUG ----- CDMainTableClass: self.updateRegionsDict() to ", self.regionsDict, " done."


    # ------------------------------------------------------------------
    # retrieve the up-to-date regionsDict for external use:
    # ------------------------------------------------------------------
    def getRegionsDict(self):
        # first rebuild the regionsDict global from its PIFOneRegionTable objects:
        self.updateRegionsDict()
        # print "___ - DEBUG ----- CDMainTableClass: getRegionsDict is ", self.regionsDict, " done."
        return self.regionsDict


    # ------------------------------------------------------------------
    # populate the table widget with data from the regionsDict global
    # ------------------------------------------------------------------
    def populateTableWithRegionsDict(self):
#         print "___ - DEBUG ----- CDMainTableClass: populateTableWithRegionsDict() = ", \
#               self.regionsTable.rowCount(), " rows to ", self.getRegionsDictElementCount(), "cells."

        # prevent regionsTable from emitting any "cellChanged" signals when
        #   updating its content programmatically:
        self.regionsTable.blockSignals(True)

        # set how many rows are needed in the table:
        self.regionsTable.setRowCount(self.getRegionsDictElementCount())

        # get the regionsDict keys in order to resize the table:
        keys = self.getRegionsDictKeys()
       
        # the entire table might be set to hide if there are no used rows:
        lThereAreRegionRowsInUse = False

        for i in xrange(self.getRegionsDictElementCount()):

            # create 1.st QTableWidgetItem **item**, set its string value to the dict key:
            lItem = QtGui.QTableWidgetItem( QtCore.QString("%1").arg(keys[i]) )
            # the table item containing the dict key ought not to be selected/edited:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this string value is shown in column 0 of the table widget:
            self.regionsTable.setItem(i, 0, lItem)

            # create 2.nd QTableWidgetItem and place a color rectangle in it:
            lItem = QtGui.QTableWidgetItem()
            lItem.setIcon(self.createColorIcon( self.regionsDict[keys[i]][0]) )
            # print "self.regionsDict[keys[i]][0] =", self.regionsDict[keys[i]][0]
            # the table item containing the region color ought not to be selected/edited:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 1 in the table:
            self.regionsTable.setItem(i, 1, lItem)
            # clicking on this item selects a single item:

            # create 3.rd QTableWidgetItem and place the region name from regionsDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.regionsDict[keys[i]][1]) )
            # print "self.regionsDict[keys[i]][1] =", self.regionsDict[keys[i]][1]
            # the table item containing the text from regionsDict ought not to be selected/edited:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 2 in the table:
            self.regionsTable.setItem(i, 2, lItem)

#
#             # create a 4.th QTableWidgetItem and place the cell size from regionsDict in it:
#             lItem = QtGui.QTableWidgetItem( \
#                        QtCore.QString("%1").arg(self.regionsDict[keys[i]][2]) )
#             # print "self.regionsDict[keys[i]][2] =", self.regionsDict[keys[i]][2]
#             # this goes to column 3 in the table:
#             self.regionsTable.setItem(i, 3, lItem)




            # create a 4.th widget (not a simple QTableWidgetItem) from the CDTableOfBlockCellSizes class,
            #   and place the cell sizes from regionsDict in it:
            lOneRegionTableWidget = CDTableOfBlockCellSizes(self)
            # populate it with data from the current row in the main regionsTable:
            aList = list( ( -1, -2, -3) )
            CDConstants.printOut ( "self.regionsDict[keys[i]][2] = " + str(self.regionsDict[keys[i]][2]), CDConstants.DebugAll )
            for j in xrange(3):
                CDConstants.printOut (  "at i = " +str(i)+ ", j = " + str(j) + "self.regionsDict[keys[i]][2][j] = " + str(self.regionsDict[keys[i]][2][j]), CDConstants.DebugAll )
                aList[j] = self.regionsDict[keys[i]][2][j]
            lOneRegionTableWidget.setOneRegionBlockCellSizes(aList)
            lOneRegionTableWidget.populateOneRegionBlockCellSizesTable()
            # this goes to column 3 in the table:
            self.regionsTable.setCellWidget(i, 3, lOneRegionTableWidget)

            # explicitly connect the "oneRegionTableChangedSignal()" signal from the
            #   lOneRegionTableWidget object, to our "slot" method
            #   so that it will respond to any change in the subtable item contents:
            answer = self.connect(lOneRegionTableWidget, \
                                  QtCore.SIGNAL("oneBlockCellDimChangedSignal()"), \
                                  self.handleOneRegionTableWidgetChanged )

            # this goes to the last column in the table:
            print "SIZE SIZE SIZE lOneRegionTableWidget.height() =", lOneRegionTableWidget.height()
            print "SIZE SIZE SIZE lOneRegionTableWidget.height() =", lOneRegionTableWidget.width()
            # first resize the table row to PIFOneRegionTable's height,
            #   then assign the widget as content to this element in the row,
            #   since resizing after assigning would not seem to work:
#             self.regionsTable.verticalHeader().resizeSection(i, lOneRegionTableWidget.height())
#             self.regionsTable.horizontalHeader().resizeSection(i, lOneRegionTableWidget.width())








            # create a 5.th QTableWidgetItem and place the region use from regionsDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.regionsDict[keys[i]][3]) )
            # print "self.regionsDict[keys[i]][3] =", self.regionsDict[keys[i]][3]
            # the table item containing the text from regionsDict ought not to be selected/edited:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 4 in the table:
            self.regionsTable.setItem(i, 4, lItem)

            # create a 6.th widget (not a simple QTableWidgetItem) from the PIFOneRegionTable class:
            lOneRegionTableWidget = PIFOneRegionTable(self)
            # create a PIFOneRegionTable widget and populate it with data
            #   from the current row in the main regionsTable:
            aDict = dict()
            # print "self.regionsDict[keys[i]][4] =", self.regionsDict[keys[i]][4]
            for j in xrange( len(self.regionsDict[keys[i]][4]) ):
                # print "at i =", i, ", j =", j, "self.regionsDict[keys[i]][4][j] =", self.regionsDict[keys[i]][4][j]
                aDict[j] = self.regionsDict[keys[i]][4][j]
            lOneRegionTableWidget.setOneRegionDict(aDict)
            lOneRegionTableWidget.populateOneRegionSubTable()
            self.regionsTable.setCellWidget(i, 5, lOneRegionTableWidget)
            # explicitly connect the "oneRegionTableChangedSignal()" signal from the
            #   lOneRegionTableWidget object, to our "slot" method
            #   so that it will respond to any change in the subtable item contents:
            answer = self.connect(lOneRegionTableWidget, \
                                  QtCore.SIGNAL("oneRegionTableChangedSignal()"), \
                                  self.handleOneRegionTableWidgetChanged )

            # this goes to the last column in the table:
            print "SIZE SIZE SIZE lOneRegionTableWidget.height() =", lOneRegionTableWidget.height()
            print "SIZE SIZE SIZE lOneRegionTableWidget.height() =", lOneRegionTableWidget.width()
            # first resize the table row to PIFOneRegionTable's height,
            #   then assign the widget as content to this element in the row,
            #   since resizing after assigning would not seem to work:
            self.regionsTable.verticalHeader().resizeSection(i, lOneRegionTableWidget.height())
            self.regionsTable.horizontalHeader().resizeSection(i, lOneRegionTableWidget.width())



            # finally determine if the table row at index i is to be visible or not, from region use:
            # print "self.regionsDict[keys[i]][3] =", self.regionsDict[keys[i]][3]
            if self.regionsDict[keys[i]][3] < 1:
                self.regionsTable.hideRow(i)
            else:
                self.regionsTable.showRow(i)
                lThereAreRegionRowsInUse = True

#         if lThereAreRegionRowsInUse is False:
#             self.regionsTable.hide()
#         else:
#             self.regionsTable.show()




#         self.regionsTable.resizeRowsToContents()
        # we have to "resize columns to contents" here, otherwise each column will
        #   be as wide as the widest element (maybe?)
        self.regionsTable.resizeColumnsToContents()
        # start with no table cell selected, and the user can then click to select:
        self.regionsTable.setCurrentCell(-1,-1)

        # allow regionsTable to emit "cellChanged" signals now that we're done
        #   updating its content programmatically:
        self.regionsTable.blockSignals(False)

        # print "___ - DEBUG ----- CDMainTableClass: populateTableWithRegionsDict() : done"



    # ------------------------------------------------------------------
    # populate the table widget with data from the regionsDict global
    # ------------------------------------------------------------------
    def updateRegionUseOfTableElements(self, pColor, pHowManyInUse):

        lColor = QtGui.QColor(pColor)

        # prevent regionsTable from emitting any "cellChanged" signals when
        #   updating its content programmatically:
        self.regionsTable.blockSignals(True)

        # get the regionsDict keys in order to resize the table:
        lKeys = self.getRegionsDictKeys()
       
        # the entire table might be set to hide if there are no used rows:
        lThereAreRegionRowsInUse = False

        for i in xrange(self.getRegionsDictElementCount()):

            # print "self.regionsDict[lKeys[i==",i,"]] =", self.regionsDict[lKeys[i]]
            # print "self.regionsDict[lKeys[i]][0] =", self.regionsDict[lKeys[i]][0]
            if self.regionsDict[lKeys[i]][0].rgba() == lColor.rgba() :
                # print "self.regionsDict[lKeys[i==",i,"]][3] =", self.regionsDict[lKeys[i]][3]
                self.regionsDict[lKeys[i]][3] = pHowManyInUse
                # print "self.regionsDict[lKeys[i==",i,"]][3] =", self.regionsDict[lKeys[i]][3]

                # create a QTableWidgetItem and in it place the updated region use from regionsDict:
                lItem = QtGui.QTableWidgetItem( \
                           QtCore.QString("%1").arg(self.regionsDict[lKeys[i]][3]) )
                # print "self.regionsDict[lKeys[i]][3] =", self.regionsDict[lKeys[i]][3]
                # the table item containing the text from regionsDict ought not to be selected/edited:
                lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
                # this goes to column 4 in the table:
                self.regionsTable.setItem(i, 4, lItem)

            # finally determine if the current row at index i is to be visible or not:
            if self.regionsDict[lKeys[i]][3] < 1:
                self.regionsTable.hideRow(i)
            else:
                self.regionsTable.showRow(i)
                lThereAreRegionRowsInUse = True

#         if lThereAreRegionRowsInUse is False:
#             self.regionsTable.hide()
#         else:
#             self.regionsTable.show()

        # allow regionsTable to emit "cellChanged" signals now that we're done
        #   updating its content programmatically:
        self.regionsTable.blockSignals(False)




    # ------------------------------------------------------------------
    # debugRegionDict() is a debugging aid function to print out information
    #   about the regionDict global
    # ------------------------------------------------------------------
    def debugRegionDict(self):
        lCount = self.getRegionsDictElementCount()
        lKeys = self.getRegionsDictKeys()
        print "---------------------------------------------"
        print " CDMainTableClass class, regionDict global: "
        print " "
        print " table rows =", lCount
        print " table keys =", lKeys
        print " table elements per row =", self.getRegionsDictMaxRowCount()
       
        print " table row content: "
        for i in xrange(lCount):
            print "i =                                     =", i
            print "key =                       lKeys[i]     =", lKeys[i]
            print "color =    self.regionsDict[lKeys[i]][0] =", self.regionsDict[lKeys[i]][0]
            print "region =   self.regionsDict[lKeys[i]][1] =", self.regionsDict[lKeys[i]][1]
            print "cellsize = self.regionsDict[lKeys[i]][2] =", self.regionsDict[lKeys[i]][2]
            for j in xrange(3):
                print "at i =", i, ", j =", j, "self.regionsDict[lKeys[i]][2][j] =", \
                      self.regionsDict[lKeys[i]][2][j]
            print "use =      self.regionsDict[lKeys[i]][3] =", self.regionsDict[lKeys[i]][3]
            for j in xrange( len(self.regionsDict[lKeys[i]][4]) ):
                print "at i =", i, ", j =", j, "self.regionsDict[lKeys[i]][4][j] =", \
                      self.regionsDict[lKeys[i]][4][j]
        print "---------------------------------------------"


    # ------------------------------------------------------------------
    # getRegionsDictKeys() is a helper function for populateTableWithRegionsDict(),
    #   it just returns the keys in the regionsDict global:
    # ------------------------------------------------------------------
    def getRegionsDictKeys(self):
        return self.regionsDict.keys()

    # ------------------------------------------------------------------
    # getRegionsDictElementCount() is a helper function for populateTableWithRegionsDict(),
    #   it just returns number of elements in the regionsDict global:
    # ------------------------------------------------------------------
    def getRegionsDictElementCount(self):
        return len(self.regionsDict)

    # ------------------------------------------------------------------
    # TODO TODO TODO: maybe we don't need this function???   :
    # ------------------------------------------------------------------
    # getRegionsDictMaxRowCount() is a helper function for populateTableWithRegionsDict(),
    #   it just returns the maximum number of elements in any regionsDict entry
    def getRegionsDictMaxRowCount(self):
        maxRowCount = 0
        keys = self.getRegionsDictKeys()
        # see how many regions are present at most in the table:
        for i in xrange(self.getRegionsDictElementCount()):
            howMany = len(self.regionsDict[keys[i]])
            if howMany > maxRowCount:
                maxRowCount = howMany
            # print "len(self.regionsDict[keys[i==",i,"]]) = ", howMany

        # print "___ - DEBUG ----- CDMainTableClass: getRegionsDictMaxRowCount() =", \
        #       howMany
        return maxRowCount


#
#     # ------------------------------------------------------------------
#     # handle mouse click events in table elements:
#     # ------------------------------------------------------------------
#     def handleTableItemSelectionChanged(self):
#         lSelectionModel = self.regionsTable.selectionModel()
#         print "___ - DEBUG ----- CDMainTableClass: handleTableItemSelectionChanged() lSelectionModel =" , lSelectionModel, " done."
#
#         # TODO if any action is necessary
#         #
#         #if len(lSelectionModel.selectedRows()):
#         #    self.deleteCellTypeButton.setEnabled(True)
#         #else:
#         #    self.deleteCellTypeButton.setEnabled(False)


    # ------------------------------------------------------------------
    # generate an icon containing a rectangle in a specified color
    # ------------------------------------------------------------------
    def createColorIcon(self, color):
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        painter.fillRect(QtCore.QRect(0, 0, 32, 32), color)
        painter.end()
        return QtGui.QIcon(pixmap)


    # ------------------------------------------------------------------
    # this is a slot method to handle "content change" events (AKA signals)
    #   arriving from the table's "plain type" items
    #   (this method is not called when changes occur in table cells built with setCellWidget() )
    # ------------------------------------------------------------------
    def handleTableCellChanged(self,pRow,pColumn):

        print "___ - DEBUG ----- CDMainTableClass: handleTableCellChanged() pRow,pColumn =" , pRow,pColumn
        # update the dict:
        self.updateRegionsDict()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))

    # ------------------------------------------------------------------
    # this is a slot method to handle "content change" events (AKA signals)
    #   arriving from table cells built with setCellWidget() :
    # ------------------------------------------------------------------
    def handleOneRegionTableWidgetChanged(self):

        # update the dict:
        self.updateRegionsDict()

        # propagate the signal upstream, for example to parent objects:
        self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))

        print "___ - DEBUG ----- CDMainTableClass: handleOneRegionTableWidgetChanged() done."



    # TODO TODO TODO: is this still necessary???

    def on_regionsTable_cellClicked(self, row, col):
        print "___ - DEBUG ----- CDMainTableClass: on_regionsTable_cellClicked() = ", self.regionsTable.rowCount()
        # Only color cell can be changed
        if col == 0:
            # Stores the current type
            #self.__curType = self.regionsTable.item(row, col).data(Qt.DisplayRole).toInt()[0]
            #print "On click before: %s" % self.__curType
            return

        currentColor = self.regionsTable.item(row, col).background().color()
        color = QColorDialog.getColor(currentColor)
        # Cell types are not changed at this time

        if color.isValid():
            keys = self.regionsDict.keys()
            self.regionsDict[keys[row]] = color
            self.regionsTable.item(row, col).setBackground(QBrush(color))

        print "___ - DEBUG ----- CDMainTableClass: on_regionsTable_cellClicked()  done."





# end of class CDMainTableClass(QtGui.QWidget)
# ======================================================================






# ======================================================================
# a QWidget-based control panel, in application-specific panel style
# ======================================================================
# note: this class emits a signal:
#
#         self.emit(QtCore.SIGNAL("regionsTableChangedSignal()"))
#
class CDSceneAssistant(QtGui.QWizard):

    def __init__(self, pParent=None):
        # it is compulsory to call the parent's __init__ class right away:
        super(CDSceneAssistant, self).__init__(pParent)

        # don't show this widget until it's completely ready:
        self.hide()
        

        # 2011 - Mitja: set name for assistant/wizard window
        #   correctly according to platform:
        if sys.platform=='win32':
            self.setWindowTitle("CellDraw - New Scene wizard")
        else:
            self.setWindowTitle("CellDraw - New Scene assistant")

        # save the parent, whatever that may be:
        lParent = pParent

        # default values for new cell types and new region types:
        # when defining cell type size by linear dimensions:
        self.cellDimX = 5
        self.cellDimY = 10
        self.cellDimZ = 1
        # when defining cell type size by target volume:
        self.cellVolume = 50
        # 2011 - Mitja: pick a color/CC3D Type for the cell,
        #    with a variable keeping track of what is to be used:
        self.chosenCC3DColor = 0
        self.chosenCC3DType = str("ECM")
        self.chosenCellName = str("newCellName")
        # should new cell type sizes be defined by linear dimensions X-Y-Z or by volume:
        self.cellSizeMode = CDConstants.PIFFSaveWithOneRasterPerRegion

        # init - create region dict, populate it with empty values:
        self.theCellTypeDict = dict()

        #
        # init - windowing GUI stuff:
        #
        # self.miInitGUI()


    
    # ----------------------------------------------------------------
    # createIntroPage()
    # ----------------------------------------------------------------
    def createIntroPage(self, pAdditionalWidget):
        page = QtGui.QWizardPage()
        page.setTitle("CellDraw - Scene Assistant")

        label = QtGui.QLabel("The scene assistant helps you choosing "
                "cell types and cell regions.")
        label.setWordWrap(True)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(pAdditionalWidget)
        page.setLayout(layout)

        return page


    # ----------------------------------------------------------------
    # createCellTypePage()
    # ----------------------------------------------------------------
    def createCellTypePage(self, pTheColorDict):
        lCellTypePage = QtGui.QWizardPage()
        lCellTypePage.setTitle("Cell Types")
        lCellTypePage.setSubTitle("Add the cell types to be used in the scene.")
        lCellTypeLayout = QtGui.QVBoxLayout()
        lCellTypeLayout.setMargin(16)
        lCellTypeLayout.setSpacing(32)
        lCellTypeLayout.setAlignment( QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lCellTypePage.setLayout(lCellTypeLayout)

        self.cellColorDict = pTheColorDict

        CDConstants.printOut( " " , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        CDConstants.printOut( "self.cellColorDict.keys() = " +str(self.cellColorDict.keys()), CDConstants.DebugAll )
        for lKey in self.cellColorDict.keys():
            CDConstants.printOut( "lKey = " + str(lKey), CDConstants.DebugAll )
            CDConstants.printOut( "lColor = " + str(self.cellColorDict[lKey]), CDConstants.DebugAll )

        # as defaut cell type name and color,
        #   pick the first key/color pair in the self.cellColorDict:
        self.chosenCC3DType = sorted(self.cellColorDict.keys(), key=self.keynat)[0]
        self.chosenCC3DColor = QtGui.QColor(self.cellColorDict[self.chosenCC3DType])

        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        CDConstants.printOut( "self.chosenCC3DType = " + str(self.chosenCC3DType), CDConstants.DebugAll )
        CDConstants.printOut( "self.chosenCC3DColor = " + str(self.chosenCC3DColor), CDConstants.DebugAll )
        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        CDConstants.printOut( " " , CDConstants.DebugAll )

        # ----------------------------------------------------------------
        # add a QGroupBox for selecting Cell Name and CC3D Type:
        #
        self.cellNameAndTypeGroupBox = QtGui.QGroupBox("New Cell Type")
        self.cellNameAndTypeGroupBox.setLayout(QtGui.QVBoxLayout())
        self.cellNameAndTypeGroupBox.layout().setMargin(2)
        self.cellNameAndTypeGroupBox.layout().setSpacing(4)
        self.cellNameAndTypeGroupBox.layout().setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lFont = QtGui.QFont()
        lFont.setWeight(QtGui.QFont.Light)
        cellNameLabel = QtGui.QLabel()
        cellNameLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        cellNameLabel.setMargin(0)
        cellNameLabel.setFont(lFont)
        cellNameLabel.setText("Cell Name:")
        self.cellNameLineEdit = QtGui.QLineEdit(self.chosenCellName)
        self.cellNameLineEdit.editingFinished.connect(self.setCellName)
        # -----------------------------
        # the cellCC3DColorToolButton is a pop-up menu button:
        cellCC3DColorLabel = QtGui.QLabel()
        cellCC3DColorLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        cellCC3DColorLabel.setMargin(0)
        cellCC3DColorLabel.setFont(lFont)
        cellCC3DColorLabel.setText("CC3D Type:")
        self.cellCC3DColorToolButton = QtGui.QToolButton()
        self.cellCC3DColorToolButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.cellCC3DColorToolButton.setText(self.chosenCC3DType)
        self.cellCC3DColorToolButton.setIcon( \
            self.createFloodFillToolButtonIcon( \
                ':/icons/floodfill.png', \
                self.chosenCC3DColor   )   )
        self.cellCC3DColorToolButton.setIconSize(QtCore.QSize(24, 24))
        self.cellCC3DColorToolButton.setToolTip("Set Cell Type")
        self.cellCC3DColorToolButton.setStatusTip("Set the Cell CC3D Type")
        self.cellCC3DColorToolButton.setPopupMode(QtGui.QToolButton.MenuButtonPopup)
        # attach a popup menu to the button, with event handler handleCellCC3DTypeMenuChanged()
        self.cellCC3DColorToolButton.setMenu(  \
            self.createColorMenu(self.handleCellCC3DTypeMenuChanged, \
            self.chosenCC3DColor)  )
        self.cellCC3DColorToolButton.menu().setToolTip("Set Cell Type")
        self.cellCC3DColorToolButton.menu().setStatusTip("Set the Cell CC3D Type")
        self.fillAction = self.cellCC3DColorToolButton.menu().defaultAction()
        # provide a "slot" function to the button, the event handler is handleCellCC3DTypeButtonClicked()
        self.cellCC3DColorToolButton.clicked.connect(self.handleCellCC3DTypeButtonClicked)
        # -----------------------------
        cellNameAndTypeLayout = QtGui.QGridLayout()
        cellNameAndTypeLayout.setMargin(0)
        cellNameAndTypeLayout.setSpacing(4)
        cellNameAndTypeLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        cellNameAndTypeLayout.addWidget(cellNameLabel, 0, 0)
        cellNameAndTypeLayout.addWidget(self.cellNameLineEdit, 0, 1)
        cellNameAndTypeLayout.addWidget(cellCC3DColorLabel, 1, 0)
        cellNameAndTypeLayout.addWidget(self.cellCC3DColorToolButton, 1, 1)
        self.cellNameAndTypeGroupBox.layout().addLayout(cellNameAndTypeLayout)
        #
        # end of QGroupBox for selecting Cell Name and CC3D Type.
        # ----------------------------------------------------------------



        # ----------------------------------------------------------------
        # add a QGroupBox for selecting cell dimensions:
        #
        self.cellDimensionsGroupBox = QtGui.QGroupBox("New Cell Size")
        self.cellDimensionsGroupBox.setLayout(QtGui.QGridLayout())
        self.cellDimensionsGroupBox.layout().setMargin(2)
        self.cellDimensionsGroupBox.layout().setSpacing(4)
        self.cellDimensionsGroupBox.layout().setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # -----------------------------
        # prepare two radio buttons, one for each distinct "cell size" mode,
        #    and the linear dimensions is set as the default.
        #
        # cell size QPushButton - linear dimensions X-Y-Z choice:
        self.linearDimensionsCellSizeButton = QtGui.QRadioButton("linear dimensions")
        self.linearDimensionsCellSizeButton.setCheckable(True)
        self.linearDimensionsCellSizeButton.setChecked(True)
        self.linearDimensionsCellSizeButton.setToolTip("Cell Size - defined by linear dimensions")
        self.linearDimensionsCellSizeButton.setStatusTip("Cell Size - defined by linear dimensions")
        # cell size QPushButton - volume choice:
        self.volumeDimensionsCellSizeButton = QtGui.QRadioButton("volume")
        self.volumeDimensionsCellSizeButton.setCheckable(True)
        self.volumeDimensionsCellSizeButton.setChecked(False)
        self.volumeDimensionsCellSizeButton.setToolTip("Cell Size - defined by volume")
        self.volumeDimensionsCellSizeButton.setStatusTip("Cell Size - defined by volume")
        #
        # a QButtonGroup - a *logical* container to make buttons mutually exclusive:
        self.theButtonGroupForCellSize = QtGui.QButtonGroup()
        self.theButtonGroupForCellSize.addButton(self.linearDimensionsCellSizeButton, CDConstants.PIFFSaveWithOneRasterPerRegion)
        self.theButtonGroupForCellSize.addButton(self.volumeDimensionsCellSizeButton, CDConstants.PIFFSaveWithPotts)
        # call handleCellSizeButtonGroupClicked() every time a button is clicked in the "theButtonGroupForCellSize":
        self.theButtonGroupForCellSize.buttonClicked[int].connect( \
            self.handleCellSizeButtonGroupClicked)
        # -----------------------------
        # prepare labels and line edits for cell size, and the target volume-related are disabled:
        self.cellDimensionXLabel    = QtGui.QLabel("X:")
        self.cellDimensionXLineEdit = self.createCellSizeLineEdit(str(self.cellDimX), self.setCellDimX)
        self.cellDimensionYLabel    = QtGui.QLabel("Y:")
        self.cellDimensionYLineEdit = self.createCellSizeLineEdit(str(self.cellDimY), self.setCellDimY)
        self.cellDimensionZLabel    = QtGui.QLabel("Z:")
        self.cellDimensionZLineEdit = self.createCellSizeLineEdit(str(self.cellDimZ), self.setCellDimZ)
        self.cellTargetVolumeLabel = QtGui.QLabel("Target Volume:")
        self.cellTargetVolumeLabel.setEnabled(False)
        self.cellTargetVolumeLineEdit = self.createCellSizeLineEdit(str(self.cellVolume), self.setCellVolume, "000000000")
        self.cellTargetVolumeLineEdit.setEnabled(False)
        # -----------------------------
        self.linearDimensionsLayout = QtGui.QHBoxLayout()
        self.linearDimensionsLayout.setMargin(0)
        self.linearDimensionsLayout.setSpacing(4)
        self.linearDimensionsLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.linearDimensionsLayout.addWidget(self.cellDimensionXLabel)
        self.linearDimensionsLayout.addWidget(self.cellDimensionXLineEdit)
        self.linearDimensionsLayout.addWidget(self.cellDimensionYLabel)
        self.linearDimensionsLayout.addWidget(self.cellDimensionYLineEdit)
        self.linearDimensionsLayout.addWidget(self.cellDimensionZLabel)
        self.linearDimensionsLayout.addWidget(self.cellDimensionZLineEdit)
        self.linearDimensionsLayout.setEnabled(True)
        # -----------------------------
        self.targetVolumeLayout = QtGui.QHBoxLayout()
        self.targetVolumeLayout.setMargin(0)
        self.targetVolumeLayout.setSpacing(4)
        self.targetVolumeLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.targetVolumeLayout.addWidget(self.cellTargetVolumeLabel)
        self.targetVolumeLayout.addWidget(self.cellTargetVolumeLineEdit)
        self.linearDimensionsLayout.setEnabled(True)
        # -----------------------------
        self.cellDimensionsGroupBox.layout().addWidget(self.linearDimensionsCellSizeButton, 0, 0)
        self.cellDimensionsGroupBox.layout().addWidget(self.volumeDimensionsCellSizeButton, 1, 0)
        self.cellDimensionsGroupBox.layout().addLayout(self.linearDimensionsLayout, 0, 1)
        self.cellDimensionsGroupBox.layout().addLayout(self.targetVolumeLayout, 1, 1)
        #
        # end of QGroupBox for selecting cell dimensions.
        # ----------------------------------------------------------------



        # ----------------------------------------------------------------
        # add a QGroupBox to contain the cell type table:
        #
        self.cellTypeTableGroupBox = QtGui.QGroupBox("Cell Types in new CellDraw Scene")
        self.cellTypeTableGroupBox.setLayout(QtGui.QVBoxLayout())
        self.cellTypeTableGroupBox.layout().setMargin(2)
        self.cellTypeTableGroupBox.layout().setSpacing(4)
        self.cellTypeTableGroupBox.layout().setAlignment( \
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.cellTypeTableGroupBox.setMinimumHeight(240)

        lRowCount = 0
        self.cellTypeTable = QtGui.QTableWidget()
        self.cellTypeTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.cellTypeTable.setShowGrid(True)
        self.cellTypeTable.setWordWrap(False)
        self.cellTypeTable.setCornerButtonEnabled(True)
        self.cellTypeTable.setRowCount(lRowCount)
        self.cellTypeTable.setColumnCount(7)
        self.cellTypeTable.setHorizontalHeaderLabels( ("CC3D\nColor", "CC3D\nType", "Name", "x", "y", "z", "Target\nVolume") )
        self.cellTypeTable.horizontalHeader().setVisible(True)
        self.cellTypeTable.horizontalHeader().setHighlightSections(True)
        self.cellTypeTable.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        self.cellTypeTable.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.ResizeToContents)
        self.cellTypeTable.horizontalHeader().setResizeMode(2, QtGui.QHeaderView.ResizeToContents)
        self.cellTypeTable.horizontalHeader().setResizeMode(3, QtGui.QHeaderView.ResizeToContents)
        self.cellTypeTable.horizontalHeader().setResizeMode(4, QtGui.QHeaderView.ResizeToContents)
        self.cellTypeTable.horizontalHeader().setResizeMode(5, QtGui.QHeaderView.ResizeToContents)
        self.cellTypeTable.horizontalHeader().setResizeMode(6, QtGui.QHeaderView.ResizeToContents)
        self.cellTypeTable.horizontalHeader().setStretchLastSection(False)
        self.cellTypeTable.verticalHeader().setVisible(True)
        self.cellTypeTable.verticalHeader().setHighlightSections(True)
        self.cellTypeTable.verticalHeader().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        self.cellTypeTable.verticalHeader().setSortIndicatorShown(True)
        self.cellTypeTable.verticalHeader().setStretchLastSection(False)
        self.cellTypeTable.setLineWidth(1)
        self.cellTypeTable.setMidLineWidth(1)
        self.cellTypeTable.setFrameShape(QtGui.QFrame.Panel)
        self.cellTypeTable.setFrameShadow(QtGui.QFrame.Plain)

        self.cellTypeTableGroupBox.layout().addWidget(self.cellTypeTable)
        #
        # end of QGroupBox to contain the cell type table.
        # ----------------------------------------------------------------
    
    
    
    
        # a separate "QDialogButtonBox" widget, to have "add" and "remove" buttons
        #   that allow the user to add/remove lines from the oneRegionTable:
        addCellTypeTableRowButton = QtGui.QPushButton("+")
        removeCellTypeTableRowButton = QtGui.QPushButton("-")
        self.addCellTypeToTableButtonBox = QtGui.QDialogButtonBox()
        self.addCellTypeToTableButtonBox.addButton(addCellTypeTableRowButton, QtGui.QDialogButtonBox.AcceptRole)
        self.addCellTypeToTableButtonBox.addButton(removeCellTypeTableRowButton, QtGui.QDialogButtonBox.RejectRole)
        # connects signals from buttons to "slot" methods:
        self.addCellTypeToTableButtonBox.accepted.connect(self.handleAddCellTypeTableRow)
        self.addCellTypeToTableButtonBox.rejected.connect(self.handleRemoveCellTypeTableRow)

        lCellTypeLayout.addStretch(64)
        lCellTypeLayout.addWidget(self.cellTypeTableGroupBox)
        lCellTypeLayout.addStretch(32)
        lCellTypeLayout.addWidget(self.cellNameAndTypeGroupBox)
        lCellTypeLayout.addStretch(32)
        lCellTypeLayout.addWidget(self.cellDimensionsGroupBox)
        lCellTypeLayout.addStretch(32)
        lCellTypeLayout.addWidget(self.addCellTypeToTableButtonBox)
        lCellTypeLayout.addStretch(64)

        return lCellTypePage
    
    

    # ------------------------------------------------------------------
    # remove the last row from the table, if there is more than one row:
    # ------------------------------------------------------------------
    def handleRemoveCellTypeTableRow(self):
        # get the oneRegionTable keys in order to resize the table:
        lKeys = self.theCellTypeDict.keys()
        lLength = len(self.theCellTypeDict)
        # don't remove the last row in the dict:
        if (lLength > 1):
            lLastKey = lKeys[lLength-1]
            # remove the last entry in the theCellTypeDict:
            del self.theCellTypeDict[lLastKey]
            # update the table:
            self.populateOneRegionSubTable()

        # propagate the change upstream by emitting a signal, for example to parent objects:
        self.emit(QtCore.SIGNAL("oneRegionTableChangedSignal()"))

        # print "___ - DEBUG ----- PIFOneRegionTable: handleRemoveCellTypeTableRow() : done"

    # ------------------------------------------------------------------
    # add a row to the table:
    # ------------------------------------------------------------------
    def handleAddCellTypeTableRow(self):
    
        # if one of the QLineEdit had focus, its text may not be saved.
        # therefore obtain the current "focusWidget" and force its update.

        # get the oneRegionTable keys in order to resize the table:
        lCellTypeDictKeys = self.theCellTypeDict.keys()
        lLength = len(self.theCellTypeDict)
        if (lLength == 0):
            lNewKey = 1
        else:
            # generate a new key for theCellTypeDict (just a sequential number)
            lNewKey = lCellTypeDictKeys[lLength-1] + 1

        # printout the entire cell type dict, for debugging:
        lNameOrTypeAlreadyPresent = False
        for lKey in lCellTypeDictKeys:
            # avoid duplicate CC3D types:
            if (self.theCellTypeDict[lKey][1] == self.chosenCC3DType):
                lNameOrTypeAlreadyPresent = True
            # avoid duplicate cell names:
            if (self.theCellTypeDict[lKey][2] == self.chosenCellName):
                lNameOrTypeAlreadyPresent = True
        # if either cell type or cell name are already present,
        #   don't add to table but return!
        if (lNameOrTypeAlreadyPresent == True):
            QtGui.QMessageBox.warning(self, "CellDraw",
                    "The Cell Name or CC3D Type\n" + \
                    "are already in the scene.")
            return False

        # all is good - the new cell type can be added to the dict and the table.
        # generate a new theCellTypeDict entry:
        self.theCellTypeDict[lNewKey] = [ self.chosenCC3DColor, self.chosenCC3DType, \
            self.chosenCellName, self.cellDimX, self.cellDimY, self.cellDimZ, self.cellVolume]

        # printout the entire cell type dict, for debugging:
        CDConstants.printOut( " " , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        lCellTypeDictKeys = self.theCellTypeDict.keys()
        for lKey in lCellTypeDictKeys:
            CDConstants.printOut( "lNewKey = " + str(lKey) +
                " self.theCellTypeDict[lKey] = " + str(self.theCellTypeDict[lKey]) ,
                CDConstants.DebugAll )
        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        CDConstants.printOut( "-=-=--=-=--=-=--=-=--=-=--=-=--=-=--=-=-" , CDConstants.DebugAll )
        CDConstants.printOut( " " , CDConstants.DebugAll )

        self.populateCellTypeTable()

        # print "___ - DEBUG ----- PIFOneRegionTable: handleAddCellTypeTableRow() done."
        
    # ------------------------------------------------------------------
    # populate the cell type table with data from theCellTypeDict
    # ------------------------------------------------------------------
    def populateCellTypeTable(self):

        # cellTypeTable shouldn't emit "cellChanged" signals when updated programmatically:
        self.cellTypeTable.blockSignals(True)

        # set how many rows are needed in the table:
        lTypesCount = len(self.theCellTypeDict)
        self.cellTypeTable.setRowCount(lTypesCount)

        # get the theCellTypeDict keys in order to resize the table:
        lKeys = self.theCellTypeDict.keys()

        for i in xrange(lTypesCount):
            lKey = lKeys[i]

            # create a first QTableWidgetItem and place a swatch in it:
            lItem = QtGui.QTableWidgetItem()
            # (setting the background color like this would not be a good idea,
            #   for example because it can be confused with selection highlight colors:
            #   lItem.setBackground(QtGui.QBrush(self.theCellTypeDict[lKey][0]))
            # this way is much better, it generates a rectangle in the same color:
            lItem.setIcon(  self.createColorIcon( self.theCellTypeDict[lKey][0] )  )
            # the color item shouldn't be selectable/editable:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 0 in the table:
            self.cellTypeTable.setItem(i, 0, lItem)

            # create a second QTableWidgetItem and set CC3D type from theCellTypeDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.theCellTypeDict[lKey][1]) )
            # the type item shouldn't be selectable/editable:
            lItem.setFlags(lItem.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            # this goes to column 1 in the table:
            self.cellTypeTable.setItem(i, 1, lItem)

            # create a third QTableWidgetItem and set Cell Name from theCellTypeDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.theCellTypeDict[lKey][2]) )
            # this goes to column 2 in the table:
            self.cellTypeTable.setItem(i, 2, lItem)

            # create a fourth QTableWidgetItem and set cell X size from theCellTypeDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.theCellTypeDict[lKey][3]) )
            # this goes to column 3 in the table:
            self.cellTypeTable.setItem(i, 3, lItem)

            # create a fifth QTableWidgetItem and set cell Y size from theCellTypeDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.theCellTypeDict[lKey][4]) )
            # this goes to column 4 in the table:
            self.cellTypeTable.setItem(i, 4, lItem)

            # create a sixth QTableWidgetItem and set cell Z size from theCellTypeDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.theCellTypeDict[lKey][5]) )
            # this goes to column 5 in the table:
            self.cellTypeTable.setItem(i, 5, lItem)

            # create a seventh QTableWidgetItem and set cell volume from theCellTypeDict in it:
            lItem = QtGui.QTableWidgetItem( \
                       QtCore.QString("%1").arg(self.theCellTypeDict[lKey][6]) )
            # this goes to column 6 in the table,
            self.cellTypeTable.setItem(i, 6, lItem)

            self.cellTypeTable.verticalHeader().setResizeMode(i, QtGui.QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def createRegionTypePage(self):
        page = QtGui.QWizardPage()
        page.setTitle("Cell Regions")
        page.setSubTitle("Add the cell region types you want to use in your scene.")
    
        cellRegionNameLabel = QtGui.QLabel("Region Name:")
        cellRegionNameLineEdit = QtGui.QLineEdit("newRegion")
    
        cellInitialCellLabel = QtGui.QLabel("Initial cell generation:")
        cellInitialCellLineEdit = QtGui.QLineEdit()
    
        lRowCount = 20
        self.cellRegionTable = QtGui.QTableWidget()
        self.cellRegionTable.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.cellRegionTable.setShowGrid(True)
        self.cellRegionTable.setWordWrap(False)
        self.cellRegionTable.setCornerButtonEnabled(True)
        self.cellRegionTable.setRowCount(lRowCount)
        self.cellRegionTable.setColumnCount(5)
        self.cellRegionTable.setHorizontalHeaderLabels( ("Color", "Cell Type", "Amount", "Fraction", "Target Volume") )
        self.cellRegionTable.horizontalHeader().setVisible(True)
        self.cellRegionTable.horizontalHeader().setHighlightSections(True)
        self.cellRegionTable.verticalHeader().setVisible(True)
        self.cellRegionTable.verticalHeader().setHighlightSections(True)
        self.cellRegionTable.verticalHeader().setSortIndicatorShown(True)
        self.cellRegionTable.verticalHeader().setStretchLastSection(True)

        regionDataEntryLayout = QtGui.QGridLayout()
        regionDataEntryLayout.addWidget(cellRegionNameLabel, 0, 0)
        regionDataEntryLayout.addWidget(cellRegionNameLineEdit, 0, 1)
        regionDataEntryLayout.addWidget(cellInitialCellLabel, 1, 0)
        regionDataEntryLayout.addWidget(cellInitialCellLineEdit, 1, 1)

        # a separate "QDialogButtonBox" widget, to have "add" and "remove" buttons
        #   that allow the user to add/remove lines from the oneRegionTable:
        addTableRowButton = QtGui.QPushButton("+")
        removeTableRowButton = QtGui.QPushButton("-")
        self.addRegionButtonBox = QtGui.QDialogButtonBox()
        self.addRegionButtonBox.addButton(addTableRowButton, QtGui.QDialogButtonBox.AcceptRole)
        self.addRegionButtonBox.addButton(removeTableRowButton, QtGui.QDialogButtonBox.RejectRole)
        # connects signals from buttons to "slot" methods:
        self.addRegionButtonBox.accepted.connect(self.handleAddCellTypeTableRow)
        self.addRegionButtonBox.rejected.connect(self.handleRemoveCellTypeTableRow)
        


    
        label = QtGui.QLabel("Add Cell Types to New Regions")
        label.setWordWrap(True)

#         layout = QtGui.QVBoxLayout()
#         layout.addWidget(label)
#         page.setLayout(layout)


        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.cellRegionTable)
        layout.addLayout(regionDataEntryLayout)
        layout.addWidget(self.addRegionButtonBox)
        page.setLayout(layout)
        
        return page
    


    # ------------------------------------------------------------------
    # 2010 - Mitja: assign CellDraw preferences object:
    # ------------------------------------------------------------------
    def setPreferencesObject(self, pCDPreferences=None):
        self.cdPreferences = pCDPreferences
        CDConstants.printOut( ">>>>>>>>>>>>>>>>>>>>>>>> CDSceneRasterizer.cdPreferences is now =" + str(self.cdPreferences), CDConstants.DebugVerbose )


    # ---------------------------------------------------------
    # set up paths to access CC3D as a subprocess:
    # ---------------------------------------------------------
    def setupPathsToCC3D(self):

        # ---------------------------------------
        # 2011 - Mitja: add calling CC3D as subprocess:
        
        # make sure that preferences obtained from CC3D settings file are up to date:
        self.cdPreferences.readCC3DPreferencesFromDisk()

        self.cc3dPath = str(self.cdPreferences.cc3dCommandPathCC3D)
        self.cc3dPathAndStartupFileName = str(self.cdPreferences.cc3dCommandPathAndStartCC3D)
        
        self.cc3dOutputLocationPath = str(self.cdPreferences.outputLocationPathCC3D)


   
    # end of def setupPathsToCC3D(self)
    # ---------------------------------------------------------


    def createSceneAssistant(self):


    
        wizard = CDSceneAssistant()
    
    
        # connect theSceneRasterizerWidget to the only instance of the cdPreferences object:
        #    read some persistent-value globals from the preferences file on disk, if it already exists.
        cdPreferences = CDPreferences()
        cdPreferences.hide()
        cdPreferences.readPreferencesFromDisk()
        cdPreferences.readCC3DPreferencesFromDisk()
        lTheColorTable = cdPreferences.populateCellColors()
    
    
        wizard.setPreferencesObject(cdPreferences)
    
        wizard.addPage(wizard.createIntroPage(lTheColorTable))
        wizard.addPage(wizard.createCellTypePage())
        wizard.addPage(wizard.createRegionTypePage())
    
        wizard.setWindowTitle("CellDraw - Scene Assistant")
        wizard.show()
        wizard.raise_()
    
    
    # ------------------------------------------------------------
    # 2011 - Mitja: creating a "Color" menu produces a QMenu item,
    #    which allows the creation of a color-picking menu:
    # ------------------------------------------------------------
    def createColorMenu(self, pSlotFunction, pDefaultColor):

        lColorMenu = QtGui.QMenu(self)
        
        for lKey in sorted(self.cellColorDict.keys(), key=self.keynat):
            lColor = QtGui.QColor(self.cellColorDict[lKey])
            lName = lKey
            # do not include the CC3D "medium" type among possible choices:
            if (str(lName) != "ECM") :
    #         for lColor, lName in zip(self.colors, self.names):
                # print "lColor =", lColor, "lName =", lName
                lAction = QtGui.QAction(self.createColorIcon(lColor),
                          lName, self, triggered=pSlotFunction)
                # set the action's data to be the color:
                lAction.setData(QtGui.QColor(lColor))
                lColorMenu.addAction(lAction)
                if lColor == pDefaultColor:
                    lColorMenu.setDefaultAction(lAction)
        return lColorMenu
    # ------------------------------------------------------------
    def keynat(self, string):
        r'''A natural sort helper function for sort() and sorted()
        without using regular expression.
    
        >>> items = ('Z', 'a', '10', '1', '9')
        >>> sorted(items)
        ['1', '10', '9', 'Z', 'a']
        >>> sorted(items, key=keynat)
        ['1', '9', '10', 'Z', 'a']
        '''
        r = []
        for c in string:
            try:
                c = int(c)
                try: r[-1] = r[-1] * 10 + c
                except: r.append(c)
            except:
                r.append(c)
        return r


    # ------------------------------------------------------------
    # 2011 - Mitja: create an icon for a color item:
    # ------------------------------------------------------------
    def createColorIcon(self, color):
        pixmap = QtGui.QPixmap(20, 20)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtCore.Qt.NoPen)
        painter.fillRect(QtCore.QRect(0, 0, 20, 20), color)
        painter.end()

        return QtGui.QIcon(pixmap)

    # ------------------------------------------------------------
    # 2011 - Mitja: create an icon for a menu item:
    # ------------------------------------------------------------
    def createFloodFillToolButtonIcon(self, pImageFile=None, pColor=None):
        if (pColor == None):
            lColor = QtGui.QColor(QtCore.Qt.green)
        else:
            lColor = pColor
        lPixmap = QtGui.QPixmap(80, 80)
        lPixmap.fill(QtCore.Qt.transparent)
        lPainter = QtGui.QPainter(lPixmap)
        if (pImageFile != None):
            lImage = QtGui.QPixmap(pImageFile)
        lTarget = QtCore.QRect(0, 0, 60, 60)
        lSource = QtCore.QRect(0, 0, 44, 44)
        # print "OOOOOOOOOOOOOO in createFloodFillToolButtonIcon() lColor =", lColor
        lPainter.fillRect(QtCore.QRect(0, 60, 80, 80), lColor)
        if (pImageFile != None):
            lPainter.drawPixmap(lTarget, lImage, lSource)
        lPainter.end()
        return QtGui.QIcon(lPixmap)

    # ---------------------------------------------------------
    def createCellSizeLineEdit(self, pText, pCallback, pMask=False):
        lineEdit = QtGui.QLineEdit(pText)
        if (pMask != False):
            lineEdit.setInputMask(pMask)
        else:
            lineEdit.setInputMask("000")
        lineEdit.editingFinished.connect(pCallback)
        return lineEdit

    # ---------------------------------------------------------
    def setCellDimX(self):
        self.cellDimX = int(self.cellDimensionXLineEdit.text())
        # update the volume too:
        self.cellVolume = self.cellDimX * self.cellDimY * self.cellDimZ
        self.cellTargetVolumeLineEdit.setText(str(self.cellVolume))

    # ---------------------------------------------------------
    def setCellDimY(self):
        self.cellDimY = int(self.cellDimensionYLineEdit.text())
        # update the volume too:
        self.cellVolume = self.cellDimX * self.cellDimY * self.cellDimZ
        self.cellTargetVolumeLineEdit.setText(str(self.cellVolume))

    # ---------------------------------------------------------
    def setCellDimZ(self):
        self.cellDimZ = int(self.cellDimensionZLineEdit.text())
        # update the volume too:
        self.cellVolume = self.cellDimX * self.cellDimY * self.cellDimZ
        self.cellTargetVolumeLineEdit.setText(str(self.cellVolume))

    # ---------------------------------------------------------
    def setCellVolume(self):
        self.cellVolume = int(self.cellTargetVolumeLineEdit.text())
        # update the linear dimensions too:
        self.cellDimX = int( math.sqrt(self.cellVolume) )
        self.cellDimY = int( math.sqrt(self.cellVolume) )
        self.cellDimZ = 1
        self.cellDimensionXLineEdit.setText(str(self.cellDimX))
        self.cellDimensionYLineEdit.setText(str(self.cellDimY))
        self.cellDimensionZLineEdit.setText(str(self.cellDimZ))

    # ---------------------------------------------------------
    def setCellName(self):
        # store the user-chosen cell name as Python str:
        self.chosenCellName = str(self.cellNameLineEdit.text())


    # ------------------------------------------------------------
    # 2011 Mitja - slot method handling "triggered" events
    #    (AKA signals) arriving from the cellCC3DColorToolButton menu:
    # ------------------------------------------------------------
    def handleCellCC3DTypeMenuChanged(self):

        self.fillAction = self.sender()
        self.chosenCC3DColor = self.fillAction.data()
        self.chosenCC3DType = self.fillAction.text()
        # print "self.chosenCC3DColor is now", self.chosenCC3DColor, "not", QtGui.QColor(self.chosenCC3DColor)
       
        self.cellCC3DColorToolButton.setIcon( \
            self.createFloodFillToolButtonIcon(':/icons/floodfill.png', \
            QtGui.QColor(self.chosenCC3DColor)  )  )
        self.cellCC3DColorToolButton.setText(str(self.chosenCC3DType))
        self.cellCC3DColorToolButton.setIconSize(QtCore.QSize(24, 24))

        self.chosenCC3DColor = QtGui.QColor(self.chosenCC3DColor)
        self.chosenCC3DType = str(self.chosenCC3DType)

    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "clicked" events
    #    (AKA signals) arriving from the cellCC3DColorToolButton button:
    # ------------------------------------------------------------
    def handleCellCC3DTypeButtonClicked(self):
        # there is no change in chosen color when the button is clicked:
        self.chosenCC3DColor = QtGui.QColor(self.chosenCC3DColor)
        self.chosenCC3DType = str(self.chosenCC3DType)



    # ------------------------------------------------------------
    # 2010 Mitja - slot method handling "buttonClicked" events
    #    (AKA signals) arriving from theButtonGroupForCellSize:
    # ------------------------------------------------------------
    def handleCellSizeButtonGroupClicked(self, pChecked):
        if self.linearDimensionsCellSizeButton.isChecked():
            lCellSizeMode = CDConstants.PIFFSaveWithOneRasterPerRegion
            self.cellDimensionXLabel.setEnabled(True)
            self.cellDimensionXLineEdit.setEnabled(True)
            self.cellDimensionYLabel.setEnabled(True)
            self.cellDimensionYLineEdit.setEnabled(True)
            self.cellDimensionZLabel.setEnabled(True)
            self.cellDimensionZLineEdit.setEnabled(True)
            self.cellTargetVolumeLabel.setEnabled(False)
            self.cellTargetVolumeLineEdit.setEnabled(False)
        elif self.volumeDimensionsCellSizeButton.isChecked():
            lCellSizeMode = CDConstants.PIFFSaveWithPotts
            self.cellDimensionXLabel.setEnabled(False)
            self.cellDimensionXLineEdit.setEnabled(False)
            self.cellDimensionYLabel.setEnabled(False)
            self.cellDimensionYLineEdit.setEnabled(False)
            self.cellDimensionZLabel.setEnabled(False)
            self.cellDimensionZLineEdit.setEnabled(False)
            self.cellTargetVolumeLabel.setEnabled(True)
            self.cellTargetVolumeLineEdit.setEnabled(True)
        #
        if lCellSizeMode is not self.cellSizeMode:
            self.cellSizeMode = lCellSizeMode

# end of class CDSceneAssistant(QtGui.QWidget)
# ======================================================================







# ======================================================================
# the following if statement checks whether the present file
#    is currently being used as standalone (main) program, and in this
#    class's (CDSceneAssistant) case it is simply used for testing:
# ======================================================================
if __name__ == '__main__':


    print "001 - DEBUG - mi __main__ xe 01"
    # every PyQt4 app must create an application object, from the QtGui module:
    miApp = QtGui.QApplication(sys.argv)



    wizard = CDSceneAssistant()


    # connect theSceneRasterizerWidget to the only instance of the cdPreferences object:
    #    read some persistent-value globals from the preferences file on disk, if it already exists.
    cdPreferences = CDPreferences()
    cdPreferences.hide()
    cdPreferences.readPreferencesFromDisk()
    cdPreferences.readCC3DPreferencesFromDisk()
    lTheColorTable = cdPreferences.populateCellColors()
    lTheColorDict = cdPreferences.getCellColorsDict()

    wizard.setPreferencesObject(cdPreferences)

    wizard.addPage(wizard.createIntroPage(lTheColorTable))
    wizard.addPage(wizard.createCellTypePage(lTheColorDict))
    wizard.addPage(wizard.createRegionTypePage())

    wizard.show()
    wizard.raise_()



    print "002 - DEBUG - mi __main__ xe 02"

    # the cell type/colormap panel:
    mainPanel = CDMainTableClass()

    print "003 - DEBUG - mi __main__ xe 03"

    # temporarily assign colors/names to a dictionary:
    #   place all used colors/type names into a dictionary, to be locally accessible

    miDict = dict({ 1: [ QtGui.QColor(QtCore.Qt.blue), "TestRegion", [8, 4, 2], 3, \
                         [[QtGui.QColor(QtCore.Qt.blue), "TestTypeFirst", 0.5, 100], \
                          [QtGui.QColor(QtCore.Qt.blue), "TestTypeSecond", 0.5, 80]]
                          ],
                    2: [ QtGui.QColor(QtCore.Qt.green), "SampleRegion", [9, 7, 6], 1, \
                         [[QtGui.QColor(QtCore.Qt.green), "SampleTypeFirst", 0.2, 50], \
                          [QtGui.QColor(QtCore.Qt.green), "SampleTypeSecond", 0.8, 75]]
                          ]
                          } )

    mainPanel.setRegionsDict(miDict)
    mainPanel.populateTableWithRegionsDict()

    # show() and raise_() have to be called here:
#     mainPanel.show()
    # raise_() is a necessary workaround to a PyQt-caused (or Qt-caused?) bug on Mac OS X:
    #   unless raise_() is called right after show(), the window/panel/etc will NOT become
    #   the foreground window and won't receive user input focus:
#     mainPanel.raise_()
    print "004 - DEBUG - mi __main__ xe 04"

    sys.exit(miApp.exec_())
    print "005 - DEBUG - mi __main__ xe 05"

# end if __name__ == '__main__'
